import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import deepchem as dc
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem
from rdkit.Chem import inchi, rdMolDescriptors

BOND_STRUCT = pa.struct([
    ("id",     pa.string()),
    ("atom",   pa.string()),
    ("order",  pa.float64()),
    ("stereo", pa.string()),
])

ATOM_STRUCT = pa.struct([
    ("id",                  pa.string()),
    ("element",             pa.string()),
    ("atomic_number",       pa.int32()),
    ("formal_charge",       pa.int32()),
    ("n_hydrogens",         pa.int32()),
    ("n_radical_electrons", pa.int32()),
    ("chirality",           pa.string()),
    ("aromatic",            pa.bool_()),
    ("bonds",               pa.list_(BOND_STRUCT)),
])

EDGE_STRUCT = pa.struct([
    ("id",    pa.string()),
    ("atoms", pa.list_(pa.string())),
])

STRUCTURE_STRUCT = pa.struct([
    ("nodes", pa.list_(pa.string())),
    ("edges", pa.list_(EDGE_STRUCT)),
])


def build_schema(tasks: tuple[str, ...]) -> pa.Schema:
    return pa.schema([
        ("smiles",           pa.string()),
        ("targets",          pa.struct([(t, pa.float64()) for t in tasks])),
        ("atoms",            pa.list_(ATOM_STRUCT)),
        ("structure",        STRUCTURE_STRUCT),
        ("canonical_smiles", pa.string()),
        ("inchi",            pa.string()),
        ("inchi_key",        pa.string()),
        ("formula",          pa.string()),
        ("exact_mass",       pa.float64()),
        ("net_charge",       pa.int32()),
    ])


def mol_to_graph_dict(mol: Chem.Mol) -> dict:
    """Emit list-of-struct shapes so pyarrow.to_pylist() lands in json2vec's Branch format."""
    per_atom_bonds: dict[int, list[dict]] = {a.GetIdx(): [] for a in mol.GetAtoms()}
    for b in mol.GetBonds():
        i, j   = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bkey   = f"b{b.GetIdx()}"
        order  = b.GetBondTypeAsDouble()
        stereo = b.GetStereo().name
        per_atom_bonds[i].append({"id": bkey, "atom": f"a{j}", "order": order, "stereo": stereo})
        per_atom_bonds[j].append({"id": bkey, "atom": f"a{i}", "order": order, "stereo": stereo})

    atoms = [
        {
            "id":                  f"a{a.GetIdx()}",
            "element":             a.GetSymbol(),
            "atomic_number":       a.GetAtomicNum(),
            "formal_charge":       a.GetFormalCharge(),
            "n_hydrogens":         a.GetTotalNumHs(),
            "n_radical_electrons": a.GetNumRadicalElectrons(),
            "chirality":           a.GetChiralTag().name,
            "aromatic":            a.GetIsAromatic(),
            "bonds":               per_atom_bonds[a.GetIdx()],
        }
        for a in mol.GetAtoms()
    ]

    structure = {
        "nodes": [f"a{a.GetIdx()}" for a in mol.GetAtoms()],
        "edges": [
            {"id": f"b{b.GetIdx()}",
             "atoms": [f"a{b.GetBeginAtomIdx()}", f"a{b.GetEndAtomIdx()}"]}
            for b in mol.GetBonds()
        ],
    }

    return {
        "atoms":            atoms,
        "structure":        structure,
        "canonical_smiles": Chem.MolToSmiles(mol),
        "inchi":            inchi.MolToInchi(mol),
        "inchi_key":        inchi.MolToInchiKey(mol),
        "formula":          rdMolDescriptors.CalcMolFormula(mol),
        "exact_mass":       rdMolDescriptors.CalcExactMolWt(mol),
        "net_charge":       Chem.GetFormalCharge(mol),
    }


def build_record(args: tuple) -> dict:
    """Worker: build one row dict from a (zinc_id, mol, tasks, y_row) tuple."""
    zinc_id, mol, tasks, y_row = args
    return {
        "smiles":  zinc_id,
        "targets": {t: float(v) for t, v in zip(tasks, y_row)},
        **mol_to_graph_dict(mol),
    }


def export_split(split, split_name: str, tasks_tuple: tuple[str, ...], out_dir: Path,
                 n_workers: int, chunksize: int, row_group_size: int) -> Path:
    out = out_dir / f"zinc15_{split_name}.parquet"
    n = len(split)
    schema = build_schema(tasks_tuple)

    def arg_stream():
        for i in range(n):
            yield (
                str(split.ids[i]),
                split.X[i],
                tasks_tuple,
                tuple(float(v) for v in split.y[i]),
            )

    def flush(writer: pq.ParquetWriter | None, batch: list[dict]) -> pq.ParquetWriter:
        table = pa.Table.from_pylist(batch, schema=schema)
        if writer is None:
            writer = pq.ParquetWriter(out, schema=schema, compression="zstd")
        writer.write_table(table)
        return writer

    t0 = time.time()
    writer: pq.ParquetWriter | None = None
    batch: list[dict] = []
    processed = 0

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for rec in pool.map(build_record, arg_stream(), chunksize=chunksize):
            batch.append(rec)
            processed += 1
            if len(batch) >= row_group_size:
                writer = flush(writer, batch)
                batch.clear()
                if processed % (row_group_size * 5) == 0:
                    rate = processed / (time.time() - t0)
                    print(f"  {split_name}: {processed:>7,}/{n:,}  ({rate:,.0f} rec/s)")

    if batch:
        writer = flush(writer, batch)
    if writer is not None:
        writer.close()

    dt = time.time() - t0
    size_mb = out.stat().st_size / 1e6 if out.exists() else 0.0
    print(f"wrote {out}  ({size_mb:.1f} MB, {n:,} records, {dt:.1f}s)")
    return out


def main() -> None:
    tasks, datasets, _transformers = dc.molnet.load_zinc15(
        featurizer="Raw",
        dataset_size="10M",
        reload=True,
    )
    train, valid, test = datasets
    print("tasks:", tasks)
    print("train / valid / test:", len(train), len(valid), len(test))
    print("first X entry type:", type(train.X[0]).__name__)
    print("first id:", train.ids[0])

    out_dir = Path(".")
    n_workers = max(1, (os.cpu_count() or 2) - 1)
    chunksize = 256          # IPC batch size for the worker pool
    row_group_size = 10_000  # rows per parquet row group; ~50-80 MB uncompressed at ZINC scale
    tasks_tuple = tuple(tasks)

    print(f"\nexporting with {n_workers} workers, "
          f"chunksize={chunksize}, row_group_size={row_group_size}")
    for split, name in [(train, "train"), (valid, "valid"), (test, "test")]:
        export_split(split, name, tasks_tuple, out_dir, n_workers, chunksize, row_group_size)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
