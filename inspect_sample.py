"""Export one ZINC15 row to Parquet and print the raw + preprocessed shapes."""

from pathlib import Path

import deepchem as dc
import pyarrow as pa
import pyarrow.parquet as pq

from export_data import build_record, build_schema


def main() -> None:
    tasks, datasets, _ = dc.molnet.load_zinc15(
        featurizer="Raw", dataset_size="250K", reload=True,
    )
    train, _valid, _test = datasets
    tasks_tuple = tuple(tasks)

    record = build_record((
        str(train.ids[0]),
        train.X[0],
        tasks_tuple,
        tuple(float(v) for v in train.y[0]),
    ))

    schema = build_schema(tasks_tuple)
    out = Path("sample.parquet")
    pq.write_table(pa.Table.from_pylist([record], schema=schema), out, compression="zstd")
    print(f"wrote {out}  ({out.stat().st_size} bytes)\n")

    raw_row = pq.read_table(out).to_pylist()[0]
    print("=== raw parquet JMESPath queryable fields ===")
    for path in jmespaths(raw_row):
        print(path)


def jmespaths(value: object, prefix: str = "") -> list[str]:
    if isinstance(value, dict) and value:
        out: list[str] = []
        for k, v in value.items():
            child = f"{prefix}.{k}" if prefix else k
            out.extend(jmespaths(v, child))
        return out
    if isinstance(value, list) and value:
        return jmespaths(value[0], f"{prefix}[*]")
    return [prefix]

if __name__ == "__main__":
    main()
