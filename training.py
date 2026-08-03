import os
from datetime import datetime
from pathlib import Path

import json2vec as jv
import lightning.pytorch as lit
import wandb
from json2vec.data.datasets.streaming import StreamingDataModule
from json2vec.data.processors import Observation, preprocess
from json2vec.structs.enums import Suffix
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from torch.optim import AdamW

# shared Unix account: force wandb to use MY key, never ~/.netrc
assert os.environ.get("WANDB_API_KEY"), "source ~/reid/wandb.env before running"
wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)


def adamw(lr: float, **kwargs):
    # json2vec calls model.optimizer(model); AdamW wants an iterable of params
    return lambda model: AdamW(model.parameters(), lr=lr, **kwargs)


@preprocess
def molecule_row(observation: dict) -> Observation:
    row = dict(observation)

    smi = row.get("canonical_smiles") or ""
    row["canonical_smiles"] = [{"char": c} for c in smi]

    targets = row.pop("targets", None) or {}
    row["targets"] = [{
        "mwt":      targets.get("mwt",None),
        "logp":     targets.get("logp",None),
        "reactive": targets.get("reactive",None),
    }]

    row["exact_mass"] = row.get("exact_mass",None)
    row["net_charge"] = row.get("net_charge",None)

    atoms = row.get("atoms") or []
    row["atoms"] = [
        {
            "id":                  a.get("id",None),
            "element":             a.get("element",None),
            "atomic_number":       a.get("atomic_number",None),
            "formal_charge":       a.get("formal_charge",None),
            "n_hydrogens":         a.get("n_hydrogens",None),
            "n_radical_electrons": a.get("n_radical_electrons",None),
            "chirality":           a.get("chirality",None),
            "aromatic":            a.get("aromatic",None),
            "bonds":               a.get("bonds") or [],
        }
        for a in atoms
    ]

    # explode {nodes, edges} into per-atom adjacency records to match the schema
    structure = row.pop("structure", None) or {}
    nodes: list[str] = structure.get("nodes") or []
    edges: list[dict] = structure.get("edges") or []
    edges_by_atom: dict[str, list[dict]] = {n: [] for n in nodes}
    for e in edges:
        edge = {
            "id":    e["id"],
            "atoms": [{"id": a} for a in e.get("atoms") or []],
        }
        for a in e.get("atoms") or []:
            edges_by_atom.setdefault(a, []).append(edge)
    row["structure"] = [
        {"nodes": n, "edges": edges_by_atom.get(n, [])} for n in nodes
    ]

    return Observation(data=row)


model = jv.Model.from_tree(
    d_model=256,
    n_layers=8,
    n_heads=16,
    embed=True,

    targets = jv.Branch(
        mwt      = jv.Number(),
        logp     = jv.Number(),
        reactive = jv.Number()
    ),

    atoms = jv.Branch(
        length              = 96,
        id                  = jv.StaticEntity(group="atoms"),
        element             = jv.Category(capacity=120, ),
        atomic_number       = jv.Category(capacity=120, ),
        formal_charge       = jv.Number(),
        n_hydrogens         = jv.Number(),
        n_radical_electrons = jv.Number(),
        chirality           = jv.Category(capacity=4, ),
        aromatic            = jv.Boolean(),

        bonds = jv.Branch(
            length = 8,
            id     = jv.StaticEntity(group="bonds"),
            atom   = jv.StaticEntity(group="atoms"),
            order  = jv.Number(),
            stereo = jv.Category(capacity=8, ),
        ),
    ),

    structure = jv.Branch(
        length = 96,
        nodes = jv.StaticEntity(group="atoms"),
        edges = jv.Branch(
            length = 8,
            id = jv.StaticEntity(group="bonds"),
            atoms = jv.Branch(
                id = jv.StaticEntity(group="atoms")
            )
        )
    ),

    canonical_smiles = jv.Branch(
        length = 1024,
        char = jv.Category(capacity=100, ),
    ),

    exact_mass = jv.Number(),
    net_charge = jv.Number()
)


datamodule = StreamingDataModule(
    model=model,
    root=Path(__file__).parent,
    suffix=Suffix.parquet,
    train=r"zinc15_train\.parquet$",
    validate=r"zinc15_valid\.parquet$",
    test=r"zinc15_test\.parquet$",
    preprocessor=molecule_row,
    file_buffer_size=200,
    observation_buffer_size=10_000,
    sharding=jv.ShardingStrategy.file,
    num_workers=8,
    replacement=True,
)


def trainer(logger):
    return lit.Trainer(
        callbacks=[
            jv.RollbackCheckpoint(monitor="loss/validate", mode="min"),
            EarlyStopping(monitor="loss/validate", mode="min", patience=10),
        ],
        min_epochs=150,
        logger=logger,
    )

logger = WandbLogger(
    entity="rebridgers-independent", 
    project="deepchem", 
    name=datetime.now().strftime("%Y-%m-%d %H:%M"),
    config={
        "learning_rate": 2e-5,
        "architecture": "Json2Vec",
        "dataset": "Zinc15",
        "min_epochs": 150,
    },
)

# phase: pretrain
model.update(dropout=0.05)
model.update(jv.where("type") == "static_entity", p_mask=0.15, p_prune=0.05)
model.update(jv.where("type") == "category", p_mask=0.15, p_prune=0.05)
model.update(jv.where("type") == "number", p_mask=0.15, p_prune=0.05)
model.update(jv.where("type") == "boolean", p_mask=0.15, p_prune=0.05)
model.optimizer = adamw(2e-5, weight_decay=0.01)

trainer(logger).fit(model=model, datamodule=datamodule)

# phase: finetune
model.update(jv.where("type") == "static_entity", p_mask=0.0, p_prune=0.0)
model.update(jv.where("type") == "category", p_mask=0.0, p_prune=0.0)
model.update(jv.where("type") == "number", p_mask=0.0, p_prune=0.0)
model.update(jv.where("type") == "boolean", p_mask=0.0, p_prune=0.0)
model.update(jv.where("name") == "mwt", p_mask=0.5, p_prune=0.3)
model.update(jv.where("name") == "logp", p_mask=0.5, p_prune=0.3)
model.update(jv.where("name") == "reactive", p_mask=0.5, p_prune=0.3)
model.optimizer = adamw(2e-5, weight_decay=0.01)
trainer(logger).fit(model=model, datamodule=datamodule)

# phase: polish
model.optimizer = adamw(2e-6, weight_decay=0.01)
trainer(logger).fit(model=model, datamodule=datamodule)

model.save("checkpoint.ckpt")
trainer(logger).test(model=model, datamodule=datamodule)

