"""Gene universe: one fixed, sorted list of NCBI Gene IDs shared by every dataset in a family.

Plain-words picture: two classes kept marks in two spreadsheets with different
column names ("Maths" vs "Arithmetic"). Before comparing students we agree on the
columns, their order, and which nicknames are the same subject. Our "nicknames"
are real: Baron writes ACKR1, Muraro writes DARC__chr1. Matching by the stable
NCBI Gene ID, not by name, is what makes those the same column.

Rules (proposal IV.D):
* Built from feature metadata (``adata.var``) only, never from expression or labels.
* Unexpressed-but-measured genes stay in; "unexpressed" is not "not measured".
* No string surgery on names; identity goes through ``ncbi_gene_id`` (Week 1).
* The order is ``sorted`` by numeric ID, never the iteration order of a set
  (string hashing is randomised per Python process).

Decisions left to configuration (``universe:`` in configs/pilot.yaml):
which ID resolutions count as verified, what to do with features sharing one
ID (drop now; summing is a documented alternative), and which datasets define
the family universe (pairwise Baron/Muraro now; all four pancreas studies later).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .provenance import read_tsv, sha256_file, sha256_ordered, write_json, write_tsv


class UniverseError(ValueError):
    """A dataset cannot be aligned to the universe, or the policy is invalid."""


@dataclass(frozen=True)
class UniversePolicy:
    keep_resolutions: tuple[str, ...] = ("official_symbol", "unique_synonym")
    # "drop": a Gene ID reached by two features of one dataset is left out (pilot).
    # "sum": add those features' counts per ID. Not implemented; DECISION for the leader.
    shared_id_policy: str = "drop"
    drop_spike_ins: bool = True

    @classmethod
    def from_config(cls, cfg: dict) -> "UniversePolicy":
        policy = cls(
            keep_resolutions=tuple(cfg.get("keep_resolutions", cls.keep_resolutions)),
            shared_id_policy=cfg.get("shared_id_policy", "drop"),
            drop_spike_ins=bool(cfg.get("drop_spike_ins", True)),
        )
        if policy.shared_id_policy != "drop":
            raise NotImplementedError(
                "shared_id_policy 'sum' is a documented alternative but is not implemented in the pilot. "
                "Implementing it means summing counts per ID inside align(), and logging which features merged."
            )
        return policy


def feature_policy(var: pd.DataFrame, policy: UniversePolicy) -> pd.DataFrame:
    """One row per feature: kept or not, and the first reason it was not."""
    table = pd.DataFrame(
        {
            "feature_id_original": var["feature_id_original"].astype(str).to_numpy(),
            "symbol_original": var["symbol_original"].astype("string").fillna("").to_numpy(),
            "symbol": var["symbol"].astype("string").fillna("").to_numpy(),
            "ncbi_gene_id": var["ncbi_gene_id"].astype("string").fillna("").to_numpy(),
            "id_resolution": var["id_resolution"].astype(str).to_numpy(),
        }
    )
    reason = pd.Series("", index=table.index, dtype=object)
    spike = var["is_spike_in"].to_numpy(dtype=bool)
    shared = var["shares_gene_id"].to_numpy(dtype=bool)
    unverified = ~table["id_resolution"].isin(policy.keep_resolutions).to_numpy()
    # Order matters only for the reason text: the most fundamental reason wins.
    reason[shared & (reason == "")] = "shares_gene_id"
    reason[unverified] = table.loc[unverified, "id_resolution"]
    if policy.drop_spike_ins:
        reason[spike] = "spike_in"
    table["reason"] = reason
    table["kept"] = reason == ""
    return table


def build_universe(var_tables: dict[str, pd.DataFrame], policy: UniversePolicy) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (universe, excluded) from the ``var`` table of each dataset.

    universe: ``ncbi_gene_id``, then ``symbol_<dataset>`` and ``feature_<dataset>``
    for each dataset, sorted by numeric Gene ID.
    excluded: ``dataset, feature_id_original, symbol_original, ncbi_gene_id, reason``.
    """
    if len(var_tables) < 1:
        raise UniverseError("build_universe needs at least one dataset")
    per_dataset, excluded = {}, []
    for name in sorted(var_tables):
        table = feature_policy(var_tables[name], policy)
        per_dataset[name] = table[table["kept"]].set_index("ncbi_gene_id")
        dropped = table.loc[~table["kept"], ["feature_id_original", "symbol_original", "ncbi_gene_id", "reason"]]
        excluded.append(dropped.assign(dataset=name))
    shared_ids = set.intersection(*(set(t.index) for t in per_dataset.values()))
    ordered = sorted(shared_ids, key=lambda g: (int(g), g))
    universe = pd.DataFrame({"ncbi_gene_id": ordered})
    for name, table in per_dataset.items():
        universe[f"symbol_{name}"] = table.loc[ordered, "symbol"].to_numpy()
        universe[f"feature_{name}"] = table.loc[ordered, "feature_id_original"].to_numpy()
        not_shared = table.loc[~table.index.isin(shared_ids)]
        excluded.append(pd.DataFrame({
            "feature_id_original": not_shared["feature_id_original"].to_numpy(),
            "symbol_original": not_shared["symbol_original"].to_numpy(),
            "ncbi_gene_id": not_shared.index.to_numpy(),
            "reason": "not_in_every_dataset",
            "dataset": name,
        }))
    excluded = pd.concat(excluded, ignore_index=True)[
        ["dataset", "feature_id_original", "symbol_original", "ncbi_gene_id", "reason"]
    ].sort_values(["dataset", "reason", "feature_id_original"], kind="stable").reset_index(drop=True)
    return universe, excluded


def write_universe(universe: pd.DataFrame, excluded: pd.DataFrame, folder: str | Path, name: str,
                   meta: dict) -> Path:
    """Write ``<name>.tsv``, ``<name>_excluded.tsv`` and a ``<name>.json`` sidecar."""
    folder = Path(folder)
    path = write_tsv(universe, folder / f"{name}.tsv")
    write_tsv(excluded, folder / f"{name}_excluded.tsv")
    counts = excluded.groupby(["dataset", "reason"]).size().reset_index(name="n").to_dict("records")
    write_json({**meta, "name": name, "n_genes": int(len(universe)), "ids_sha256": sha256_ordered(universe["ncbi_gene_id"]),
                "file_sha256": sha256_file(path), "excluded_counts": counts}, folder / f"{name}.json")
    return path


def read_universe_ids(path: str | Path) -> list[str]:
    return read_tsv(path)["ncbi_gene_id"].tolist()


def align(adata, universe_ids: list[str]):
    """Return a copy of ``adata`` whose columns are exactly ``universe_ids``, in that order.

    The Gene ID becomes the column name; the deposited names stay in
    ``var['feature_id_original']``. Values are copied, never changed. A universe
    gene that is missing, or reached by two features, stops the run: silently
    filling zeros would turn "not measured" into "not expressed".
    """
    ids = adata.var["ncbi_gene_id"].astype("string")
    present = ids.notna().to_numpy()
    positions = pd.Series(np.flatnonzero(present), index=ids[present].to_numpy())
    duplicated = positions.index[positions.index.duplicated()]
    wanted = pd.Index(universe_ids)
    missing = wanted.difference(positions.index)
    clashing = wanted.intersection(duplicated)
    if len(missing) or len(clashing):
        dataset = adata.uns.get("annotbench", {}).get("dataset", "dataset")
        raise UniverseError(
            f"{dataset}: cannot align to the universe: {len(missing)} universe genes are absent "
            f"(e.g. {list(missing[:5])}) and {len(clashing)} are reached by several features "
            f"(e.g. {list(clashing[:5])})."
        )
    columns = positions.loc[wanted].to_numpy()
    out = adata[:, columns].copy()
    out.var_names = pd.Index(list(universe_ids), dtype=str)
    out.var.index.name = None
    out.uns.setdefault("annotbench", {})["universe_ids_sha256"] = sha256_ordered(universe_ids)
    return out
