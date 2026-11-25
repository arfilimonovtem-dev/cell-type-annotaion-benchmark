"""Scenario runner: run every scenario x seed x method the same way, and always write a record.

Plain-words picture: the race official with a clipboard. For every race it does
the same things in the same order: line up the runners, fire the gun, time them,
write down the result. A runner who trips is written down as "fell at 200 m",
never erased and never given an invented time (proposal VII.D).

Order of work
-------------
Once per pilot (``prepare``):
  1. load each dataset with ``load_dataset`` + ``attach_labels`` (never read an .h5ad directly)
  2. QC every cell with fixed rules; write retention by dataset x donor x class
  3. build the gene universe from ``var`` tables; align every dataset to it
  4. normalization constants per dataset, from QC-passing cells (role-independent)
Per scenario x seed:
  5. make, validate and write the split file of the reference (metadata only); for a
     cross-dataset scenario also the query set (which query cells are scored, and how)
  6. cut the three blocks the methods see: train, calibration, query (sorted by cell ID)
Per method instance (``run_one``), timed in three stages:
  7. preprocessing: fit the preprocessor on train, transform all three blocks
  8. fit: the method learns from training cells only
  9. annotate: predict calibration and query cells
 10. write predictions, scores, fitted state and run_record.json (always, even on failure)

Nothing fitted is cached across scenarios: E3's "refit affected stages" happens
because every run refits from its own ``train`` block.

Runs are independent, so ``jobs > 1`` runs them in parallel worker processes.
Every run pins numerical libraries to ``threads_per_run`` threads (default 1),
so the output bytes do not depend on how many runs share the machine.
"""
from __future__ import annotations

import multiprocessing
import os
import time
import tracemalloc
import traceback
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import config as project_config
from .labels import attach_labels, load_label_map
from .loading import load_dataset, load_gene_reference_from_registry
from .methods import IneligibleInput, MissingScoreOutput, make_method, resolve_instance
from .methods.base import TrainingData, check_output
from .preprocess import Preprocessor, PreprocessConfig, hvg_query_coverage, library_sizes, resolve_target_sums
from .provenance import (
    git_state, host_label, package_versions, sha256_file, sha256_ids, sha256_ordered, sha256_text, utc_now,
    read_tsv, write_json, write_tsv,
)
from .qc import retention_table, run_qc
from .splits import (
    FIELD_VALUE, QUERY_ROLES, Scenario, SplitError, SplitRules, make_query_set, make_split, train_classes_of,
    write_query_set, write_split,
)
from .universe import UniversePolicy, align, build_universe, write_universe

STATUSES = ("success", "software_failure", "ineligible_input", "missing_score_output", "timeout")  # io_spec §6


# --------------------------------------------------------------------------- configuration
@dataclass
class PilotConfig:
    path: Path
    sha256: str
    name: str
    registry_path: Path
    label_map_path: Path
    seeds: list[int]
    split_version: str
    label_task: str
    qc: dict
    universe: dict
    split_rules: SplitRules
    preprocess: PreprocessConfig
    inner_cv_refit_preprocessing: bool
    methods: dict
    scenarios: list[Scenario]
    threads_per_run: int = 1

    @property
    def datasets(self) -> list[str]:
        names = set(self.universe.get("datasets") or [])
        for s in self.scenarios:
            names |= {s.reference, s.query}
        return sorted(names)


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_config.REPO_ROOT / path


def load_pilot_config(path: str | Path) -> PilotConfig:
    path = _repo_path(str(path))
    raw = project_config.load_yaml(path)
    label_task = raw.get("label_task", "pancreas_harmonized_v1")
    for instance, params in (raw.get("methods") or {}).items():
        if not FIELD_VALUE.match(instance):
            raise SplitError(f"method instance '{instance}' contains characters not allowed in a run ID")
        resolve_instance(instance, params)
    return PilotConfig(
        path=path,
        sha256=sha256_file(path),
        name=raw.get("name", path.stem),
        registry_path=_repo_path(raw.get("registry", "configs/datasets.yaml")),
        label_map_path=_repo_path(raw.get("label_map", "labels/label_map.csv")),
        seeds=[int(s) for s in raw.get("seeds", [0])],
        split_version=raw.get("split_version", "pilot_v1"),
        label_task=label_task,
        qc=raw.get("qc") or {},
        universe=raw.get("universe") or {},
        split_rules=SplitRules.from_config(raw.get("split_rules")),
        preprocess=PreprocessConfig.from_config(raw.get("preprocess") or {}),
        inner_cv_refit_preprocessing=bool((raw.get("inner_cv") or {}).get("refit_preprocessing", True)),
        methods=raw.get("methods") or {},
        scenarios=[Scenario.from_config(entry, label_task) for entry in raw.get("scenarios") or []],
        threads_per_run=int(raw.get("threads_per_run", 1)),
    )


@dataclass
class OutputLayout:
    root: Path

    @property
    def splits(self) -> Path:
        return self.root / "splits"

    @property
    def query_sets(self) -> Path:
        return self.root / "query_sets"

    @property
    def universes(self) -> Path:
        return self.root / "gene_universes"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def runs(self) -> Path:
        return self.root / "results" / "runs"

    @property
    def reports(self) -> Path:
        return self.root / "reports" / "week2"


# --------------------------------------------------------------------------- preparation
@dataclass
class PreparedDataset:
    name: str
    unit: str
    counts: sp.csr_matrix              # cells x universe genes, the registry unit, untouched values
    obs: pd.DataFrame                  # cell metadata + qc_pass, qc_reason, libsize_all_endogenous
    row: dict = field(default_factory=dict)  # cell_id -> row position

    def rows(self, cell_ids) -> np.ndarray:
        return np.fromiter((self.row[c] for c in cell_ids), dtype=np.int64, count=len(cell_ids))


@dataclass
class PreparedPilot:
    datasets: dict[str, PreparedDataset]
    universe_ids: list[str]
    universe_sha256: str
    universe_file_sha256: str
    target_sums: dict[str, float]
    cl_ids: dict[str, str]
    qc_retention: pd.DataFrame
    symbols: dict[str, dict[str, str]] = field(default_factory=dict)  # dataset -> {gene id: symbol}

    def watch_genes(self, dataset: str, wanted) -> dict[str, str]:
        """{Gene ID: symbol} for every watched symbol.

        A symbol outside this dataset's universe genes is kept under the key
        ``not_in_universe:<symbol>``, so the HVG report says it is missing
        instead of silently leaving it out.
        """
        by_symbol = {sym: gid for gid, sym in self.symbols.get(dataset, {}).items()}
        return {by_symbol.get(s, f"not_in_universe:{s}"): s for s in wanted}


def load_datasets(cfg: PilotConfig, data_root: str | Path | None = None) -> dict:
    """``load_dataset`` + ``attach_labels`` for every dataset the pilot needs. The only door to the data."""
    registry = project_config.load_registry(cfg.registry_path)
    label_map = load_label_map(cfg.label_map_path)
    root = project_config.resolve_data_root(data_root)
    reference = load_gene_reference_from_registry(registry, root)
    adatas = {}
    for name in cfg.datasets:
        adata = load_dataset(name, root, registry, reference)
        attach_labels(adata, label_map, project_config.dataset_config(registry, name))
        adatas[name] = adata
    return adatas


def harmonized_to_cl(label_map: pd.DataFrame) -> dict[str, str]:
    table = label_map.loc[label_map["eligible_primary"], ["harmonized_label", "cl_id"]].drop_duplicates()
    clash = table["harmonized_label"].duplicated(keep=False)
    if clash.any():
        raise ValueError(f"harmonized labels with several CL IDs: {sorted(set(table.loc[clash, 'harmonized_label']))}")
    return dict(zip(table["harmonized_label"], table["cl_id"]))


def prepare(cfg: PilotConfig, adatas: dict, out: OutputLayout) -> PreparedPilot:
    """QC, universe, alignment and normalization constants: everything that is scenario-independent."""
    qc_dir = out.reports / "qc"
    retention, rule_tables = [], []
    for name in cfg.datasets:
        adata = adatas[name]
        metrics, verdict, rules = run_qc(adata, cfg.qc)
        adata.obs["qc_pass"] = verdict["qc_pass"].to_numpy()
        adata.obs["qc_reason"] = verdict["qc_reason"].to_numpy()
        adata.obs["libsize_all_endogenous"] = metrics["total_counts"].to_numpy()
        write_tsv(pd.concat([metrics, verdict], axis=1).rename_axis("cell_id").reset_index(),
                  qc_dir / f"{name}_qc_metrics.tsv.gz")
        retention.append(retention_table(adata.obs, verdict))
        rule_tables.append(rules)
    qc_retention = pd.concat(retention, ignore_index=True)
    write_tsv(qc_retention, qc_dir / "qc_retention.tsv")
    write_tsv(pd.concat(rule_tables, ignore_index=True), qc_dir / "qc_rules.tsv")

    universe_cfg = cfg.universe
    members = universe_cfg.get("datasets") or cfg.datasets
    policy = UniversePolicy.from_config(universe_cfg)
    universe, excluded = build_universe({n: adatas[n].var for n in members}, policy)
    universe_name = universe_cfg.get("name", "pilot_universe")
    universe_path = write_universe(universe, excluded, out.universes, universe_name, {
        "status": universe_cfg.get("status", "provisional"), "datasets": sorted(members),
        "policy": asdict(policy), "note": universe_cfg.get("note", ""),
    })
    universe_ids = universe["ncbi_gene_id"].tolist()

    prepared, pools = {}, {}
    for name in cfg.datasets:
        aligned = align(adatas[name], universe_ids)
        counts = sp.csr_matrix(aligned.X, dtype=np.float32, copy=True)
        obs = aligned.obs.copy()
        obs.index = obs.index.astype(str)
        dataset = PreparedDataset(name, adatas[name].uns["annotbench"]["assay_unit"], counts, obs,
                                  {c: i for i, c in enumerate(obs.index)})
        prepared[name] = dataset
        pool = obs["qc_pass"].to_numpy(dtype=bool)
        totals = (library_sizes(counts) if cfg.preprocess.library_size == "universe"
                  else obs["libsize_all_endogenous"].to_numpy(dtype=float))
        pools[name] = totals[pool]
    target_sums = resolve_target_sums(cfg.preprocess, pools)
    write_json({"mode": cfg.preprocess.target_sum, "library_size": cfg.preprocess.library_size,
                "target_sums": target_sums, "pool": "cells passing QC, any role",
                "pool_sizes": {n: int(len(v)) for n, v in pools.items()},
                "pool_median_library_size": {n: float(np.median(v)) for n, v in pools.items()}},
               out.reports / "normalization_constants.json")
    symbols = {n: dict(zip(universe["ncbi_gene_id"], universe[f"symbol_{n}"].astype(str)))
               for n in members if f"symbol_{n}" in universe.columns}
    return PreparedPilot(prepared, universe_ids, sha256_ordered(universe_ids), sha256_file(universe_path),
                         target_sums, harmonized_to_cl(load_label_map(cfg.label_map_path)), qc_retention, symbols)


# --------------------------------------------------------------------------- scenario blocks
@dataclass
class Block:
    """Cells of one role group, sorted by cell ID, with raw universe-aligned counts."""

    name: str                  # "train" | "calibration" | "query"
    dataset: str
    cell_ids: np.ndarray
    counts: sp.csr_matrix
    totals: np.ndarray
    labels: np.ndarray | None = None   # only kept for train
    groups: np.ndarray | None = None   # inner_fold, only for train


def make_blocks(split: pd.DataFrame, scenario: Scenario, prepared: PreparedPilot,
                query_set: pd.DataFrame | None = None) -> dict[str, Block]:
    """Train and calibration come from the split; the query from the split's held-out donors
    (within-dataset) or from the query set (cross-dataset)."""
    if scenario.within == (query_set is not None):
        raise SplitError("a cross-dataset scenario needs its query set; a within-dataset one must not have one")

    def cut(name: str, dataset: str, roles: tuple[str, ...], with_labels: bool, table: pd.DataFrame = split) -> Block:
        part = table[(table["dataset"] == dataset) & table["role"].isin(roles)].sort_values("cell_id", kind="stable")
        data = prepared.datasets[dataset]
        rows = data.rows(part["cell_id"].tolist())
        return Block(
            name=name, dataset=dataset, cell_ids=part["cell_id"].to_numpy(dtype=str),
            counts=data.counts[rows], totals=data.obs["libsize_all_endogenous"].to_numpy(dtype=float)[rows],
            labels=part["label_harmonized"].to_numpy(dtype=str) if with_labels else None,
            groups=part["inner_fold"].to_numpy(dtype=str) if with_labels else None,
        )

    return {
        "train": cut("train", scenario.reference, ("train",), True),
        "calibration": cut("calibration", scenario.reference, ("calibration",), False),
        "query": cut("query", scenario.query, QUERY_ROLES, False, split if query_set is None else query_set),
    }


# --------------------------------------------------------------------------- one run
class Stage:
    """Time one stage and record its peak traced memory (Python/numpy allocations made during it)."""

    def __init__(self, record: dict, name: str):
        self.record, self.name = record, name

    def __enter__(self):
        tracemalloc.reset_peak()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        _, peak = tracemalloc.get_traced_memory()
        self.record["stages"][self.name] = {"wall_seconds": round(time.perf_counter() - self.start, 4),
                                            "peak_memory_mb": round(peak / 1e6, 1)}
        return False


def _predictions(run_id: str, block: Block, output, cl_ids: dict, cell_set: str) -> pd.DataFrame:
    rejected = np.asarray(output.rejected, dtype=bool)
    labels = np.asarray(output.predicted_label, dtype=str)
    top = output.scores.max(axis=1) if output.scores is not None else np.full(len(labels), np.nan)
    return pd.DataFrame({
        "run_id": run_id, "dataset": block.dataset, "cell_id": block.cell_ids, "cell_set": cell_set,
        "output_kind": "native", "predicted_label": labels,
        "predicted_cl_id": [cl_ids.get(label, "") for label in labels],
        "prediction_level": np.where(rejected, "rejected", "leaf"),
        "rejected": np.where(rejected, "true", "false"),
        "top_score": top, "score_type": output.score_type or "",
    })


def _scores(block: Block, output, cell_set: str) -> pd.DataFrame:
    table = pd.DataFrame(output.scores, columns=output.vocabulary)
    table.insert(0, "score_type", output.score_type)
    table.insert(0, "cell_set", cell_set)
    table.insert(0, "cell_id", block.cell_ids)
    table.insert(0, "dataset", block.dataset)
    return table


def run_one(method_name: str, params: dict, blocks: dict[str, Block], prepared: PreparedPilot,
            cfg: PilotConfig, scenario: Scenario, seed: int, split_id: str, split_file: Path,
            out: OutputLayout, query_set_file: Path | None = None) -> dict:
    """One method instance on one scenario x seed. ``method_name`` is the instance name."""
    from threadpoolctl import threadpool_limits

    run_id = f"{split_id}__method-{method_name}"
    folder = out.runs / run_id
    folder.mkdir(parents=True, exist_ok=True)
    train, calib, query = blocks["train"], blocks["calibration"], blocks["query"]
    record = {
        "run_id": run_id, "split_id": split_id, "method": method_name,
        "method_family": resolve_instance(method_name, params)[0], "exp": scenario.exp,
        "reference": scenario.reference, "query": scenario.query, "perturbation": scenario.perturbation,
        "seed": seed, "status": None, "stages": {}, "error_type": None, "error_message": None,
        "input_unit": {d: prepared.datasets[d].unit for d in sorted({scenario.reference, scenario.query})},
        "representation": None, "method_settings": None,
        "preprocess_settings": asdict(cfg.preprocess), "target_sums": prepared.target_sums,
        "query_adaptive_steps": (["normalization constant (dataset median) computed from unlabeled query cells"]
                                 if cfg.preprocess.target_sum == "dataset_median" else []),
        "inner_cv_refit_preprocessing": cfg.inner_cv_refit_preprocessing,
        "n_cells": {"train": len(train.cell_ids), "calibration": len(calib.cell_ids), "query": len(query.cell_ids)},
        "hashes": {
            "train_ids_sha256": sha256_ids(train.cell_ids),
            "calibration_ids_sha256": sha256_ids(calib.cell_ids),
            "query_ids_sha256": sha256_ids(query.cell_ids),
            "universe_sha256": prepared.universe_sha256,
            "universe_file_sha256": prepared.universe_file_sha256,
            "split_file_sha256": sha256_file(split_file),
            "query_set_file_sha256": sha256_file(query_set_file) if query_set_file else None,
            "config_sha256": cfg.sha256,
        },
        "threads_per_run": cfg.threads_per_run,
        "vocabulary": None, "warnings": [], "hvg_anomalies": [], "hvg_query_coverage": None,
        **git_state(), "environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "package_versions": package_versions(), "host": host_label(), "started_utc": utc_now(),
    }
    gene_ids = prepared.universe_ids
    caught: list = []
    tracemalloc.start()
    limits = threadpool_limits(limits=cfg.threads_per_run)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            method = make_method(method_name, params)
            record["representation"] = method.representation
            method.check_units(record["input_unit"])
            if len(train.cell_ids) == 0:
                raise IneligibleInput("the split has no training cells")

            with Stage(record, "preprocessing"):
                prep = Preprocessor(cfg.preprocess, seed).fit(
                    train.counts, gene_ids, train.cell_ids, train.dataset, prepared.target_sums,
                    train.totals, train.groups if cfg.preprocess.hvg_batch_key else None,
                    watch_genes=prepared.watch_genes(train.dataset, cfg.preprocess.hvg_watch_genes),
                    groups=train.groups)
                reps = {b.name: _representation(prep, b, gene_ids, method.representation)
                        for b in (train, calib, query)}
                prep.save(folder / "fitted", {"split_id": split_id, "run_id": run_id})
                record["hashes"]["hvg_sha256"] = sha256_ordered(prep.hvg["ncbi_gene_id"])
                record["hvg_anomalies"] = list(prep.hvg_report["anomalies"])
                if len(query.cell_ids):
                    coverage = hvg_query_coverage(query.counts, gene_ids, prep.hvg["ncbi_gene_id"].tolist())
                    record["hvg_query_coverage"] = {k: v for k, v in coverage.items() if k != "undetected_hvgs"}
                    if coverage["anomaly"]:
                        record["hvg_anomalies"].append("query: " + coverage["anomaly"])

            inner = _inner_representation(train, prep, cfg, seed, gene_ids, method.representation, reps["train"])
            with Stage(record, "fit"):
                method.fit(TrainingData(reps["train"], train.labels, train.cell_ids, train.groups, inner), seed)
            with Stage(record, "annotate"):
                outputs = {"calibration": method.predict(reps["calibration"]) if len(calib.cell_ids) else None,
                           "query": method.predict(reps["query"]) if len(query.cell_ids) else None}

            for cell_set, output in outputs.items():
                if output is not None:
                    check_output(output, len(blocks[cell_set].cell_ids))
            record["vocabulary"] = next(o.vocabulary for o in outputs.values() if o is not None)
            record["method_settings"] = method.settings()
            method.save(folder / "fitted")
            predictions = [_predictions(run_id, blocks[s], o, prepared.cl_ids, s) for s, o in outputs.items() if o is not None]
            write_tsv(pd.concat(predictions, ignore_index=True), folder / "predictions.tsv.gz")
            if any(o.scores is None for o in outputs.values() if o is not None):
                raise MissingScoreOutput(f"{method_name} returned no per-class scores")
            scores = [_scores(blocks[s], o, s) for s, o in outputs.items() if o is not None]
            write_tsv(pd.concat(scores, ignore_index=True), folder / "scores.tsv.gz")
            record["status"] = "success"
    except IneligibleInput as error:
        record.update(status="ineligible_input", error_type=type(error).__name__, error_message=str(error))
    except MissingScoreOutput as error:
        record.update(status="missing_score_output", error_type=type(error).__name__, error_message=str(error))
    except Exception as error:  # noqa: BLE001 - every failure must be recorded, never swallowed silently
        record.update(status="software_failure", error_type=type(error).__name__, error_message=str(error),
                      traceback=traceback.format_exc(limit=8))
    finally:
        limits.restore_original_limits()
        tracemalloc.stop()
        record["warnings"] = sorted({f"{w.category.__name__}: {str(w.message)[:200]}" for w in caught})
        record["finished_utc"] = utc_now()
        if record["status"] not in ("success", "missing_score_output"):
            # A failure record must never sit next to predictions left by an earlier run.
            for stale in ("predictions.tsv.gz", "scores.tsv.gz"):
                (folder / stale).unlink(missing_ok=True)
        write_json(record, folder / "run_record.json")
    return record


def _representation(prep: Preprocessor, block: Block, gene_ids, representation: str) -> np.ndarray:
    if representation == "counts":
        return block.counts  # count-based methods receive the untouched counts (IV.D)
    if len(block.cell_ids) == 0:
        return np.zeros((0, prep.pca_components.shape[0] if representation == "pca" else len(prep.hvg)))
    return prep.transform(block.counts, gene_ids, block.dataset, block.totals)[representation]


def _inner_representation(train: Block, prep: Preprocessor, cfg: PilotConfig, seed: int, gene_ids,
                          representation: str, rep_train: np.ndarray):
    """Representations for inner folds of train; refits the preprocessor per fold when configured."""
    def inner(fit_idx: np.ndarray, apply_idx: np.ndarray):
        if not cfg.inner_cv_refit_preprocessing or representation == "counts":
            return rep_train[fit_idx], rep_train[apply_idx]
        fold = Preprocessor(cfg.preprocess, seed).fit(
            train.counts[fit_idx], gene_ids, train.cell_ids[fit_idx], train.dataset, prep.target_sums,
            train.totals[fit_idx], train.groups[fit_idx] if cfg.preprocess.hvg_batch_key else None,
            diagnose=False)
        rep_fit = fold.transform(train.counts[fit_idx], gene_ids, train.dataset, train.totals[fit_idx])[representation]
        rep_apply = fold.transform(train.counts[apply_idx], gene_ids, train.dataset, train.totals[apply_idx])[representation]
        return rep_fit, rep_apply

    return inner


# --------------------------------------------------------------------------- the whole pilot
@dataclass
class ScenarioSplit:
    split_id: str
    split: pd.DataFrame
    path: Path
    query_set: pd.DataFrame | None = None
    query_set_path: Path | None = None


def make_scenario_split(cfg: PilotConfig, prepared: PreparedPilot, scenario: Scenario, seed: int,
                        out: OutputLayout) -> ScenarioSplit:
    """Split the reference (seeded); for a cross-dataset scenario, also build the query set."""
    split = make_split(prepared.datasets[scenario.reference].obs, scenario, cfg.split_rules, seed)
    split_id = scenario.split_id(seed)
    query_set = query_path = None
    if not scenario.within:
        query_set = make_query_set(prepared.datasets[scenario.query].obs, scenario, train_classes_of(split),
                                   cfg.split_rules)
        query_path = write_query_set(query_set, out.query_sets, cfg.split_version, scenario.scenario_id)
    path = write_split(split, cfg.split_rules, out.splits, cfg.split_version, split_id, query_set)
    return ScenarioSplit(split_id, split, path, query_set, query_path)


INDEX_COLUMNS = ["run_id", "split_id", "exp", "query", "perturbation", "method", "method_family", "seed", "status",
                 "error_type", "n_train", "n_calibration", "n_query", "wall_seconds_total"]

# State shared with worker processes. Set before the pool forks, so workers inherit
# the prepared data instead of receiving a pickled copy per task.
_WORKER: dict = {}


def _run_task(task: tuple[int, int, str]) -> dict:
    scenario_index, seed, instance = task
    cfg, prepared, out = _WORKER["cfg"], _WORKER["prepared"], _WORKER["out"]
    scenario = cfg.scenarios[scenario_index]
    item: ScenarioSplit = _WORKER["splits"][(scenario_index, seed)]
    blocks = make_blocks(item.split, scenario, prepared, item.query_set)
    return run_one(instance, cfg.methods[instance], blocks, prepared, cfg, scenario, seed, item.split_id,
                   item.path, out, item.query_set_path)


def _index_row(record: dict) -> dict:
    total = sum(s["wall_seconds"] for s in record["stages"].values())
    return ({k: record[k] for k in ("run_id", "split_id", "exp", "query", "perturbation", "method", "method_family",
                                    "seed", "status", "error_type")}
            | {"n_train": record["n_cells"]["train"], "n_calibration": record["n_cells"]["calibration"],
               "n_query": record["n_cells"]["query"], "wall_seconds_total": round(total, 3)})


def run_pilot(cfg: PilotConfig, out_root: str | Path | None = None, data_root: str | Path | None = None,
              adatas: dict | None = None, exp_filter: list[str] | None = None,
              method_filter: list[str] | None = None, prepare_only: bool = False,
              seed_filter: list[int] | None = None, jobs: int = 1) -> pd.DataFrame:
    """Run the configured pilot and return the run index (one row per run)."""
    out = OutputLayout(Path(out_root) if out_root else project_config.REPO_ROOT)
    started = utc_now()
    adatas = adatas if adatas is not None else load_datasets(cfg, data_root)
    prepared = prepare(cfg, adatas, out)
    splits, items, tasks = [], {}, []
    seeds = [s for s in cfg.seeds if not seed_filter or s in seed_filter]
    for i, scenario in enumerate(cfg.scenarios):
        if exp_filter and scenario.exp not in exp_filter:
            continue
        for seed in seeds:
            item = make_scenario_split(cfg, prepared, scenario, seed, out)
            items[(i, seed)] = item
            splits.append({"split_id": item.split_id, "sha256": sha256_file(item.path),
                           "query_set": item.query_set_path.name if item.query_set_path else None,
                           "query_set_sha256": sha256_file(item.query_set_path) if item.query_set_path else None})
            roles = item.split["role"].value_counts().sort_index()
            line = f"split {item.split_id}: " + ", ".join(f"{r}={n}" for r, n in roles.items())
            if item.query_set is not None:
                q = item.query_set["role"].value_counts().sort_index()
                line += " | query set: " + ", ".join(f"{r}={n}" for r, n in q.items())
            print(line)
            for instance, params in cfg.methods.items():
                family = resolve_instance(instance, params)[0]
                if method_filter and instance not in method_filter and family not in method_filter:
                    continue
                tasks.append((i, seed, instance))
    records = []
    if not prepare_only and tasks:
        _WORKER.update(cfg=cfg, prepared=prepared, out=out, splits=items)
        try:
            if jobs > 1:
                with multiprocessing.get_context("fork").Pool(jobs) as pool:
                    iterator = pool.imap(_run_task, tasks)
                    records = [_report(r) for r in iterator]
            else:
                records = [_report(_run_task(t)) for t in tasks]
        finally:
            _WORKER.clear()
    index = pd.DataFrame([_index_row(r) for r in records], columns=INDEX_COLUMNS)
    if not prepare_only:
        index_path = out.results / "run_index.tsv"
        if index_path.exists():
            previous = read_tsv(index_path)
            index = pd.concat([previous[~previous["run_id"].isin(index["run_id"])], index.astype(str)],
                              ignore_index=True).sort_values("run_id").reset_index(drop=True)
        write_tsv(index, index_path)
        anomalies = pd.DataFrame([{"run_id": r["run_id"], "anomaly": a} for r in records for a in r["hvg_anomalies"]],
                                 columns=["run_id", "anomaly"])
        anomaly_path = out.results / "hvg_anomalies.tsv"
        if anomaly_path.exists():
            previous = read_tsv(anomaly_path)
            anomalies = pd.concat([previous[~previous["run_id"].isin(index["run_id"])], anomalies], ignore_index=True)
        write_tsv(anomalies.sort_values(["run_id", "anomaly"]).reset_index(drop=True), anomaly_path)
    write_json({"config": str(cfg.path), "config_sha256": cfg.sha256, "universe_sha256": prepared.universe_sha256,
                "target_sums": prepared.target_sums, "splits": splits, "jobs": jobs,
                "threads_per_run": cfg.threads_per_run, **git_state(),
                "started_utc": started, "finished_utc": utc_now(), "prepare_only": prepare_only},
               out.results / "pilot_manifest.json")
    return index


def _report(record: dict) -> dict:
    total = sum(s["wall_seconds"] for s in record["stages"].values())
    print(f"  {record['run_id'].split('__', 2)[-1][:70]:<70} {record['method']:<12} {record['status']:<17} {total:7.1f}s"
          + (f"  {record['error_type']}: {record['error_message']}" if record["error_type"] else ""), flush=True)
    return record
