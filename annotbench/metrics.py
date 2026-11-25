"""Pilot evaluation: score saved predictions against labels from the split or the query set.

Plain-words picture: the exam is over and the answer sheets are collected
(``predictions.tsv.gz``). Only now does the teacher open the answer key, which
was never in the room: true labels live in the split file (held-out reference
donors) or in the query set (another dataset), never in a method's own folder
(docs/io_spec.md §5).

What is computed (proposal VII.A, VII.C; flat metrics only, the ontology-based
hierarchical error of VII.B needs the pinned Cell Ontology tree and comes later):

* closed-set metrics on **seen** query cells (``evaluation``): accuracy,
  macro-F1 over the eligible true classes, per-class recall / precision / F1,
  and the confusion matrix. A prediction of a class the query does not have is
  simply wrong for that cell; such classes are not added to the macro average.
* a **common rejection threshold**: the largest top-score threshold that rejects
  at most 5% of the reference calibration cells (VII.A). With it: the false
  rejection rate on seen query cells (FRR), the true rejection rate on unseen
  ones (TRR), balanced rejection accuracy (TRR + 1 - FRR) / 2 (H1), coverage and
  accuracy among accepted cells.
* a threshold-free view of unseen-type detection: AUROC of "low top score" for
  unseen vs seen query cells.
* the class-balanced FRR (mean of per-class FRRs over seen classes), which VII.A
  asks for next to the plain rate. With one unseen class, class-balanced TRR equals TRR.
* for probability outputs only: expected calibration error with 10 equal-width
  bins (eq. 6), the multiclass Brier score (VII.C) and the mean confidence. kNN
  vote fractions are scores, not probabilities (proposal V), so they get neither.

Everything here is descriptive pilot output. The proposal's primary analyses
(paired summaries, family weighting, Holm-corrected tests) come later.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .methods.base import PROBABILITY_SCORES
from .provenance import read_tsv, utc_now, write_json, write_tsv
from .splits import QUERY_ROLES, query_set_path, read_query_set, read_split, split_path

TARGET_FRR = 0.05          # proposal VII.A: at most 5% false rejection on reference calibration cells
ECE_BINS = 10              # proposal VII.C, eq. 6


# --------------------------------------------------------------------------- building blocks
def common_threshold(calibration_scores, target: float = TARGET_FRR) -> tuple[float, float]:
    """(threshold, achieved calibration FRR). A cell is rejected when its top score < threshold.

    The threshold is the largest observed calibration score t with
    share(calibration score < t) <= target. Tied scores are all rejected or all
    accepted together, so with coarse scores (kNN vote fractions) the achieved
    rate can sit well below the target: the deterministic conservative rule of
    VII.A. The achieved rate is always reported next to the threshold.
    """
    s = np.sort(np.asarray(calibration_scores, dtype=float))
    s = s[np.isfinite(s)]
    if len(s) == 0:
        return float("nan"), float("nan")
    candidates = np.unique(s)
    frr = np.searchsorted(s, candidates, side="left") / len(s)
    threshold = float(candidates[frr <= target].max())
    return threshold, float(np.mean(s < threshold))


def expected_calibration_error(confidence, correct, bins: int = ECE_BINS) -> float:
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    if len(confidence) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        members = which == b
        if members.any():
            total += members.mean() * abs(correct[members].mean() - confidence[members].mean())
    return float(total)


def auroc_low_score(unseen_scores, seen_scores) -> float:
    """P(an unseen cell has a lower top score than a seen cell), ties counted half."""
    from scipy.stats import rankdata

    unseen, seen = np.asarray(unseen_scores, dtype=float), np.asarray(seen_scores, dtype=float)
    if len(unseen) == 0 or len(seen) == 0:
        return float("nan")
    ranks = rankdata(np.concatenate([-unseen, -seen]))   # high rank = low score
    u = ranks[: len(unseen)].sum() - len(unseen) * (len(unseen) + 1) / 2
    return float(u / (len(unseen) * len(seen)))


def multiclass_brier(probabilities: np.ndarray, vocabulary: list[str], true_labels) -> float:
    """Mean over cells of sum_k (p_k - y_k)^2; every true label must be in the vocabulary."""
    if len(probabilities) == 0:
        return float("nan")
    position = {c: i for i, c in enumerate(vocabulary)}
    onehot = np.zeros_like(probabilities, dtype=float)
    onehot[np.arange(len(probabilities)), [position[c] for c in true_labels]] = 1.0
    return float(np.mean(np.sum((probabilities - onehot) ** 2, axis=1)))


def class_balanced_rate(flags: pd.Series, classes: pd.Series) -> float:
    """Mean over classes of the per-class share of True flags."""
    if len(flags) == 0:
        return float("nan")
    return float(flags.astype(float).groupby(classes.to_numpy()).mean().mean())


def query_token(split_id: str) -> str:
    return split_id.split("__")[2].removeprefix("qry-")


# --------------------------------------------------------------------------- truth
def load_truth(out_root: Path, split_version: str, record: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(query truth, calibration truth): dataset, cell_id, label_harmonized, role."""
    split = read_split(split_path(Path(out_root) / "splits", split_version, record["split_id"]))
    columns = ["dataset", "cell_id", "label_harmonized", "role"]
    calibration = split.loc[split["role"] == "calibration", columns]
    if record["reference"] == record["query"]:
        query = split.loc[split["role"].isin(QUERY_ROLES), columns]
    else:
        scenario_id = record["split_id"].rsplit("__seed-", 1)[0]
        query_set = read_query_set(query_set_path(Path(out_root) / "query_sets", split_version, scenario_id))
        query = query_set.loc[query_set["role"].isin(QUERY_ROLES), columns]
    return query.reset_index(drop=True), calibration.reset_index(drop=True)


# --------------------------------------------------------------------------- one run
def evaluate_run(run_dir: Path, record: dict, query_truth: pd.DataFrame, calibration_truth: pd.DataFrame) -> dict:
    """Metric tables for one successful run."""
    from sklearn.metrics import f1_score, precision_recall_fscore_support

    pred = read_tsv(run_dir / "predictions.tsv.gz")
    pred["top_score"] = pred["top_score"].replace("", "nan").astype(float)
    score_type = pred["score_type"].iloc[0] if len(pred) else ""
    query = pred[pred["cell_set"] == "query"].merge(query_truth, on=["dataset", "cell_id"], how="inner")
    calib = pred[pred["cell_set"] == "calibration"].merge(calibration_truth, on=["dataset", "cell_id"], how="inner")
    seen = query[query["role"] == "evaluation"]
    unseen = query[query["role"] == "evaluation_open_set"]
    classes = sorted(set(seen["label_harmonized"]))
    correct_seen = seen["predicted_label"] == seen["label_harmonized"]

    threshold, calibration_frr = common_threshold(calib["top_score"])
    rejected_seen = seen["top_score"] < threshold
    rejected_unseen = unseen["top_score"] < threshold
    frr = float(rejected_seen.mean()) if len(seen) else float("nan")
    trr = float(rejected_unseen.mean()) if len(unseen) else float("nan")
    scored = pd.concat([seen, unseen])
    summary = {
        "run_id": record["run_id"], "exp": record["exp"], "reference": record["reference"],
        "query": record["query"], "query_token": query_token(record["split_id"]),
        "setting": "within" if record["reference"] == record["query"] else "cross",
        "perturbation": record["perturbation"], "method": record["method"],
        "method_family": record.get("method_family", record["method"]), "seed": record["seed"],
        "score_type": score_type, "n_seen": len(seen), "n_unseen": len(unseen), "n_calibration": len(calib),
        "n_seen_classes": len(classes),
        "accuracy": float(correct_seen.mean()) if len(seen) else float("nan"),
        "macro_f1": (float(f1_score(seen["label_harmonized"], seen["predicted_label"], labels=classes,
                                    average="macro", zero_division=0)) if len(seen) else float("nan")),
        "calibration_accuracy": float((calib["predicted_label"] == calib["label_harmonized"]).mean()) if len(calib) else float("nan"),
        "threshold": threshold, "calibration_frr": calibration_frr, "query_frr": frr,
        "query_frr_class_balanced": class_balanced_rate(rejected_seen, seen["label_harmonized"]), "trr": trr,
        "balanced_rejection_accuracy": (trr + 1 - frr) / 2 if len(unseen) and len(seen) else float("nan"),
        "coverage": float((scored["top_score"] >= threshold).mean()) if len(scored) else float("nan"),
        "accepted_accuracy": float(correct_seen[~rejected_seen].mean()) if (~rejected_seen).any() else float("nan"),
        "auroc_unseen": auroc_low_score(unseen["top_score"], seen["top_score"]),
        "mean_confidence_seen": float(seen["top_score"].mean()) if len(seen) else float("nan"),
        "ece_seen": float("nan"), "ece_calibration": float("nan"), "brier_seen": float("nan"),
    }
    if score_type in PROBABILITY_SCORES:
        summary["ece_seen"] = expected_calibration_error(seen["top_score"], correct_seen)
        summary["ece_calibration"] = expected_calibration_error(
            calib["top_score"], calib["predicted_label"] == calib["label_harmonized"])
        scores = read_tsv(run_dir / "scores.tsv.gz")
        vocabulary = [c for c in scores.columns if c not in {"dataset", "cell_id", "cell_set", "score_type"}]
        merged = seen[["dataset", "cell_id", "label_harmonized"]].merge(
            scores[scores["cell_set"] == "query"], on=["dataset", "cell_id"], how="inner")
        summary["brier_seen"] = multiclass_brier(merged[vocabulary].astype(float).to_numpy(), vocabulary,
                                                 merged["label_harmonized"])

    per_class = []
    if len(seen):
        precision, recall, f1, support = precision_recall_fscore_support(
            seen["label_harmonized"], seen["predicted_label"], labels=classes, zero_division=0)
        for cls, p, r, f, n in zip(classes, precision, recall, f1, support):
            wrong = seen[(seen["label_harmonized"] == cls) & (seen["predicted_label"] != cls)]["predicted_label"]
            per_class.append({"run_id": record["run_id"], "class": cls, "n": int(n), "recall": float(r),
                              "precision": float(p), "f1": float(f),
                              "top_confusion": wrong.value_counts().index[0] if len(wrong) else ""})
    confusion = (scored.groupby(["role", "label_harmonized", "predicted_label"]).size().rename("n").reset_index()
                 .rename(columns={"label_harmonized": "true_label"}))
    confusion.insert(0, "run_id", record["run_id"])
    return {"summary": summary, "per_class": per_class, "confusion": confusion}


# --------------------------------------------------------------------------- the whole pilot
SUMMARY_METRICS = ["accuracy", "macro_f1", "calibration_accuracy", "threshold", "calibration_frr", "query_frr",
                   "query_frr_class_balanced", "trr", "balanced_rejection_accuracy", "coverage", "accepted_accuracy",
                   "auroc_unseen", "mean_confidence_seen", "ece_seen", "ece_calibration", "brier_seen"]


def evaluate_pilot(out_root: str | Path, split_version: str) -> dict[str, pd.DataFrame]:
    """Score every successful run in results/run_index.tsv; write results/metrics/*.tsv."""
    out_root = Path(out_root)
    index = read_tsv(out_root / "results" / "run_index.tsv")
    summaries, per_class, confusion, skipped = [], [], [], []
    truth_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for row in index.itertuples():
        run_dir = out_root / "results" / "runs" / row.run_id
        if row.status != "success":
            skipped.append({"run_id": row.run_id, "status": row.status})
            continue
        record = json.loads((run_dir / "run_record.json").read_text())
        if record["split_id"] not in truth_cache:
            truth_cache[record["split_id"]] = load_truth(out_root, split_version, record)
        tables = evaluate_run(run_dir, record, *truth_cache[record["split_id"]])
        summaries.append(tables["summary"])
        per_class.extend(tables["per_class"])
        confusion.append(tables["confusion"])
    run_summary = pd.DataFrame(summaries)
    keys = ["exp", "setting", "query", "perturbation", "method"]
    long = run_summary.melt(id_vars=keys + ["query_token", "seed"], value_vars=SUMMARY_METRICS,
                            var_name="metric", value_name="value").dropna(subset=["value"])
    scenario_summary = (long.groupby(keys + ["metric"])["value"]
                        .agg(n_runs="size", mean="mean", sd="std", min="min", max="max").reset_index())
    fold_summary = (long.groupby(keys + ["query_token", "metric"])["value"]
                    .agg(n_runs="size", mean="mean", sd="std", min="min", max="max").reset_index())
    tables = {
        "run_summary": run_summary, "scenario_summary": scenario_summary, "fold_summary": fold_summary,
        "per_class": pd.DataFrame(per_class),
        "confusion": pd.concat(confusion, ignore_index=True) if confusion else pd.DataFrame(),
        "skipped_runs": pd.DataFrame(skipped, columns=["run_id", "status"]),
    }
    folder = out_root / "results" / "metrics"
    for name, table in tables.items():
        write_tsv(table, folder / f"{name}.tsv")
    write_json({"split_version": split_version, "n_runs_scored": len(summaries), "n_runs_skipped": len(skipped),
                "target_frr": TARGET_FRR, "ece_bins": ECE_BINS, "created_utc": utc_now()},
               folder / "metrics_manifest.json")
    return tables
