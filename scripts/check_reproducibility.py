"""Reproducibility check for the Week 2 pilot (guide Step 6).

"I ran it twice and it looked the same" is not evidence. This script:

1. runs the documented entry point twice, in two fresh processes, into two folders:
       python -m annotbench run --config <config> --out-root <out>/run_a
       python -m annotbench run --config <config> --out-root <out>/run_b
2. compares the two runs file by file (Check 2):
       split files, gene universe, HVG lists, predicted labels  -> exactly equal
       scores, top scores, PCA loadings                          -> equal within --tolerance
       run records                                               -> equal except timings, timestamps, host
3. compares seed 0 with every other seed inside run_a (Check 3):
       must change:     which cells are train vs train_reserve (the cap draw)
       must not change: donor roles, eligible cells, calibration and query cells, universe, vocabulary
4. prints PASS / FAIL per check with the compared hashes, and writes
   <out>/reproducibility_report.json and .md. Exit code 1 if anything fails.

Usage (from the repo root):
    python scripts/check_reproducibility.py --config configs/pilot.yaml --out results/reproducibility
    python scripts/check_reproducibility.py --skip-runs --out results/reproducibility   # re-compare only
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from annotbench.provenance import read_tsv, sha256_content, utc_now, write_json  # noqa: E402

VOLATILE_RECORD_KEYS = {"stages", "started_utc", "finished_utc", "host", "git_dirty", "traceback"}
EXACT_TABLE_COLUMNS = {"predictions.tsv.gz": ["dataset", "cell_id", "cell_set", "predicted_label",
                                              "predicted_cl_id", "prediction_level", "rejected", "score_type"]}


class Report:
    def __init__(self):
        self.rows = []

    def add(self, check: str, passed: bool, detail: str):
        self.rows.append({"check": check, "result": "PASS" if passed else "FAIL", "detail": detail})
        print(f"[{'PASS' if passed else 'FAIL'}] {check}: {detail}")

    @property
    def failed(self) -> int:
        return sum(r["result"] == "FAIL" for r in self.rows)


def run_entry_point(config: str, out: Path, extra: list[str]) -> None:
    command = [sys.executable, "-m", "annotbench", "run", "--config", config, "--out-root", str(out), *extra]
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO, check=True, env={**os.environ, "PYTHONWARNINGS": "ignore"})


def numeric_close(a: pd.DataFrame, b: pd.DataFrame, columns: list[str], tolerance: float) -> tuple[bool, float]:
    if not columns:
        return True, 0.0
    x = a[columns].replace("", np.nan).astype(float).to_numpy()
    y = b[columns].replace("", np.nan).astype(float).to_numpy()
    same_nan = np.array_equal(np.isnan(x), np.isnan(y))
    diff = np.nanmax(np.abs(x - y)) if np.isfinite(x).any() else 0.0
    return bool(same_nan and diff <= tolerance), float(diff)


def compare_runs(a: Path, b: Path, report: Report, tolerance: float) -> None:
    split_files = sorted((a / "splits").rglob("*.tsv.gz"))
    same = [sha256_content(f) == sha256_content(b / f.relative_to(a)) for f in split_files]
    report.add("split files identical", bool(split_files) and all(same),
               f"{sum(same)}/{len(split_files)} files with equal decompressed content")
    for f in sorted((a / "gene_universes").glob("*.tsv")):
        report.add(f"gene universe {f.name} identical", sha256_content(f) == sha256_content(b / f.relative_to(a)),
                   f"sha256 {sha256_content(f)[:16]}")

    runs = sorted(p.name for p in (a / "results" / "runs").iterdir() if p.is_dir())
    for run_id in runs:
        ra, rb = a / "results" / "runs" / run_id, b / "results" / "runs" / run_id
        problems = []
        rec_a = {k: v for k, v in json.loads((ra / "run_record.json").read_text()).items() if k not in VOLATILE_RECORD_KEYS}
        rec_b = {k: v for k, v in json.loads((rb / "run_record.json").read_text()).items() if k not in VOLATILE_RECORD_KEYS}
        if rec_a != rec_b:
            problems.append("run records differ in " + ", ".join(sorted(k for k in rec_a if rec_a.get(k) != rec_b.get(k))))
        if rec_a.get("status") != "success":
            problems.append(f"status {rec_a.get('status')}")
        hvg = ra / "fitted" / "hvg.tsv"
        if hvg.exists() and read_tsv(hvg)["ncbi_gene_id"].tolist() != read_tsv(rb / "fitted" / "hvg.tsv")["ncbi_gene_id"].tolist():
            problems.append("HVG lists differ")
        max_diff = 0.0
        if (ra / "predictions.tsv.gz").exists():
            pa, pb = read_tsv(ra / "predictions.tsv.gz"), read_tsv(rb / "predictions.tsv.gz")
            exact = EXACT_TABLE_COLUMNS["predictions.tsv.gz"]
            if not pa[exact].equals(pb[exact]):
                problems.append("predicted labels differ")
            ok, diff = numeric_close(pa, pb, ["top_score"], tolerance)
            max_diff = max(max_diff, diff)
            if not ok:
                problems.append(f"top_score differs by {diff:.3g}")
        if (ra / "scores.tsv.gz").exists():
            sa, sb = read_tsv(ra / "scores.tsv.gz"), read_tsv(rb / "scores.tsv.gz")
            classes = [c for c in sa.columns if c not in {"dataset", "cell_id", "cell_set", "score_type"}]
            if list(sa.columns) != list(sb.columns) or not sa["cell_id"].equals(sb["cell_id"]):
                problems.append("score tables have different cells or columns")
            else:
                ok, diff = numeric_close(sa, sb, classes, tolerance)
                max_diff = max(max_diff, diff)
                if not ok:
                    problems.append(f"scores differ by {diff:.3g}")
        loadings = ra / "fitted" / "pca_loadings.tsv.gz"
        if loadings.exists():
            la, lb = read_tsv(loadings), read_tsv(rb / "fitted" / "pca_loadings.tsv.gz")
            ok, diff = numeric_close(la, lb, [c for c in la.columns if c.startswith("PC")], tolerance)
            max_diff = max(max_diff, diff)
            if not ok:
                problems.append(f"PCA loadings differ by {diff:.3g}")
        byte_identical = all(
            (ra / n).read_bytes() == (rb / n).read_bytes()
            for n in ("predictions.tsv.gz", "scores.tsv.gz") if (ra / n).exists()
        )
        report.add(f"same seed reproduces {run_id}", not problems,
                   "; ".join(problems) or f"max numeric difference {max_diff:.3g}; byte-identical: {byte_identical}")


def compare_seeds(a: Path, report: Report) -> None:
    splits = sorted((a / "splits").rglob("*__seed-*.tsv.gz"))
    groups: dict[str, list[Path]] = {}
    for path in splits:
        groups.setdefault(path.name.split("__seed-")[0], []).append(path)
    for scenario, paths in sorted(groups.items()):
        paths = sorted(paths)
        base = read_tsv(paths[0]).set_index(["dataset", "cell_id"])
        for other_path in paths[1:]:
            other = read_tsv(other_path).set_index(["dataset", "cell_id"]).loc[base.index]
            changed = base["role"] != other["role"]
            moved = set(base.loc[changed, "role"]) | set(other.loc[changed, "role"])
            merged = lambda r: r.replace("train_reserve", "train")  # noqa: E731
            donor_roles_a = set(zip(base["donor"], merged(base["role"])))
            donor_roles_b = set(zip(other["donor"], merged(other["role"])))
            label = f"{scenario} seed {paths[0].name[-9:-7]} vs {other_path.name[-9:-7]}"
            report.add(f"{label}: train/train_reserve draw changes", bool(changed.any()),
                       f"{int(changed.sum())} cells moved between {sorted(moved) or 'no roles'}")
            report.add(f"{label}: only train vs train_reserve moved", moved <= {"train", "train_reserve"},
                       f"roles touched: {sorted(moved)}")
            report.add(f"{label}: donor roles unchanged", donor_roles_a == donor_roles_b,
                       f"{len(donor_roles_a)} (donor, role) pairs")
            for role in ("calibration", "evaluation", "evaluation_open_set", "excluded"):
                same = set(base.index[base["role"] == role]) == set(other.index[other["role"] == role])
                report.add(f"{label}: {role} cells unchanged", same, f"{int((base['role'] == role).sum())} cells")
            vocab_a = sorted(set(base.loc[base["role"] == "train", "label_harmonized"]))
            vocab_b = sorted(set(other.loc[other["role"] == "train", "label_harmonized"]))
            report.add(f"{label}: vocabulary unchanged", vocab_a == vocab_b, ", ".join(vocab_a))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--out", default="results/reproducibility")
    parser.add_argument("--tolerance", type=float, default=1e-10,
                        help="largest allowed absolute difference for scores and loadings (BLAS threading can "
                             "change the last digits; set OMP_NUM_THREADS=1 to demand exact equality)")
    parser.add_argument("--skip-runs", action="store_true", help="only compare existing run_a and run_b")
    parser.add_argument("--exp", action="append", help="pass through to `annotbench run --exp`")
    parser.add_argument("--method", action="append", help="pass through to `annotbench run --method`")
    args = parser.parse_args(argv)
    out = (REPO / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    extra = sum((["--exp", e] for e in args.exp or []), []) + sum((["--method", m] for m in args.method or []), [])
    if not args.skip_runs:
        for name in ("run_a", "run_b"):
            run_entry_point(args.config, out / name, extra)
    report = Report()
    compare_runs(out / "run_a", out / "run_b", report, args.tolerance)
    compare_seeds(out / "run_a", report)
    summary = {"created_utc": utc_now(), "config": args.config, "tolerance": args.tolerance,
               "omp_num_threads": os.environ.get("OMP_NUM_THREADS"), "checks": report.rows,
               "n_checks": len(report.rows), "n_failed": report.failed}
    write_json(summary, out / "reproducibility_report.json")
    table = pd.DataFrame(report.rows)
    lines = [f"# Reproducibility report", "", f"Config `{args.config}`, tolerance {args.tolerance}, "
             f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}.", "",
             f"**{len(report.rows) - report.failed} of {len(report.rows)} checks passed.**", "",
             "| check | result | detail |", "| --- | --- | --- |"]
    lines += [f"| {r.check} | {r.result} | {r.detail} |" for r in table.itertuples()]
    (out / "reproducibility_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{'PASS' if not report.failed else 'FAIL'}: {len(report.rows) - report.failed}/{len(report.rows)} checks passed. "
          f"Report: {out / 'reproducibility_report.md'}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
