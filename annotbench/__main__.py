"""Command line for the benchmark pipeline. Run `python -m annotbench --help` from the repo root."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import config
from .environment import check_inputs, check_python, check_r
from .integrity import IntegrityError, raise_if_failed, run_integrity_checks
from .labels import LabelMapError, load_label_map
from .loading import DatasetLoadError, describe_dataset, load_dataset, load_gene_reference_from_registry
from .preprocess import PreprocessError
from .reporting import write_dataset_report, write_integrity_report
from .splits import SplitError
from .universe import UniverseError

PILOT_DATASETS = ["baron", "muraro"]


def relative_to_repo(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(config.REPO_ROOT))
    except ValueError:
        return str(path)


def audit_dataset(name: str, args, registry: dict, label_map: pd.DataFrame, gene_reference):
    """Load one dataset, write its reports, and stop if a fatal check failed."""
    root = config.resolve_data_root(args.data_root)
    cfg = config.dataset_config(registry, name)
    adata = load_dataset(name, root, registry, gene_reference)
    directory = Path(args.reports_dir) / "integrity"
    directory.mkdir(parents=True, exist_ok=True)
    write_dataset_report(describe_dataset(adata, root), directory)
    results = run_integrity_checks(adata, cfg, registry, label_map, root)
    report = write_integrity_report(name, results, adata, directory)
    raise_if_failed(name, results, relative_to_repo(report))
    failed_warnings = sum(1 for r in results if r.passed is False)
    print(f"{name}: {adata.n_obs:,} cells x {adata.n_vars:,} features, {len(results)} checks, "
          f"0 fatal failures, {failed_warnings} warnings to review -> {relative_to_repo(report)}")
    return adata


def command_check_env(args) -> int:
    table = pd.concat([check_python(), check_r()], ignore_index=True)
    print(table.to_string(index=False))
    missing = table[table["found"] == "MISSING"]
    if len(missing):
        print(f"\nMissing: {', '.join(missing['tool'])}. See docs/environment.md.", file=sys.stderr)
        return 2
    return 0


def command_check_inputs(args) -> int:
    root = config.resolve_data_root(args.data_root)
    table = check_inputs(root, Path(args.input_manifest))
    print(table[["relative_path", "status"]].to_string(index=False))
    directory = Path(args.reports_dir)
    directory.mkdir(parents=True, exist_ok=True)
    table.to_csv(directory / "input_check.tsv", sep="\t", index=False)
    bad = table[table["status"] != "ok"]
    if len(bad):
        print(f"\nINPUT STOP: {len(bad)} input(s) missing or changed. See docs/environment.md.", file=sys.stderr)
        return 2
    return 0


def command_integrity(args) -> int:
    registry = config.load_registry(args.registry)
    label_map = load_label_map(args.label_map)
    reference = load_gene_reference_from_registry(registry, config.resolve_data_root(args.data_root))
    # `week1` reuses this command but has no --dataset option, so the attribute may be absent.
    for name in getattr(args, "dataset", None) or PILOT_DATASETS:
        audit_dataset(name, args, registry, label_map, reference)
    return 0


def command_week1(args) -> int:
    for step in (command_check_env, command_check_inputs, command_integrity):
        status = step(args)
        if status:
            return status
    return 0


def command_run(args) -> int:
    """Week 2: QC, universe, splits, then every scenario x seed x method (or stop after splits)."""
    from .runner import load_pilot_config, run_pilot

    cfg = load_pilot_config(args.config)
    index = run_pilot(cfg, out_root=args.out_root, data_root=args.data_root, exp_filter=args.exp,
                      method_filter=args.method, prepare_only=args.command == "prepare",
                      seed_filter=args.seed, jobs=args.jobs)
    if args.command == "prepare":
        return 0
    failed = index[index["status"] != "success"]
    print(f"\n{len(index)} runs, {len(index) - len(failed)} succeeded, {len(failed)} recorded as failures.")
    return 0


def command_evaluate(args) -> int:
    """Week 2: score every successful run against labels from the split / query set."""
    from . import config as project_config
    from .metrics import evaluate_pilot
    from .runner import load_pilot_config

    cfg = load_pilot_config(args.config)
    out_root = Path(args.out_root) if args.out_root else project_config.REPO_ROOT
    tables = evaluate_pilot(out_root, cfg.split_version)
    summary = tables["scenario_summary"]
    shown = summary[summary["metric"].isin(["accuracy", "macro_f1", "balanced_rejection_accuracy"])]
    print(shown.pivot_table(index=["exp", "setting", "perturbation", "method"], columns="metric", values="mean")
          .round(3).to_string())
    print(f"\n{len(tables['run_summary'])} runs scored -> {out_root / 'results' / 'metrics'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="annotbench", description=__doc__)
    parser.add_argument("--data-root", default=None, help=f"data folder (default: ${config.DATA_ROOT_ENV}, then <repo>/data)")
    parser.add_argument("--registry", default=str(config.DEFAULT_REGISTRY))
    parser.add_argument("--label-map", default=str(config.DEFAULT_LABEL_MAP))
    parser.add_argument("--input-manifest", default=str(config.DEFAULT_INPUT_MANIFEST))
    parser.add_argument("--reports-dir", default=str(config.DEFAULT_REPORTS))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check-env", help="compare installed tools with the tested versions").set_defaults(run=command_check_env)
    commands.add_parser("check-inputs", help="verify input files against their checksums").set_defaults(run=command_check_inputs)
    integrity = commands.add_parser("integrity", help="load datasets and run integrity checks")
    integrity.add_argument("--dataset", action="append", choices=PILOT_DATASETS, help="repeat to name several (default: all)")
    integrity.set_defaults(run=command_integrity)
    commands.add_parser("week1", help="run every Week 1 step in order").set_defaults(run=command_week1)
    for name, text in (("run", "run the pilot: QC, universe, splits and every scenario x seed x method"),
                       ("prepare", "QC, gene universe and split files only (no methods)")):
        sub = commands.add_parser(name, help=text)
        sub.add_argument("--config", default="configs/pilot.yaml", help="pilot configuration file")
        sub.add_argument("--out-root", default=None, help="where splits/, gene_universes/, results/ and reports/week2/ go (default: repo root)")
        sub.add_argument("--exp", action="append", choices=["e1", "e2", "e3"], help="only these experiments (repeatable)")
        sub.add_argument("--method", action="append", help="only these method instances or families (repeatable)")
        sub.add_argument("--seed", action="append", type=int, help="only these seeds (repeatable)")
        sub.add_argument("--jobs", type=int, default=1, help="runs in parallel (worker processes); outputs do not change")
        sub.set_defaults(run=command_run)
    evaluate = commands.add_parser("evaluate", help="score the saved predictions (writes results/metrics/)")
    evaluate.add_argument("--config", default="configs/pilot.yaml", help="pilot configuration file")
    evaluate.add_argument("--out-root", default=None, help="the --out-root the pilot was run with (default: repo root)")
    evaluate.set_defaults(run=command_evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.run(args)
    except (IntegrityError, config.MissingInputError, DatasetLoadError, LabelMapError, SplitError, UniverseError,
            PreprocessError) as error:
        print(f"\n{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
