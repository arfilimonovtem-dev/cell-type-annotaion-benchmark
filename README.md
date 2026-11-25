# cell-type-annotaion-benchmark

Hierarchy-Aware Open-Set Evaluation of Reference-Based Cell Type Annotation under Dataset Shift in scRNA-seq.

This repository holds the benchmark code. Raw and derived data are not in git (see `docs/environment.md`).

## Quick start (Baron → Muraro pilot)

All commands run from the repository root.

**1. Create the environment**

```bash
conda env create -f environment.yml
conda activate annotbench-pilot
```

**2. Point the code at the data.** The layout and the origin of every input file are in `docs/environment.md`; checksums are in `manifests/pilot_inputs.tsv`.

```bash
export ANNOTBENCH_DATA_ROOT=/path/to/data     # or pass --data-root to every command
```

**3. Verify the inputs and the data (Week 1)**

```bash
python -m annotbench check-env
python -m annotbench check-inputs
python -m annotbench integrity
```

**4. Run the pilot (Week 2).** One command runs QC, builds the gene universe, writes every split file and query set, and runs every scenario × seed × method instance listed in `configs/pilot.yaml` (300 runs; about an hour with 5 workers on 8 cores):

```bash
python -m annotbench run --config configs/pilot.yaml --jobs 5
```

`--jobs` sets how many runs work in parallel; it never changes a result. `--exp`, `--method` and `--seed` (all repeatable) run a subset.

**5. Score the pilot.** Joins the saved predictions with the true labels from the split files and query sets, and writes the metric tables:

```bash
python -m annotbench evaluate --config configs/pilot.yaml
```

To write the QC reports, gene universe, split files and query sets only (no methods), so they can be reviewed first:

```bash
python -m annotbench prepare --config configs/pilot.yaml
```

**6. Check reproducibility.** Runs the pilot twice from scratch (with a different number of workers each time) and compares everything; prints PASS/FAIL per check. Week 2 checked seeds 0 and 1:

```bash
python scripts/check_reproducibility.py --config configs/pilot.yaml --out results/reproducibility --seed 0 --seed 1 --jobs 6 --jobs-b 3
```

**7. Tests** (toy data only, under a minute):

```bash
python -m unittest discover -s tests -t .
```

## Where the outputs go

| Output | Path | Format |
| --- | --- | --- |
| QC metrics, rules and retention by dataset × donor × class | `reports/week2/qc/` | TSV |
| Normalization constants | `reports/week2/normalization_constants.json` | JSON |
| Gene universe and excluded features | `gene_universes/pancreas_pilot_v1*.tsv` | TSV (+ JSON sidecar) |
| Split files (reference cells only), counts, restrictions, manifest | `splits/pilot_v2/` | `docs/io_spec.md` §5 |
| Query sets (query-dataset cells of E2/E3, no seed) | `query_sets/pilot_v2/` | `docs/io_spec.md` §5 |
| Predictions, scores, fitted state (incl. `hvg_report.json`), run record | `results/runs/<run_id>/` | `docs/io_spec.md` §2, §5, §6 |
| Eligibility and failure matrix | `results/run_index.tsv` | one row per run |
| HVG anomaly warnings of every run | `results/hvg_anomalies.tsv` | TSV |
| Metrics (per run, per scenario, per fold, per class, confusion) | `results/metrics/` | TSV |
| Week 2 draft (pilot_v1) outputs, kept unchanged | `archive/pilot_v1/` | see its README |

Use `--out-root DIR` to send all of these somewhere else.

## Where the code is

| Step | Module |
| --- | --- |
| Loading and labels (Week 1) | `annotbench/loading.py`, `annotbench/labels.py`, `annotbench/genes.py` |
| Per-cell QC | `annotbench/qc.py` |
| Splits | `annotbench/splits.py` |
| Gene universe and alignment | `annotbench/universe.py` |
| Preprocessing (normalize, HVG, scale, PCA) | `annotbench/preprocess.py` |
| Method adapters | `annotbench/methods/` (`pca_knn.py`, `linear_svm.py`) |
| Scenario runner | `annotbench/runner.py` |
| Pilot metrics | `annotbench/metrics.py` |
| Hashes, deterministic writing | `annotbench/provenance.py` |

Every pilot decision (what, where, why, who) is in `docs/week2_decisions.md`. The interface contract every method follows is `docs/io_spec.md`.
