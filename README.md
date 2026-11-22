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

**4. Run the pilot (Week 2).** One command runs QC, builds the gene universe, writes every split file, and runs every scenario × seed × method listed in `configs/pilot.yaml`:

```bash
python -m annotbench run --config configs/pilot.yaml
```

To write the QC reports, gene universe and split files only (no methods), so they can be reviewed first:

```bash
python -m annotbench prepare --config configs/pilot.yaml
```

**5. Check reproducibility.** Runs the pilot twice from scratch and compares everything; prints PASS/FAIL per check:

```bash
python scripts/check_reproducibility.py --config configs/pilot.yaml --out results/reproducibility
```

**6. Tests** (toy data only, under a minute):

```bash
python -m unittest discover -s tests -t .
```

## Where the outputs go

| Output | Path | Format |
| --- | --- | --- |
| QC metrics, rules and retention by dataset × donor × class | `reports/week2/qc/` | TSV |
| Normalization constants | `reports/week2/normalization_constants.json` | JSON |
| Gene universe and excluded features | `gene_universes/pancreas_pilot_v1*.tsv` | TSV (+ JSON sidecar) |
| Split files, counts, restrictions, manifest | `splits/pilot_v1/` | `docs/io_spec.md` §5 |
| Predictions, scores, fitted state, run record | `results/runs/<run_id>/` | `docs/io_spec.md` §2, §5, §6 |
| Eligibility and failure matrix | `results/run_index.tsv` | one row per run |

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
| Hashes, deterministic writing | `annotbench/provenance.py` |

Open decisions for the pilot are listed in `docs/week2_decisions.md`. The interface contract every method follows is `docs/io_spec.md`.
