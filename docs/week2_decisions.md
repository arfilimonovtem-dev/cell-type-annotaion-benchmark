# Week 2 pilot: decision log (pilot_v2)

Status: **fixed on 2026-11-23.** Every value below is set in `configs/pilot.yaml` (or in the file named in "Where"). Nothing in `annotbench/` hard-codes a number. Changing a decision means changing one config line, giving the run a new `split_version`, and re-running.

- **Team** = decided by the team leader and Artem (2026-11-19 or 2026-11-23).
- **Mentor** = Artem asked for "the best option in each case, stated with what, where and why" (2026-11-23, item 9). These choices were made in the pilot_v2 rework and can be overruled.
- **Rule** = forced by the proposal or the data; no real choice.

The Week 2 draft (pilot_v1) and its outputs are in `archive/pilot_v1/` with `configs/pilot_v1.yaml`. The draft version of this file is in git history (commit `745b572`).

## What changed from the draft (v1 → v2)

| | v1 draft | v2 (now) |
| --- | --- | --- |
| Who trains | Baron | Baron only, all 13 classes |
| Muraro in the split file | yes (as `evaluation`) | **no**. Muraro is listed in a separate, seed-free query set |
| E1 | human4 held out | **every Baron donor held out once** (4 folds) |
| E3 | Baron → Muraro only | Baron → Muraro **plus** the matched within-Baron control (4 folds) |
| Normalization | 10,000 per cell | **each dataset scaled to its own median library size** |
| kNN | k = 15, uniform votes | **k = 5 and k = 10**, distance-weighted |
| SVM | placeholder, one-vs-rest Platt | **final**, temperature scaling |
| HVGs | `seurat` | `seurat`, plus an **anomaly report** for every run |
| Muraro `mesenchymal` | excluded | **stellate** (`labels/label_map_v2.csv`) |
| Seeds | 0, 1 | **0–9** (proposal VI.B) |
| Runs | 12 | **300** (10 scenarios × 10 seeds × 3 method instances) |
| Metrics | none (on purpose) | `python -m annotbench evaluate` (accuracy, macro-F1, BRA, ECE, …) |

## Decisions

| # | Decision | Value (where) | Why | Who |
| --- | --- | --- | --- | --- |
| A | When exam metrics are computed | once, after every decision below was fixed | looking earlier invites tuning on the test set | Team (2026-11-22) |
| 1 | UMI bounds | 200–100,000 (`qc.default`) | removes empty droplets and 5 labelled Muraro cells with 104k–210k UMIs (likely doublets) | Team |
| 2 | Ribosomal filter | off (`max_pct_ribo: null`) | RP share is cell-type biology here: a 20% cut removes 57% of T cells, 56% of mast, 49% of acinar | Mentor |
| 3 | Hb and genes-detected filters | off | Hb is ~0 in both datasets; the UMI bounds already remove empty and huge cells | Mentor |
| 4 | Mitochondrial filter | not applicable | neither deposit has MT features; reported `not_applicable`, not silently passed | Rule |
| 5 | Fixed vs MAD thresholds | fixed | MAD thresholds adapt to each dataset's own distribution, which hides differences we want to see | Mentor |
| 6 | Normalization target | **`target_sum: dataset_median`** (Baron 5,016; Muraro 13,230) | each dataset scales to its own median library size. Declared as a query-adaptive step in every run record, because the Muraro median is computed from Muraro cells (all of them, no labels) | Team (2026-11-23) |
| 7 | Genes counted in library size | universe genes (`library_size: universe`) | the same gene set in both datasets; a median 95% of each Muraro cell's UMIs fall on universe genes | Mentor |
| 8 | HVG flavor | **`seurat`**, record anomalies (`hvg_flavor`, `hvg_watch_genes`) | kept the original; every run writes `fitted/hvg_report.json` and `results/hvg_anomalies.tsv` collects the warnings (see below) | Team (2026-11-23) |
| 8b | HVG donor batch key | none (`hvg_batch_key: null`) | with only 2–3 training donors per fold, per-donor HVG selection is noisy; donor-dominated HVGs are reported instead | Mentor |
| 8c | Watched genes | GCG, INS, SST, PPY, GHRL, PRSS1, CPA1, KRT19, PECAM1, PDGFRB, SOX10, C1QB, TPSAB1, CD3E, PTPRC | one textbook marker per class plus PTPRC; the report says where each one ranks | Mentor |
| 9 | Clip after scaling | 10 | scanpy convention | Mentor |
| 10 | HVGs, PCs | 2,000 and 50 | | Team (2026-11-19) |
| 11 | k for kNN | **k = 5 and k = 10**, two method instances (`pca_knn.k5`, `pca_knn.k10`) | compare downstream instead of choosing now; inner-CV macro-F1 was 0.736 (k=5) and 0.675 (k=10) | Team (2026-11-23) |
| 12 | kNN vote weights | distance (`weights: distance`) | same inner-CV macro-F1 as uniform, and continuous scores, so the 5% rejection threshold is not stuck on ~k+1 steps | Mentor |
| 13 | Calibration donor | **human1**, except **human2** in the folds where human1 or human3 is the test donor (scenario list) | human1 gives ≥ 20 calibration cells to 8 of the 9 classes Baron and Muraro share (epsilon never can). When human3 is the test donor, human3's 843 acinar cells are gone; calibrating on human1 would take 110 of the 115 left and leave 5 acinar training cells. human2 has 3 acinar, so training keeps 112 | Mentor (asked to choose, item 4) |
| 14 | E1 test donors | **rotate: human1, human2, human3, human4** (4 scenarios) | leave-one-donor-out (proposal VI.A); one held-out donor is one sample of donor variability, not an estimate of it | Team (2026-11-23) |
| 15 | Class removed in E3 | **delta** | Team (2026-11-23) |
| 15b | E3 setting | Baron → Muraro **and** within-Baron control, same 4 folds as E1 | H1 compares rejection under transfer with rejection without transfer; without the within-dataset control it has nothing to compare | Mentor |
| 16 | Seeds | 0–9 | proposal VI.B / VIII | Rule |
| 17 | Support thresholds | 20 calibration (VII.C), 20 train, 10 evaluation; `support_policy: flag` | weak classes stay in and are reported in `*.restrictions.tsv`, never silently dropped | Mentor |
| 18 | Training classes | **all 13 Baron classes** (scenario field `train_classes`, default `all`) | Baron is the reference. Classes Muraro does not have (macrophage, mast, Schwann, T cell) stay in the vocabulary; a Muraro cell predicted as one of them is simply wrong | Team (2026-11-23) |
| 19 | Muraro in training and splits | **never**. The split file holds Baron only; Muraro cells are listed in `query_sets/pilot_v2/<scenario>.tsv.gz` | Muraro is the query dataset, used only to test dataset shift | Team (2026-11-23) |
| 20 | Features sharing one NCBI Gene ID | dropped (43 Baron, 98 Muraro features) | summing them would need a rule per gene family; dropping is reported in `gene_universes/*_excluded.tsv` | Mentor |
| 21 | Universe | pairwise Baron × Muraro, `status: provisional` (16,847 genes) | the 4-study pancreas universe comes when Segerstolpe and Xin are registered | Mentor |
| 22 | Muraro expression unit | recovered UMIs (`umi`) | same unit as Baron. Known limit: saturation at 256 UMIs per gene and cell (INS in beta cells). Notebook "Testing on Muraro" shows how to check its effect | Mentor (leader was checking) |
| 23 | Muraro `mesenchymal` (80 cells) | **stellate** (`labels/label_map_v2.csv`; `label_task: pancreas_harmonized_v2`) | they express PDGFRB, COL1A1/2, COL3A1, LUM, DCN, PDGFRA and lack endothelial, immune, Schwann and epithelial markers: the profile of Baron's (activated) stellate cells | Mentor |
| 24 | SVM | **final**: `LinearSVC` (liblinear, one-vs-rest), C = 1, no class weights, `scaled_hvg`, `dual: true` | this is the project's SVM | Team (2026-11-23) |
| 24b | SVM probabilities | **temperature scaling** on donor-grouped out-of-fold decision scores (`probability_calibration: temperature_inner_folds`) | it never changes the SVM's predicted class; on held-out Baron donors it beat one-vs-rest Platt on log-loss (0.65 vs 1.00), ECE (0.13 vs 0.20) and Brier score (0.24 vs 0.32). Proposal H2 names temperature scaling as the supporting calibration analysis | Mentor |
| 25 | Strict inner CV | refit HVG/scaler/PCA on inner-training donors (`inner_cv.refit_preprocessing: true`) | no information crosses inner folds | Mentor |
| 26 | Timeouts | not implemented | needs one process per run; comes with the workflow manager | later |
| 27 | Parallel runs | `--jobs N` worker processes; `threads_per_run: 1` | each run pins its numerical libraries to one thread, so outputs are byte-identical for any `--jobs` (the reproducibility check runs with 6 and 3 workers) | Mentor |
| 28 | `cell_set` column in predictions and scores | approved | calibration cells need predictions for the VII.A threshold; the column keeps them out of evaluation metrics | Mentor |
| 29 | What goes into git | code, configs, labels, docs, tests; `splits/`, `query_sets/`, `gene_universes/` (small versioned definitions, proposal IX); `results/run_index.tsv`, `results/pilot_manifest.json`, `results/hvg_anomalies.tsv`, `results/metrics/`, the reproducibility report. **Not** `results/runs/` or the reproducibility run folders (derived, regenerable, hundreds of MB; `.gitignore`) | Mentor |
| 30 | Pilot metrics | see "How the pilot is scored" below (`annotbench/metrics.py`) | Mentor |

## The split design in one picture

```text
Baron (reference, 4 donors)                     Muraro (query, 4 donors)
 ├─ split file, one per scenario × seed           └─ query set, one per scenario (no seed)
 │   train / train_reserve   (fit)                    evaluation          (scored, seen class)
 │   calibration             (thresholds only)        evaluation_open_set (scored, unseen class: delta in E3)
 │   evaluation(_open_set)   (E1/E3-within only:      challenge_unlabeled (uncertain labels)
 │                            the held-out donor)     excluded            (author QC / our QC)
 │   excluded                (removed class, QC)
```

The query set does not depend on the seed: the seed only decides which Baron cells above the 500-per-class cap are `train` and which are `train_reserve`.

| Scenario | Test cells | Calibration donor | Training donors |
| --- | --- | --- | --- |
| E1, test human1 | human1 (1,937) | human2 | human3, human4 |
| E1, test human2 | human2 (1,724) | human1 | human3, human4 |
| E1, test human3 | human3 (3,605) | human2 | human1, human4 |
| E1, test human4 | human4 (1,303) | human1 | human2, human3 |
| E2 | Muraro (2,117 labelled) | human1 | human2–4 |
| E3 cross | Muraro (1,924 seen + 193 delta) | human1 | human2–4, delta removed |
| E3 within × 4 | as E1, delta of the test donor is the unseen type | as E1 | as E1, delta removed |

## How the pilot is scored (`python -m annotbench evaluate`)

- **Closed-set**, on seen query cells: accuracy, macro-F1 over the classes the query actually has, per-class recall/precision/F1, confusion matrix.
- **Common rejection threshold** (VII.A): the largest top-score threshold that rejects at most 5% of the calibration cells. Tied scores are kept together, so the achieved rate can be below 5%; it is always reported. With this threshold: FRR on seen query cells, TRR on unseen ones, balanced rejection accuracy BRA = (TRR + 1 − FRR) / 2 (H1), coverage, accuracy of accepted cells.
- **AUROC** of "low top score" for unseen vs seen cells: unseen-type detection without a threshold.
- **Class-balanced FRR** (mean of per-class FRRs), next to the plain rate (VII.A). With delta as the only unseen class, class-balanced TRR equals TRR.
- **ECE** (10 bins), **multiclass Brier score** and mean confidence, for the SVM only. kNN vote fractions are scores, not probabilities (proposal V).
- All of it is descriptive. The paired summaries, family weighting and Holm-corrected tests of proposal VIII come with the full benchmark.

Outputs: `results/metrics/{run_summary,scenario_summary,fold_summary,per_class,confusion,skipped_runs}.tsv`.

## HVG anomaly report (decision 8)

Every run's `fitted/hvg_report.json` records, on the training cells the HVGs were chosen from:

- genes with no variability score (constant or all-zero in train), and a tie at the 2,000 cutoff;
- HVGs detected in fewer than 10 training cells;
- HVGs with ≥ 90% of their training expression in one donor;
- the rank of every watched gene, whether it was selected, and its mean-expression rank;
- in the run record: how many of the frozen HVGs are never detected in the query cells (flagged above 1%).

`results/hvg_anomalies.tsv` lists every warning with its run. They change nothing; they are there so a surprise (for example "INS is not an HVG") is written down when it happens.

## Results (pilot_v2, computed once on 2026-11-23, after the decisions were fixed)

Means over seeds (and over the 4 folds for within-Baron scenarios). Full tables: `results/metrics/`; explained in `ABI_Week2_mentor_notebook_v2.ipynb` §10–11.

| Scenario | Method | Accuracy | Macro-F1 | FRR | TRR | BRA | AUROC (unseen) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E1 within Baron | linear_svm | 0.973 | 0.829 | 0.064 | | | |
| | pca_knn.k10 | 0.968 | 0.773 | 0.065 | | | |
| | pca_knn.k5 | 0.969 | 0.810 | 0.053 | | | |
| E2 Baron → Muraro | linear_svm | 0.878 | 0.833 | 0.009 | | | |
| | pca_knn.k10 | 0.868 | 0.730 | 0.064 | | | |
| | pca_knn.k5 | 0.867 | 0.757 | 0.052 | | | |
| E3 within Baron (delta unseen) | linear_svm | 0.976 | 0.851 | 0.060 | 0.815 | 0.877 | 0.963 |
| | pca_knn.k10 | 0.969 | 0.756 | 0.057 | 0.488 | 0.715 | 0.751 |
| | pca_knn.k5 | 0.971 | 0.801 | 0.045 | 0.425 | 0.690 | 0.694 |
| E3 Baron → Muraro (delta unseen) | linear_svm | 0.866 | 0.807 | 0.010 | 0.264 | 0.627 | 0.964 |
| | pca_knn.k10 | 0.856 | 0.729 | 0.043 | 0.668 | 0.813 | 0.825 |
| | pca_knn.k5 | 0.856 | 0.769 | 0.028 | 0.517 | 0.744 | 0.746 |

What to remember:

- Macro-F1 in E1 averages up to 13 classes (5 of them rare); in E2 it averages the 9 Muraro classes. Do not compare them directly.
- The ~10-point accuracy drop to Muraro is about 10× the spread between Baron donors. Most of it is acinar → ductal (Muraro acinar recall 0.06–0.17 vs 0.96 within Baron).
  - The Muraro mentor notebook rules out depth, the normalization constant and UMI saturation as causes.
  - Muraro acinar cells put ~3% of their counts into digestive enzymes (Baron ~19%, Enge ~26%, Segerstolpe ~6%), and they detect ductal genes (SOX9 86% vs 12% in Baron).
  - In a one-seed demo on the same split, rank-based SingleR and scmap recognize 80% and 59% of them, while magnitude-based models (our baselines, CellTypist) get 6–17%: a method × shift interaction.
  - Technical and biological causes cannot be separated with one Muraro study.
- SVM: the ranking of unseen cells survives transfer (AUROC 0.96 in both settings), but the 5% threshold does not. The SVM is overconfident on Muraro, so FRR falls to 0.01 and TRR to 0.26 (H1 mechanism). kNN shows the reverse pattern because Muraro delta cells fall between beta and gamma; one removed class cannot support a general claim.
- **Open question for the team (raised by the Muraro notebook §5).** Most of the SVM's threshold failure comes from normalizing each dataset to its own median (decision 1). Normalizing Muraro with Baron's constant gives an E3-cross SVM BRA of 0.83 (thinned to Baron depth: 0.84) instead of 0.63, against 0.88 within Baron. Mentor recommendation: keep decision 1 as primary and pre-declare a fixed-constant sensitivity analysis. A switch would be a post-result amendment; the principled argument for one is that a fixed constant is not query-adaptive.
- Known label issue: 13 Muraro cells labelled ductal are macrophages by marker evidence (PTPRC, TYROBP, C1QB, CD68 detected; CFTR not). They are to be handled with model-independent marker rules in a future label-map version, not with model predictions.
- The temperature-scaled SVM is underconfident within Baron (ECE 0.106) and overconfident on Muraro (ECE 0.090, Brier 0.077 → 0.219). The ECE values are similar but point in opposite directions.
- Seeds vary much less than folds (SVM accuracy SD 0.003 vs 0.010).
- k = 5 vs k = 10: same accuracy; k = 5 gives +0.03–0.045 macro-F1 (rare classes), k = 10 gives +0.07 BRA across datasets. Both instances are kept; if one must go, the mentor recommendation is k = 10 (rejection is a primary hypothesis). The team decides.
- E1 folds human1 and human2 train on the same donors ({human3, human4}), so for a given seed they share one model; E1 has 4 test donors but 3 distinct training sets.
