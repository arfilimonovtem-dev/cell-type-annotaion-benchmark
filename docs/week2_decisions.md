# Week 2 pilot: open decisions

Status: **draft, nothing here is approved yet.** The Week 2 draft pipeline runs end to end on Baron → Muraro (`python -m annotbench run --config configs/pilot.yaml`), but every value below is a placeholder in `configs/pilot.yaml` until the team leader (or you) confirms it. No number in `annotbench/` is hard-coded; changing a decision means changing one config line and re-running.

Evidence comes from the Week 2 mentor notebook (`ABI_Week2_draft_mentor_notebook.ipynb`, section numbers in brackets) and from the pilot outputs in `results/`, `splits/pilot_v1/`, `gene_universes/` and `reports/week2/`.

"Who" says who is expected to decide: **leader** for scientific or project-level choices, **you** for engineering choices you can defend yourself, **decided** when the leader already gave the value.

## Summary table

| # | Decision | Draft placeholder (config key) | Evidence [notebook §] | Who |
| --- | --- | --- | --- | --- |
| A | When to compute exam metrics (E1 human4, E2/E3 Muraro) | **decided 2026-11-22: only after decisions 1–27 are fixed**, as a one-time look | looking earlier invites tuning settings on the test set; only calibration-donor sanity metrics exist so far [§9] | decided (Artem) |
| 1 | Upper UMI bound for Muraro | 100,000 for both (`qc.default.max_total_counts`) | 6 Muraro cells exceed it, 5 of them labelled and eligible (3 beta, 1 alpha, 1 acinar; 104k–210k UMIs, 10–12k genes: likely doublets). Muraro is 2.6× deeper than Baron [§3] | leader |
| 2 | Ribosomal % filter | none (`max_pct_ribo: null`) | a 20% cut would remove 57% of T cells, 56% of mast, 49% of acinar, 19% of ductal [§3] | leader |
| 3 | Hb and genes-detected filters | none (`max_pct_hb`, `min/max_genes_detected: null`) | Hb 99th percentile is 0 in both datasets; naive `^HB` would catch HBEGF, HBP1, HBS1L, HBBP1 [§3] | leader |
| 4 | Mitochondrial filter | not applicable (`max_pct_mt: null`) | 0 MT features in either matrix; reported `not_applicable` [§3] | rule |
| 5 | Fixed vs MAD-based QC thresholds | fixed | options table [§3] | leader |
| 6 | Normalization constant | `preprocess.target_sum: 10000` | per-dataset library-size normalization is already what lognorm does; `dataset_median` (Baron 5,016, Muraro 13,230) adds a 0.599 log shift between datasets and doubles the Muraro → Baron nearest-neighbour distance ratio (1.33 → 2.86) [§6.1] | **leader (asked for per-dataset)** |
| 7 | Genes counted in library size | `library_size: universe` | a median 95% of each Muraro cell's UMIs (1st percentile 83%) fall on universe genes [§6.2] | you |
| 8 | HVG flavor, donor batch key | `hvg_flavor: seurat`, `hvg_batch_key: null` | `seurat` drops INS, SST and PPY from the 2,000 (GCG ranks 1,508); `seurat_v3` keeps PPY and scores best in inner CV [§6.3, §7] | you / leader |
| 9 | Clip after scaling | `clip_max: 10` | scanpy convention [§6.4] | you |
| 10 | Number of HVGs and PCs | 2,000 and 50 | leader's values; 50 PCs keep 35.8% of variance, later PCs add ~1% each [§6.5] | decided |
| 11 | k for kNN | `methods.pca_knn.k: 15`; option `k_grid` | inner CV macro-F1: k=5 0.736, k=10 0.675, k=15 0.639, k=30 0.594, k=100 0.510; accuracy flat to k=15 [§7] | **leader asked** |
| 12 | kNN vote weights | `weights: uniform` | distance weights give the same macro-F1 with continuous scores; at k=15, 92% of calibration cells are unanimous (11 distinct score values) [§7] | you |
| 13 | Calibration donor(s) | `calibration_donors: [human1]` | human1 gives ≥ 20 calibration cells to 7/8 shared classes and keeps human3's 843 acinar in training; human3 also reaches 7/8 but leaves 115 acinar [§4] | **leader (analysing Baron)** |
| 14 | E1 exam donor(s); full leave-one-donor-out? | `eval_donors: [human4]` | human3 supplies 99.6% of acinar training cells in E1 [§4] | leader |
| 15 | Class removed in E3 | `remove: [delta]` | 601 Baron cells excluded, 193 Muraro cells become `evaluation_open_set` [§4] | leader |
| 16 | Seeds | `seeds: [0, 1]` | proposal VI.B asks for 10; seed changes move 2,344 cells, only between train and train_reserve [§4] | leader |
| 17 | Support thresholds and policy | 20 train, 10 eval, `support_policy: flag` | restriction reports in `splits/pilot_v1/*.restrictions.tsv` [§4] | leader |
| 18 | Reference classes absent from the query | `train_classes: all` | options [§4] | leader |
| 19 | Features sharing one NCBI Gene ID | `shared_id_policy: drop` (43 Baron, 98 Muraro features) | `sum` not implemented [§5] | leader |
| 20 | Pairwise vs tissue-family universe | pairwise, `status: provisional` (16,847 genes) | Segerstolpe and Xin not yet in the registry [§5] | leader |
| 21 | Muraro expression unit | `umi` (registry) | 256-UMI saturation; leader is checking [io_spec §7.1] | **leader (in progress)** |
| 22 | Muraro `mesenchymal` (80 cells) | excluded | io_spec §7.2 | leader |
| 23 | Predictions for calibration cells, `cell_set` column | written | needed for the VII.A rejection threshold [§9] | leader |
| 24 | SVM representation, C, probability calibration, code | placeholder `LinearSVC`, `scaled_hvg`, Platt on donor-grouped inner folds, `dual: true` | [§8] | **leader (Friday)** |
| 25 | Strict inner CV (refit preprocessing per inner fold) | `inner_cv.refit_preprocessing: true` | slower, but no information crosses inner folds [§8] | you |
| 26 | Timeouts | not implemented | needs one process per run (Nextflow stage) | later |
| 27 | Commit `results/`, `splits/`, `gene_universes/` to git? | not decided (all untracked) | splits and universes are small versioned definitions (proposal IX); predictions are derived and regenerable | you / leader |

## Details for the questions the leader asked

### QC (decisions 1–5)

The leader's rules are implemented in `annotbench/qc.py` and set in `configs/pilot.yaml` under `qc`. QC is **stateless per cell**: each cell passes or fails on its own numbers, so it can run before splits without leaking anything between cells.

- **200–100,000 UMIs.** Every Baron cell passes (1,201–34,509 UMIs). In Muraro, 268 cells are below 200, and all of them are already `author_qc_fail`. The upper bound removes 6 Muraro cells, 5 of them labelled. Their gene counts (10–12k) suggest doublets, so the bound may be doing the right thing. Still, it is a Muraro-specific effect of a shared threshold, and the leader should see it.
- **Mitochondrial.** Neither deposit contains MT genes. The rule reports `not_applicable` instead of silently passing.
- **Ribosomal.** RP fraction depends on cell type: acinar, mast and T cells are naturally high. A single cut-off removes biology, not damaged cells. The draft leaves the rule off. Turning it on is one config line: `max_pct_ribo`.
- **Hemoglobin.** Effectively zero in both datasets, so a threshold would change nothing. The draft uses the stricter `^HB[ABDEGMQZ]\d*$` pattern, because plain `^HB` would count HBEGF and HBP1.
- Outputs: `reports/week2/qc/` (per-cell metrics, verdicts, rule status, retention per dataset/donor/label).

### Normalization (decision 6)

"Per-dataset library size normalization" can mean two things:

- **Reading A (implemented, default):** every cell is scaled by its own library size to a common target (10,000). Each dataset is normalized on its own, and nothing is pooled across datasets.
- **Reading B (`target_sum: dataset_median`):** the target is the median library size of each dataset's QC-passing cells. This is implemented and switchable. The median is computed per dataset independent of roles, so it is identical across E1/E2/E3. For a cell type with the same expression proportions, it shifts log1p values between Baron and Muraro by ≈ 0.6. In the draft's checks this made Muraro look *further* from Baron (|mean z| 0.222 → 0.360; nearest-neighbour distance ratio 1.33 → 2.86).

**Suggested reply:** Reading A unless the leader specifically meant B. Both are recorded in `normalization_constants.json` and in each run record (`target_sums`).

### k for kNN (decisions 11–12)

Evidence from leave-one-donor-out CV on the E2 training donors (human2–4), never touching calibration or Muraro cells:

| k | 1 | 3 | 5 | 10 | 15 | 20 | 30 | 50 | 100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| macro-F1 | 0.668 | 0.660 | 0.736 | 0.675 | 0.639 | 0.617 | 0.594 | 0.564 | 0.510 |

- Macro-F1 falls for larger k, because rare classes (T cell and epsilon: 5 training cells each, Schwann: 8) get outvoted. Accuracy stays flat (0.859 → 0.850 at k=15).
- Uniform vote fractions are coarse. The VII.A threshold (5% false rejection) can only sit on one of ~k+1 steps.
- **Suggested reply:** k ≈ 10 with `weights: distance`, or set `k_grid: [5, 10, 15, 20, 30]` so the adapter chooses k by this donor-grouped CV inside each reference (`select_k`; ties go to the smaller k; saved in `fitted/k_tuning.tsv`). Either way, k comes from reference data only.
- The result depends on the HVG flavor (decision 8): with `seurat_v3`, macro-F1 is 0.751 / 0.732 / 0.679 / 0.620 at k = 5 / 10 / 15 / 30.

### SVM (decision 24)

`annotbench/methods/linear_svm.py` is an **adapter plus a placeholder model**. The adapter part (input checks, label vocabulary, score normalization, saving, run records) is shared infrastructure. The two functions marked `LEADER` are what the leader's own SVM replaces: `_new_classifier` and `_fit_platt`.

- `dual: true` matters: liblinear's primal solver took over 5 minutes per fit on the 2,842 × 2,000 E1 block, while the dual solver took ~20 s.
- Probability calibration is fit on donor-grouped out-of-fold decision scores from training cells only, never on calibration-role cells. Those stay reserved for the VII.A threshold.

## Additions to `docs/io_spec.md` pending approval

The draft writes two things the Week 1 spec does not list yet:

1. A `cell_set` column in `predictions.tsv.gz` and `scores.tsv.gz`, with values `calibration` or `query`. Calibration cells get predictions too, because VII.A sets the rejection threshold on them, and the column keeps them out of evaluation metrics. It deliberately does not say `evaluation` vs `evaluation_open_set`: that distinction comes from labels, so it stays in the split file.
2. `scores.tsv.gz` also carries `dataset`, and `run_record.json` has extra provenance fields: `hashes` (config, split file, universe, HVG list, train/calibration/query cell IDs), `target_sums`, `query_adaptive_steps`, `method_settings`, `preprocess_settings`, `package_versions`, `git_dirty`, `warnings`.

If approved, io_spec §4–6 should be updated to match.
