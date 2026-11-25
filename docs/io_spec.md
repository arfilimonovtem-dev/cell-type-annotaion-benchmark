# Shared input and output specification

**Status: agreed for the Week 2 pilot (pilot_v2, 2026-11-23).** Sections 3, 5 and 6 describe what the code writes now; section 7 records how the Week 1 open questions were settled (details in `docs/week2_decisions.md`). Roman-numeral references such as (IV-D) point to sections of the study proposal.

Every method, in Python or R, reads and writes these same files. Plain formats (TSV, JSON, MatrixMarket) are used at boundaries so both languages can read them without a bridge.

## 1. Preprocessing inputs and outputs

**Inputs**

| Input | Description |
| --- | --- |
| Dataset object | From `annotbench.loading.load_dataset`. `X` holds the unit named in `uns["annotbench"]["assay_unit"]`. All deposited layers are kept. |
| Split file | `splits/<version>/<split_id>.tsv.gz`: reference-dataset cells only (format in section 5). |
| Query set | `query_sets/<version>/<scenario_id>.tsv.gz`: query-dataset cells of a cross-dataset scenario, independent of the seed (section 5). |
| Gene universe | `gene_universes/<family>_v<N>.tsv`, listing `ncbi_gene_id`. Built from feature metadata only, never from query expression (IV-D). |
| QC rules | Versioned per assay (IV-D). |
| Method requirement | The units a method accepts, declared by its adapter (Table 1). |

**Outputs**, in `results/preprocessed/<split_id>/<recipe>/`:

| File | Content |
| --- | --- |
| `reference_train.h5ad`, `calibration.h5ad`, `query.h5ad` | One object per role, with genes in one fixed order. Original cell IDs are unchanged. |
| `r_export/<role>/matrix.mtx.gz`, `obs.tsv.gz`, `var.tsv.gz` | The same data for R methods. |
| `qc_retention.tsv` | Cells kept and removed by dataset, donor and class (IV-D). |
| `fitted/` | Saved transformations (section 2). |
| `provenance.json` | Described in section 2. |

**Rules:** a matrix in an incompatible unit is never passed to a method (IV-D). The adapter records `ineligible_input` instead.

## 2. Saved fitted transformations

Anything learned from data is fitted once, saved, and only applied afterwards.

| Step | Fitted on | Saved as | Applied to |
| --- | --- | --- | --- |
| Library-size normalization and log1p | Nothing, computed per cell | Recipe parameters | All roles |
| Highly variable genes | `train` | `hvg.tsv` (`ncbi_gene_id`, rank) | All roles |
| Scaling | `train` | `scaler.tsv` (gene, mean, sd) | All roles |
| PCA | `train` | `pca_loadings.tsv.gz`, `pca_center.tsv` | All roles |
| Classifier or neighbor index | `train` | Method's native file | Calibration and query |
| Score transformation | Inner folds of `train` | `score_transform.json` | Calibration and query |
| Rejection thresholds | `calibration`, after the score rule is frozen | `thresholds.json` | Query |
| Conformal quantiles | `calibration`, after the score rule is frozen | `conformal.json` | Query |

Each `fitted/` folder has one `provenance.json`:

```json
{
  "split_id": "exp-e2__ref-baron__qry-muraro__task-pancreas_harmonized_v1__pert-none__seed-00",
  "recipe": "lognorm_v1",
  "roles_used": ["train"],
  "n_cells": 2694,
  "cell_ids_sha256": "hash of the sorted cell IDs used",
  "git_commit": "abc1234",
  "package_versions": {"scanpy": "1.12.4"},
  "created_utc": "2026-11-10T12:00:00Z"
}
```

`cell_ids_sha256` lets a test prove that a transformation saw only training cells (IV-E).

## 3. Scenario identifiers

A scenario is indexed by reference, query, label task, perturbation, method and seed (VI-F). Fields are joined by double underscores.

```text
scenario_id = exp-<e1..e6>__ref-<reference>__qry-<query>__task-<label_task>__pert-<perturbation>
split_id    = <scenario_id>__seed-<NN>
run_id      = <split_id>__method-<method instance>
```

A **method instance** is a method family with fixed settings, named in the config (`pca_knn.k5`, `pca_knn.k10`). A family with one instance uses the family name (`linear_svm`). The run record stores both `method` (instance) and `method_family`.

Examples:

```text
exp-e2__ref-baron__qry-muraro__task-pancreas_harmonized_v2__pert-none__seed-03
exp-e1__ref-baron__qry-baron.human3__task-pancreas_harmonized_v2__pert-none__seed-00
exp-e3__ref-baron__qry-muraro__task-pancreas_harmonized_v2__pert-remove.delta__seed-00__method-linear_svm
exp-e3__ref-baron__qry-baron.human2__task-pancreas_harmonized_v2__pert-remove.delta__seed-07__method-pca_knn.k10
exp-e4__ref-baron__qry-muraro__task-pancreas_harmonized_v2__pert-rare.delta.n20__seed-04__method-pca_knn.k5   (planned)
```

Field values use lowercase letters, digits, `_` and `.`, so a double underscore always separates fields. The method is absent from `split_id` because every method receives identical splits (VI-B). Within-dataset query tokens name the held-out donor (`baron.human3`).

## 4. Model output vocabulary

| Term | Allowed values |
| --- | --- |
| `method` (family) | `singler`, `scmap`, `seurat_transfer`, `celltypist`, `linear_svm`, `pca_knn` |
| method instance | `<family>` or `<family>.<tag>`, e.g. `pca_knn.k10` (section 3) |
| `output_kind` | `native`, `common_threshold`, `wrapped` (V, VI-C) |
| `predicted_label` | A `harmonized_label` from `labels/label_map.csv`, or empty when rejected |
| `predicted_cl_id` | A Cell Ontology ID. For `wrapped`, this may be an ancestor term. Empty when rejected. |
| `prediction_level` | `leaf`, `ancestor`, `rejected` |
| `score_type` | `probability`, `calibrated_probability`, `decision_score`, `vote_fraction`, `similarity`, `correlation`, `prediction_score` |

A model's internal class names are converted by its adapter. No method-specific label reaches a results file. Rejection is an empty label with `prediction_level = rejected`, so "unassigned" can never be mistaken for a cell type.

## 5. File formats

**Split file**, one row per cell of the **reference dataset only**. Key: `dataset` + `cell_id`. The query dataset of a cross-dataset scenario never appears here, so no code path can train on it.

| Column | Meaning |
| --- | --- |
| `dataset`, `cell_id` | Stable identifier from the source deposit |
| `donor` | Standardized donor |
| `label_original`, `label_harmonized` | From the data and from the label map |
| `role` | `train`, `train_reserve`, `calibration`, `evaluation`, `evaluation_open_set`, `challenge_unlabeled`, `excluded` |
| `role_reason` | Why this cell has this role |
| `inner_fold` | Donor used for donor-grouped tuning, on `train` and `train_reserve` cells |

In within-dataset scenarios (E1, E3-within) the held-out donor's cells are `evaluation` or `evaluation_open_set` in the split file. Next to each split: `<split_id>.counts.tsv` (cells by role and class) and `<split_id>.restrictions.tsv` (support checks, section 8).

**Query set**, `query_sets/<version>/<scenario_id>.tsv.gz`, one row per cell of the query dataset of a cross-dataset scenario. Same columns as the split file except `inner_fold`. Roles: `evaluation` (a class the reference trained on), `evaluation_open_set` (`class_removed:<x>` or `class_not_in_training`), `challenge_unlabeled`, `excluded`. It has no seed because the seed only touches reference cells; every seed of a scenario is scored against the same query set.
**Prediction file**, `results/runs/<run_id>/predictions.tsv.gz`, one row per calibration or query cell and `output_kind`:

| Column | Type | Notes |
| --- | --- | --- |
| `run_id` | text | |
| `dataset`, `cell_id` | text | Must match the split file or query set |
| `cell_set` | text | `calibration` (reference cells used only for thresholds, VII.A) or `query` (scored cells) |
| `output_kind` | text | Section 4 |
| `predicted_label`, `predicted_cl_id` | text | Empty when rejected |
| `prediction_level` | text | Section 4 |
| `rejected` | true/false | |
| `top_score` | number | Empty if the method gives no score |
| `score_type` | text | Section 4 |

True labels are deliberately absent. Metric code (`annotbench/metrics.py`) joins them from the split file or the query set, so a method cannot read them from its own output folder. `cell_set` says only which cells were scored for what purpose; whether a query cell is seen or unseen is a label fact and stays in the split file or query set.

**Per-class scores**, `results/runs/<run_id>/scores.tsv.gz`: `dataset`, `cell_id`, `cell_set`, `score_type`, then one column per class in sorted vocabulary order. If a method has no defensible per-class score, the file is omitted and the run record says `missing_score_output` (V-A).

## 6. Runtime and failure records

Every run writes `results/runs/<run_id>/run_record.json`, even when it fails. Failed runs are never dropped or given invented accuracy values (VII-D).

```json
{
  "run_id": "exp-e2__ref-baron__qry-muraro__task-pancreas_harmonized_v1__pert-none__seed-00__method-linear_svm",
  "status": "success",
  "stages": {
    "preprocessing": {"wall_seconds": 12.4, "peak_memory_mb": 1830},
    "fit": {"wall_seconds": 41.0, "peak_memory_mb": 2210},
    "annotate": {"wall_seconds": 3.1, "peak_memory_mb": 1420}
  },
  "error_type": null,
  "error_message": null,
  "input_unit": "umi_counts",
  "git_commit": "abc1234",
  "environment": "annotbench-pilot",
  "host": "machine-label",
  "started_utc": "2026-11-10T12:00:00Z"
}
```

`status` is exactly one of `success`, `software_failure`, `ineligible_input`, `missing_score_output`, `timeout` (VII-D). The eligibility and failure matrix is the concatenation of all run records (`results/run_index.tsv`).

The pilot's records also carry: `method_family`; `hashes` (config, split file, query set file, universe, HVG list, train/calibration/query cell IDs); `target_sums`; `query_adaptive_steps` (the per-dataset median library size is one); `method_settings`; `preprocess_settings`; `vocabulary`; `hvg_anomalies` and `hvg_query_coverage`; `threads_per_run`; `package_versions`; `git_commit`, `git_dirty`; `warnings`.

## 7. Week 1 open questions and how they were settled

Settled on 2026-11-23 (`docs/week2_decisions.md` has the reasons):

1. Muraro unit: recovered UMIs, saturation at 256 recorded as a known limit (decision 22).
2. Muraro `mesenchymal`: stellate, after a marker review (`labels/label_map_v2.csv`, decision 23).
3. Validation role: no separate validation donor; donor-grouped inner folds inside the training donors.
4. Split thresholds: 20 calibration (VII-C), 20 train, 10 evaluation, donor-dominance flag at 90%, `support_policy: flag` (decision 17).
5. Class set: the reference trains on all its classes (`train_classes: all`, decision 18). Only Baron is a reference in the pilot.
6. Cell Ontology release: still open. IDs stay marked unverified until a release is pinned.

The original questions, as asked in Week 1:

1. **Muraro expression unit.** The pilot proposes recovered UMIs (`umi`), which match Baron's unit. Their limit is saturation at 256 UMIs per gene and cell, reached by 1,408 of 12.4 million stored values. Corrected transcripts stay available in a layer.
2. **Muraro `mesenchymal` (80 cells).** It is excluded until reviewed. Accepting it as stellate adds a ninth shared class candidate.
3. **Validation role.** The proposal lists a distinct internal-validation role (IV-E). Baron's counts argue against giving it a whole donor: 843 of Baron's 958 acinar cells come from one donor (`human3`), and no choice of calibration and validation donors leaves calibration, validation and training with 20 or more acinar cells each. The suggested alternative is donor-grouped inner folds inside the training cells.
4. **Pilot thresholds for the split step.** Only the minimum of 20 calibration cells per class comes from the proposal (VII-C). Suggested values for the rest: at least 20 training and 10 evaluation cells per class, and a flag when one donor supplies 90% or more of a class's training cells.
5. **Whether each transfer direction may use its own class set.** Baron and Muraro share 8 eligible classes, but they don't support them equally. Muraro has 21 endothelial cells and no donor with 20, so endothelial can meet the 20-cell calibration minimum only when Baron is the reference. Epsilon has too few cells in both datasets (18 and 3).
6. **Cell Ontology release.** The IDs in the label map are marked unverified until a release is pinned.

## 8. The split step (built in Week 2: `annotbench/splits.py`)

These are the requirements the split step was built against. `validate_split` and `validate_query_set` re-check them on every file they write. One design change from the original plan: the query dataset of a cross-dataset scenario now lives in its own query set file, not in the split file.

**Inputs that Week 1 already provides.** After `annotbench.labels.attach_labels`, every cell has `donor`, `label_original`, `label_harmonized`, `eligible_primary` and `exclusion_reason`. The integrity checks confirm that the donor column agrees with the cell IDs. The split step needs nothing else.

| Requirement (proposal section) | What the split step must do |
| --- | --- |
| Query evaluation cells are never used to tune parameters (IV-E) | Decide roles from donors, labels and fixed seeds only, and never read expression values |
| Within-dataset evaluation uses donor-grouped splits where feasible, and inner tuning respects donor groups (IV-E, VI-A) | E1 holds out whole donors, and tuning folds are grouped by donor too |
| An independent calibration subset of the remaining reference (VI-A) | E1 calibration cells come from the development donors and are never used for fitting. The proposal gives no size for this subset |
| Rejection thresholds and conformal quantiles use independent calibration cells (IV-E) | A `calibration` role that is never used for fitting. In E2, no donor supplies both training and calibration cells |
| Class-conditional calibration needs at least 20 calibration cells per class (VII-C) | Check every class, and flag those below 20 |
| Transfer is evaluated in both directions, and unmatched query populations go to a separate open-set analysis (VI-B) | E2 pairs Baron → Muraro and Muraro → Baron. Query classes absent from training become `evaluation_open_set` |
| All eligible methods receive identical scenario definitions (VI-B) | One split file per scenario, shared by every method (section 3) |
| References are capped at 500 training cells per eligible class and sampled over 10 fixed seeds (VI-B) | Apply the cap and the seed list. Cells above the cap stay on record as `train_reserve` |
| Uncertain labels and suspected doublets are excluded and may form an unlabeled challenge set (IV-A) | Those cells become `challenge_unlabeled`, never `evaluation` |
| Missing training classes and inadequate donor or calibration support trigger documented restrictions, not silent resampling (IV-E) | Report every restriction, and never redraw a split to make it look better |
| A manifest records files and checksums (IV-D) | List every split file with its checksum |

The counts behind decisions 3 to 5 are eligible cells by dataset, donor and harmonized class, computed from the output of `attach_labels`.
