# Shared input and output specification

**Status: draft for agreement with the team leader.** Open decisions are listed together in section 7. Roman-numeral references such as (IV-D) point to sections of the study proposal.

Every method, in Python or R, reads and writes these same files. Plain formats (TSV, JSON, MatrixMarket) are used at boundaries so both languages can read them without a bridge.

## 1. Preprocessing inputs and outputs

**Inputs**

| Input | Description |
| --- | --- |
| Dataset object | From `annotbench.loading.load_dataset`. `X` holds the unit named in `uns["annotbench"]["assay_unit"]`. All deposited layers are kept. |
| Split file | `splits/<version>/<split_id>.tsv.gz` (format in section 5). Written by the split step, which is anticipated but not built yet (section 8). |
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
split_id = exp-<e1..e6>__ref-<reference>__qry-<query>__task-<label_task>__pert-<perturbation>__seed-<NN>
run_id   = <split_id>__method-<method>
```

Examples:

```text
exp-e2__ref-baron__qry-muraro__task-pancreas_harmonized_v1__pert-none__seed-03
exp-e1__ref-baron__qry-baron.human3__task-pancreas_harmonized_v1__pert-none__seed-00
exp-e3__ref-baron__qry-muraro__task-pancreas_harmonized_v1__pert-remove.delta__seed-00__method-linear_svm
exp-e4__ref-baron__qry-muraro__task-pancreas_harmonized_v1__pert-rare.delta.n20__seed-04__method-pca_knn
```

Field values use lowercase letters, digits, `_` and `.`, so a double underscore always separates fields. The method is absent from `split_id` because every method receives identical splits (VI-B). All four examples show the planned format only. No split files exist yet (section 8).

## 4. Model output vocabulary

| Term | Allowed values |
| --- | --- |
| `method` | `singler`, `scmap`, `seurat_transfer`, `celltypist`, `linear_svm`, `pca_knn` |
| `output_kind` | `native`, `common_threshold`, `wrapped` (V, VI-C) |
| `predicted_label` | A `harmonized_label` from `labels/label_map.csv`, or empty when rejected |
| `predicted_cl_id` | A Cell Ontology ID. For `wrapped`, this may be an ancestor term. Empty when rejected. |
| `prediction_level` | `leaf`, `ancestor`, `rejected` |
| `score_type` | `probability`, `calibrated_probability`, `decision_score`, `vote_fraction`, `similarity`, `correlation`, `prediction_score` |

A model's internal class names are converted by its adapter. No method-specific label reaches a results file. Rejection is an empty label with `prediction_level = rejected`, so "unassigned" can never be mistaken for a cell type.

## 5. File formats

**Split file** (planned: the split step in section 8 will write it), one row per cell in both datasets. Key: `dataset` + `cell_id`.

| Column | Meaning |
| --- | --- |
| `dataset`, `cell_id` | Stable identifier from the source deposit |
| `donor` | Standardized donor |
| `label_original`, `label_harmonized` | From the data and from the label map |
| `role` | `train`, `train_reserve`, `calibration`, `evaluation`, `evaluation_open_set`, `challenge_unlabeled`, `excluded` |
| `role_reason` | Why this cell has this role |
| `inner_fold` | Donor used for donor-grouped tuning, on `train` and `train_reserve` cells |

**Prediction file**, `results/runs/<run_id>/predictions.tsv.gz`, one row per query cell and `output_kind`:

| Column | Type | Notes |
| --- | --- | --- |
| `run_id` | text | |
| `dataset`, `cell_id` | text | Must match the split file |
| `output_kind` | text | Section 4 |
| `predicted_label`, `predicted_cl_id` | text | Empty when rejected |
| `prediction_level` | text | Section 4 |
| `rejected` | true/false | |
| `top_score` | number | Empty if the method gives no score |
| `score_type` | text | Section 4 |

True labels are deliberately absent. Metric code joins them from the split file, so a method cannot read them from its own output folder.

**Per-class scores**, `results/runs/<run_id>/scores.tsv.gz`: `cell_id`, `score_type`, then one column per class in sorted vocabulary order. If a method has no defensible per-class score, the file is omitted and the run record says `missing_score_output` (V-A).

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

`status` is exactly one of `success`, `software_failure`, `ineligible_input`, `missing_score_output`, `timeout` (VII-D). The eligibility and failure matrix is the concatenation of all run records.

## 7. Decisions needed from the team leader

1. **Muraro expression unit.** The pilot proposes recovered UMIs (`umi`), which match Baron's unit. Their limit is saturation at 256 UMIs per gene and cell, reached by 1,408 of 12.4 million stored values. Corrected transcripts stay available in a layer.
2. **Muraro `mesenchymal` (80 cells).** It is excluded until reviewed. Accepting it as stellate adds a ninth shared class candidate.
3. **Validation role.** The proposal lists a distinct internal-validation role (IV-E). Baron's counts argue against giving it a whole donor: 843 of Baron's 958 acinar cells come from one donor (`human3`), and no choice of calibration and validation donors leaves calibration, validation and training with 20 or more acinar cells each. The suggested alternative is donor-grouped inner folds inside the training cells.
4. **Pilot thresholds for the split step.** Only the minimum of 20 calibration cells per class comes from the proposal (VII-C). Suggested values for the rest: at least 20 training and 10 evaluation cells per class, and a flag when one donor supplies 90% or more of a class's training cells.
5. **Whether each transfer direction may use its own class set.** Baron and Muraro share 8 eligible classes, but they don't support them equally. Muraro has 21 endothelial cells and no donor with 20, so endothelial can meet the 20-cell calibration minimum only when Baron is the reference. Epsilon has too few cells in both datasets (18 and 3).
6. **Cell Ontology release.** The IDs in the label map are marked unverified until a release is pinned.

## 8. The split step (anticipated, not built)

Week 1 stops before splits. The assignments depend on decisions 3 to 5 above, and they should be saved only once their rules are approved. This section records what the split step must satisfy, so whoever builds it starts from the proposal's requirements.

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
