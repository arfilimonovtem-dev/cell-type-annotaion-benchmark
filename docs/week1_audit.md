# Week 1 audit of the existing implementation

Scope: `scripts/export_sce.R`, `scripts/build_h5ad.py`, `data_preview.ipynb`, `data_preview_dataset_B_and_C.ipynb`, `preprocessing.ipynb`. Section numbers refer to the study proposal.

## 1. Reusable code

| Code | Reused for | Status |
| --- | --- | --- |
| `scripts/export_sce.R` | Baron export from `.rds`. The same plain-file format can later carry data back to R methods. | Used unchanged |
| `scripts/build_h5ad.py` lines 37-42 | Reading exports and fixing orientation | Used unchanged |
| `integrity_checks` (B_and_C cell 3) | Checks for NaN, negatives, non-integers and duplicates | Extended in `annotbench/integrity.py` |
| `unique_map` and `resolve` (`preprocessing.ipynb` cell 12) | Gene mapping that rejects ambiguous names | Moved to `annotbench/genes.py` |
| `build_muraro` (B_and_C cell 45) | Label join and UMI recovery | Moved to `scripts/build_muraro.py` |
| `soft_samples` (B_and_C cell 3) | Reading GEO sample metadata | Reduced to what the Muraro build needs |

Not reusable for models: `throwaway_pca` fits on all cells (IV-E), and the marker-based provisional Muraro labels are ruled out by IV-A.

## 2. Exploratory code that needed refactoring

| Problem | Where | Resolution |
| --- | --- | --- |
| Notebook globals and an absolute project path | `build_xin`, `build_muraro`, `PROJECT` | Muraro done. Xin and later datasets are still to do. |
| "counts" does not name a unit (UMIs or reads) | `build_h5ad.py` line 46 | Each layer's unit is declared in `configs/datasets.yaml` |
| Unrestricted `.` to `-` gene renaming (IV-D) | `preprocessing.ipynb` cell 12 and plan step 1 | `repair_r_names` accepts a repair only if NCBI knows the name |
| Silent NaN from four copies of `LINEAGE` | Notebooks | `labels/label_map.csv`, and an unlisted label now stops the run |
| One split stored inside `obs` | Plan step 3 | Planned as one split file per scenario (`docs/io_spec.md` sections 5 and 8). Not built in Week 1. |
| Mitochondrial count relied on the mapping rule, not the gene | `preprocessing.ipynb` cell 14 | `is_mt` column based on gene identity |
| The plan says Muraro is excluded | `preprocessing.ipynb` cell 1 | Superseded by the proposal. The notebook text still needs updating. |

## 3. Steps not yet implemented

- The split step. Its requirements, and the decisions it waits on, are in `docs/io_spec.md` sections 7 and 8.
- A policy applied to duplicate, ambiguous and unmatched genes. They are flagged but not yet acted on.
- Assay-aware QC rules with retention by dataset, donor and class, plus the common-filter sensitivity analysis.
- The per-family gene universe file.
- Normalization, HVG selection, scaling and PCA fitted on training cells and saved.
- Export from Python to the R methods.
- Method adapters and the common output files.
- Metric code: HOE, TRR/FRR, ECE, Brier and conformal, with toy-ontology tests.

## 4. Required dependencies

See `docs/environment.md`. For the Baron-Muraro pilot, Python packages are present. Seurat, Nextflow and Java are absent, and none of them is needed until later milestones.

## 5. Missing input files

No input needed for the pilot is missing. `manifests/pilot_inputs.tsv` records the size and checksum of each one.

| Item | State |
| --- | --- |
| Source URL for `baron-human.rds` | Not recorded. The authors' original GEO files were not found in the project or in `Downloads/proj_data`. |
| Cell Ontology release | Not chosen or downloaded |
| `GSE84133_family.soft` and the Zheng tarballs | Outside the project folder. Not needed for this pilot. |

## 6. Environment

See `docs/environment.md`.
