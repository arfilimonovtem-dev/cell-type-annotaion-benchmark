# Environment for the Baron-Muraro pilot

These steps take a fresh clone to a complete Week 1 run. All commands run from the repository root.

## 1. Python

```bash
conda env create -f environment.yml
conda activate annotbench-pilot
```

`environment.yml` pins the versions the pilot was tested with. Week 1 itself uses anndata, numpy, pandas, scipy, h5py, PyYAML and GEOparse. scanpy and scikit-learn are pinned now because the SVM and PCA-kNN baselines need them next.

## 2. R (only to rebuild Baron from its .rds file)

R 4.3.3 with Bioconductor 3.18:

```r
install.packages("BiocManager")
BiocManager::install(version = "3.18")
BiocManager::install(c("SingleCellExperiment", "SummarizedExperiment"))
```

Tested versions are SingleCellExperiment 1.24.0, SummarizedExperiment 1.32.0 and Matrix 1.6.5. R is not needed if you already have `interim/baron_human.h5ad` and `interim/baron_human/obs.csv`.

## 3. Data

Raw and derived data are not in git. Put them in any folder with this layout:

```text
<data root>/
  raw/baron_GSE84133/baron-human.rds
  raw/muraro_GSE85241/GSE85241_cellsystems_dataset_4donors_updated.csv.gz
  raw/muraro_GSE85241/GSE85241_cel-seq_barcodes.csv.gz
  raw/muraro_GSE85241/GSE85241_family.soft.gz
  raw/muraro_scRNAseq_EH2693/coldata.csv
  raw/ncbi_gene_info/Homo_sapiens.gene_info.gz
  interim/baron_human.h5ad             (built in step 4)
  interim/baron_human/obs.csv          (built in step 4)
  interim/muraro_human.h5ad            (built in step 4)
```

Where each raw file comes from:

| File | Source | Note |
| --- | --- | --- |
| `baron-human.rds` | Hemberg-lab scRNA-seq collection | The original download URL was not recorded. Get it from a teammate who has it. |
| `GSE85241_*` | GEO accession GSE85241 | Supplementary files and the family SOFT file. |
| `coldata.csv` | Bioconductor ExperimentHub EH2693 | See `README.txt` in that folder for the exact fetch and export commands. |
| `Homo_sapiens.gene_info.gz` | NCBI gene_info | NCBI updates this file, so a new download will not match the recorded checksum. Copy the pinned file from a teammate. |

Tell the code where the data is, using either option. With neither, it uses `<repo>/data`.

```bash
export ANNOTBENCH_DATA_ROOT=/path/to/data      # option 1: once per shell
python -m annotbench --data-root /path/to/data week1   # option 2: per command
```

## 4. Build the expression files

Skip this if the three `interim/` paths above already exist.

```bash
DATA=${ANNOTBENCH_DATA_ROOT:-data}
Rscript scripts/export_sce.R "$DATA/raw/baron_GSE84133/baron-human.rds" "$DATA/interim/baron_human"
python scripts/build_h5ad.py "$DATA/interim/baron_human" "$DATA/interim/baron_human.h5ad"
python scripts/build_muraro.py --data-root "$DATA"
```

`scripts/build_muraro.py` will not overwrite an existing file unless you pass `--force`.

## 5. Run Week 1

```bash
python -m annotbench check-env      # tools and versions
python -m annotbench check-inputs   # files against their checksums
python -m annotbench integrity      # dataset and integrity reports
python -m annotbench week1          # all of the above, in order
python -m unittest discover -s tests -t .
```

A run that fails a fatal check exits with code 2 and prints what failed, with the path of the full report. The whole `week1` command took about 30 seconds on the development machine and peaked at about 2.1 GB of memory.

## 6. Not needed yet

| Tool | Needed for | State on the development machine |
| --- | --- | --- |
| Seurat, SeuratObject | Seurat transfer, one of the six methods | Not installed |
| SingleR, scmap | Later method adapters | Installed, 2.4.1 and 1.24.0 |
| CellTypist | Later method adapter | Installed, 1.7.1 |
| Nextflow and Java | The final workflow | Not installed |
| Docker | Per-method containers | Installed |
| Cell Ontology release | Back-off wrapper and HOE | No release chosen or downloaded |

## 7. What has been verified

- The pipeline runs on the development machine with the pinned versions.
- Running from a different working directory with the data in a different folder produced byte-identical reports.
- Rebuilding both datasets from raw files gave the same matrices, layers, cell metadata and gene metadata as the existing objects.

**Not verified:** creating the environment from `environment.yml` on a clean machine. Please report anything that fails the first time a teammate tries it.

On the development machine pandas warns that numexpr 2.10.1 is older than it would like. The warning is harmless for this code. That machine's `dataenv` also holds a leftover `scipy-1.16.1` metadata folder, while the installed scipy is 1.16.0.
