# Dataset report: muraro

- Input: `interim/muraro_human.h5ad` (82,889,613 bytes)
- SHA-256: `52be31051ebbf72543205cf51b6863456f49537e27b80bb3cf875bb4587f348a`
- Cells: 3,072. Features: 19,140.
- Example cell IDs: D28-1_1, D28-1_2, D28-1_3
- Example feature IDs: A1BG-AS1__chr19, A1BG__chr19, A1CF__chr10

## Matrices

| matrix | unit | shape (cells x features) | stored values | fraction stored | min | max | all integer |
| --- | --- | --- | --- | --- | --- | --- | --- |
| X (copy of umi) | umi_counts | 3072 x 19140 | 12442034 | 0.2116 | 1.0 | 256.0 | True |
| layer transcripts | corrected_transcripts | 3072 x 19140 | 12442034 | 0.2116 | 1.0019582509994507 | 1597.0111083984375 | False |
| layer umi | umi_counts | 3072 x 19140 | 12442034 | 0.2116 | 1.0 | 256.0 | True |

## Cells per donor

| donor | cells |
| --- | --- |
| D28 | 768 |
| D29 | 768 |
| D30 | 768 |
| D31 | 768 |

## How feature identifiers resolved against NCBI gene_info

| resolution | features |
| --- | --- |
| ambiguous_name | 59 |
| official_symbol | 16672 |
| spike_in | 81 |
| unique_synonym | 1606 |
| unmatched | 722 |

## Cell metadata columns (obs)

| name | type | missing | distinct |
| --- | --- | --- | --- |
| library | category | 0 | 32 |
| donor | string | 0 | 4 |
| well | int64 | 0 | 96 |
| gsm | category | 0 | 32 |
| barcode | category | 0 | 96 |
| cell_type1 | category | 946 | 10 |
| author_qc_pass | bool | 0 | 2 |
| dataset | str | 0 | 1 |
| label_original | string | 946 | 10 |

## Feature metadata columns (var)

| name | type | missing | distinct |
| --- | --- | --- | --- |
| symbol | string | 0 | 19127 |
| chromosome | category | 0 | 25 |
| is_ercc | bool | 0 | 2 |
| feature_id_original | str | 0 | 19140 |
| symbol_original | string | 0 | 19127 |
| symbol_repaired | bool | 0 | 1 |
| ncbi_gene_id | string | 862 | 18228 |
| id_resolution | string | 0 | 5 |
| is_spike_in | bool | 0 | 2 |
| is_mt | bool | 0 | 1 |
| shares_gene_id | bool | 0 | 2 |
