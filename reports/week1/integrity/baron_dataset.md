# Dataset report: baron

- Input: `interim/baron_human.h5ad` (113,920,064 bytes)
- SHA-256: `289c61f1fcc467404118d7a11a452cf6299af39f6c504d3272303b3c105f5159`
- Cells: 8,569. Features: 20,125.
- Example cell IDs: human1_lib1.final_cell_0001, human1_lib1.final_cell_0002, human1_lib1.final_cell_0003
- Example feature IDs: A1BG, A1CF, A2M

## Matrices

| matrix | unit | shape (cells x features) | stored values | fraction stored | min | max | all integer |
| --- | --- | --- | --- | --- | --- | --- | --- |
| X (copy of counts) | umi_counts | 8569 x 20125 | 16171764 | 0.0938 | 1.0 | 4318.0 | True |
| layer counts | umi_counts | 8569 x 20125 | 16171764 | 0.0938 | 1.0 | 4318.0 | True |
| layer logcounts | log_normalized | 8569 x 20125 | 16171764 | 0.0938 | 0.22513917088508606 | 11.809162139892578 | False |

## Cells per donor

| donor | cells |
| --- | --- |
| human1 | 1937 |
| human2 | 1724 |
| human3 | 3605 |
| human4 | 1303 |

## How feature identifiers resolved against NCBI gene_info

| resolution | features |
| --- | --- |
| ambiguous_name | 51 |
| official_symbol | 18765 |
| unique_synonym | 1258 |
| unmatched | 51 |

## Cell metadata columns (obs)

| name | type | missing | distinct |
| --- | --- | --- | --- |
| human | int64 | 0 | 4 |
| cell_type1 | category | 0 | 14 |
| total_features | int64 | 0 | 2398 |
| log10_total_features | float64 | 0 | 2398 |
| total_counts | int64 | 0 | 5570 |
| log10_total_counts | float64 | 0 | 5570 |
| pct_counts_top_50_features | float64 | 0 | 8544 |
| pct_counts_top_100_features | float64 | 0 | 8540 |
| pct_counts_top_200_features | float64 | 0 | 8541 |
| pct_counts_top_500_features | float64 | 0 | 8537 |
| total_features_endogenous | int64 | 0 | 2398 |
| log10_total_features_endogenous | float64 | 0 | 2398 |
| total_counts_endogenous | int64 | 0 | 5570 |
| log10_total_counts_endogenous | float64 | 0 | 5570 |
| pct_counts_endogenous | int64 | 0 | 1 |
| pct_counts_top_50_features_endogenous | float64 | 0 | 8544 |
| pct_counts_top_100_features_endogenous | float64 | 0 | 8540 |
| pct_counts_top_200_features_endogenous | float64 | 0 | 8541 |
| pct_counts_top_500_features_endogenous | float64 | 0 | 8537 |
| total_features_feature_control | int64 | 0 | 1 |
| log10_total_features_feature_control | int64 | 0 | 1 |
| total_counts_feature_control | int64 | 0 | 1 |
| log10_total_counts_feature_control | int64 | 0 | 1 |
| pct_counts_feature_control | int64 | 0 | 1 |
| total_features_ERCC | int64 | 0 | 1 |
| log10_total_features_ERCC | int64 | 0 | 1 |
| total_counts_ERCC | int64 | 0 | 1 |
| log10_total_counts_ERCC | int64 | 0 | 1 |
| pct_counts_ERCC | int64 | 0 | 1 |
| is_cell_control | bool | 0 | 1 |
| dataset | str | 0 | 1 |
| donor | string | 0 | 4 |
| label_original | string | 0 | 14 |

## Feature metadata columns (var)

| name | type | missing | distinct |
| --- | --- | --- | --- |
| feature_symbol | str | 0 | 20125 |
| is_feature_control | bool | 0 | 1 |
| is_feature_control_ERCC | bool | 0 | 1 |
| mean_counts | float64 | 0 | 4711 |
| log10_mean_counts | float64 | 0 | 4711 |
| rank_counts | float64 | 0 | 4711 |
| n_cells_counts | int64 | 0 | 3693 |
| pct_dropout_counts | float64 | 0 | 3693 |
| total_counts | int64 | 0 | 4711 |
| log10_total_counts | float64 | 0 | 4711 |
| feature_id_original | str | 0 | 20125 |
| symbol_original | string | 0 | 20125 |
| symbol | string | 0 | 20125 |
| symbol_repaired | bool | 0 | 2 |
| ncbi_gene_id | string | 102 | 19999 |
| id_resolution | string | 0 | 4 |
| is_spike_in | bool | 0 | 1 |
| is_mt | bool | 0 | 1 |
| shares_gene_id | bool | 0 | 2 |
