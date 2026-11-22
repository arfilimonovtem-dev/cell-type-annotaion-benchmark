# Reproducibility report

Config `configs/pilot.yaml`, tolerance 1e-10, OMP_NUM_THREADS=None.

**39 of 39 checks passed.**

| check | result | detail |
| --- | --- | --- |
| split files identical | PASS | 6/6 files with equal decompressed content |
| gene universe pancreas_pilot_v1.tsv identical | PASS | sha256 adbea6f84cbdf163 |
| gene universe pancreas_pilot_v1_excluded.tsv identical | PASS | sha256 32a40635ffa7a930 |
| same seed reproduces exp-e1__ref-baron__qry-baron.human4__task-pancreas_harmonized_v1__pert-none__seed-00__method-linear_svm | PASS | max numeric difference 0; byte-identical: True |
| same seed reproduces exp-e1__ref-baron__qry-baron.human4__task-pancreas_harmonized_v1__pert-none__seed-00__method-pca_knn | PASS | max numeric difference 0; byte-identical: True |
| same seed reproduces exp-e1__ref-baron__qry-baron.human4__task-pancreas_harmonized_v1__pert-none__seed-01__method-linear_svm | PASS | max numeric difference 0; byte-identical: True |
| same seed reproduces exp-e1__ref-baron__qry-baron.human4__task-pancreas_harmonized_v1__pert-none__seed-01__method-pca_knn | PASS | max numeric difference 0; byte-identical: True |
| same seed reproduces exp-e2__ref-baron__qry-muraro__task-pancreas_harmonized_v1__pert-none__seed-00__method-linear_svm | PASS | max numeric difference 0; byte-identical: True |
| same seed reproduces exp-e2__ref-baron__qry-muraro__task-pancreas_harmonized_v1__pert-none__seed-00__method-pca_knn | PASS | max numeric difference 0; byte-identical: True |
| same seed reproduces exp-e2__ref-baron__qry-muraro__task-pancreas_harmonized_v1__pert-none__seed-01__method-linear_svm | PASS | max numeric difference 0; byte-identical: True |
| same seed reproduces exp-e2__ref-baron__qry-muraro__task-pancreas_harmonized_v1__pert-none__seed-01__method-pca_knn | PASS | max numeric difference 0; byte-identical: True |
| same seed reproduces exp-e3__ref-baron__qry-muraro__task-pancreas_harmonized_v1__pert-remove.delta__seed-00__method-linear_svm | PASS | max numeric difference 0; byte-identical: True |
| same seed reproduces exp-e3__ref-baron__qry-muraro__task-pancreas_harmonized_v1__pert-remove.delta__seed-00__method-pca_knn | PASS | max numeric difference 0; byte-identical: True |
| same seed reproduces exp-e3__ref-baron__qry-muraro__task-pancreas_harmonized_v1__pert-remove.delta__seed-01__method-linear_svm | PASS | max numeric difference 0; byte-identical: True |
| same seed reproduces exp-e3__ref-baron__qry-muraro__task-pancreas_harmonized_v1__pert-remove.delta__seed-01__method-pca_knn | PASS | max numeric difference 0; byte-identical: True |
| exp-e1__ref-baron__qry-baron.human4__task-pancreas_harmonized_v1__pert-none seed 00 vs 01: train/train_reserve draw changes | PASS | 1932 cells moved between ['train', 'train_reserve'] |
| exp-e1__ref-baron__qry-baron.human4__task-pancreas_harmonized_v1__pert-none seed 00 vs 01: only train vs train_reserve moved | PASS | roles touched: ['train', 'train_reserve'] |
| exp-e1__ref-baron__qry-baron.human4__task-pancreas_harmonized_v1__pert-none seed 00 vs 01: donor roles unchanged | PASS | 4 (donor, role) pairs |
| exp-e1__ref-baron__qry-baron.human4__task-pancreas_harmonized_v1__pert-none seed 00 vs 01: calibration cells unchanged | PASS | 1937 cells |
| exp-e1__ref-baron__qry-baron.human4__task-pancreas_harmonized_v1__pert-none seed 00 vs 01: evaluation cells unchanged | PASS | 1303 cells |
| exp-e1__ref-baron__qry-baron.human4__task-pancreas_harmonized_v1__pert-none seed 00 vs 01: evaluation_open_set cells unchanged | PASS | 0 cells |
| exp-e1__ref-baron__qry-baron.human4__task-pancreas_harmonized_v1__pert-none seed 00 vs 01: excluded cells unchanged | PASS | 0 cells |
| exp-e1__ref-baron__qry-baron.human4__task-pancreas_harmonized_v1__pert-none seed 00 vs 01: vocabulary unchanged | PASS | acinar, alpha, beta, delta, ductal, endothelial, epsilon, gamma, macrophage, mast, schwann, stellate, t_cell |
| exp-e2__ref-baron__qry-muraro__task-pancreas_harmonized_v1__pert-none seed 00 vs 01: train/train_reserve draw changes | PASS | 2344 cells moved between ['train', 'train_reserve'] |
| exp-e2__ref-baron__qry-muraro__task-pancreas_harmonized_v1__pert-none seed 00 vs 01: only train vs train_reserve moved | PASS | roles touched: ['train', 'train_reserve'] |
| exp-e2__ref-baron__qry-muraro__task-pancreas_harmonized_v1__pert-none seed 00 vs 01: donor roles unchanged | PASS | 16 (donor, role) pairs |
| exp-e2__ref-baron__qry-muraro__task-pancreas_harmonized_v1__pert-none seed 00 vs 01: calibration cells unchanged | PASS | 1937 cells |
| exp-e2__ref-baron__qry-muraro__task-pancreas_harmonized_v1__pert-none seed 00 vs 01: evaluation cells unchanged | PASS | 2037 cells |
| exp-e2__ref-baron__qry-muraro__task-pancreas_harmonized_v1__pert-none seed 00 vs 01: evaluation_open_set cells unchanged | PASS | 0 cells |
| exp-e2__ref-baron__qry-muraro__task-pancreas_harmonized_v1__pert-none seed 00 vs 01: excluded cells unchanged | PASS | 1031 cells |
| exp-e2__ref-baron__qry-muraro__task-pancreas_harmonized_v1__pert-none seed 00 vs 01: vocabulary unchanged | PASS | acinar, alpha, beta, delta, ductal, endothelial, epsilon, gamma, macrophage, mast, schwann, stellate, t_cell |
| exp-e3__ref-baron__qry-muraro__task-pancreas_harmonized_v1__pert-remove.delta seed 00 vs 01: train/train_reserve draw changes | PASS | 2344 cells moved between ['train', 'train_reserve'] |
| exp-e3__ref-baron__qry-muraro__task-pancreas_harmonized_v1__pert-remove.delta seed 00 vs 01: only train vs train_reserve moved | PASS | roles touched: ['train', 'train_reserve'] |
| exp-e3__ref-baron__qry-muraro__task-pancreas_harmonized_v1__pert-remove.delta seed 00 vs 01: donor roles unchanged | PASS | 24 (donor, role) pairs |
| exp-e3__ref-baron__qry-muraro__task-pancreas_harmonized_v1__pert-remove.delta seed 00 vs 01: calibration cells unchanged | PASS | 1723 cells |
| exp-e3__ref-baron__qry-muraro__task-pancreas_harmonized_v1__pert-remove.delta seed 00 vs 01: evaluation cells unchanged | PASS | 1844 cells |
| exp-e3__ref-baron__qry-muraro__task-pancreas_harmonized_v1__pert-remove.delta seed 00 vs 01: evaluation_open_set cells unchanged | PASS | 193 cells |
| exp-e3__ref-baron__qry-muraro__task-pancreas_harmonized_v1__pert-remove.delta seed 00 vs 01: excluded cells unchanged | PASS | 1632 cells |
| exp-e3__ref-baron__qry-muraro__task-pancreas_harmonized_v1__pert-remove.delta seed 00 vs 01: vocabulary unchanged | PASS | acinar, alpha, beta, ductal, endothelial, epsilon, gamma, macrophage, mast, schwann, stellate, t_cell |
