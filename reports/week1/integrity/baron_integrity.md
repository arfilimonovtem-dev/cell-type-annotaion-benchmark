# Integrity report: baron

Checks run: 33. Fatal failures: 0. Warnings to review: 3.

Features flagged for review are listed in `baron_feature_issues.tsv`.

| group | check | severity | result | observed | expected | detail |
| --- | --- | --- | --- | --- | --- | --- |
| cell_ids | duplicate_cell_ids | fatal | pass | 0 | 0 |  |
| cell_ids | blank_cell_ids | fatal | pass | 0 | 0 |  |
| orientation | cell_axis_holds_cell_ids | fatal | pass | 0 malformed | all match ^human\d+_lib\d+\.final_cell_\d+$ |  |
| gene_ids | duplicate_feature_ids | fatal | pass | 0 | 0 |  |
| gene_ids | ambiguous_gene_names | warning | review | 51 | 0 | Name points to several NCBI genes, so it receives no Gene ID. AGPAT9, AIM1, C10orf2, C17orf112, C17orf49 |
| gene_ids | unmatched_gene_names | warning | review | 51 | 0 | Name is unknown to NCBI gene_info. BACH1.IT1, CRIPAK, DKFZp434J0226, DKFZp434P228, ERCC6.PGBD3 |
| gene_ids | features_sharing_a_gene_id | warning | review | 43 | 0 | Several features resolve to one Gene ID. AKAP2, ARHGAP42, C10orf113, C10orf12, C10orf131 |
| values | nonfinite_in_X | fatal | pass | 0 | 0 |  |
| values | negative_in_X | fatal | pass | 0 | 0 |  |
| values | non_integer_in_X | fatal | pass | 0 | 0 | unit is declared as umi_counts |
| values | nonfinite_in_layer counts | fatal | pass | 0 | 0 |  |
| values | negative_in_layer counts | fatal | pass | 0 | 0 |  |
| values | non_integer_in_layer counts | fatal | pass | 0 | 0 | unit is declared as umi_counts |
| values | nonfinite_in_layer logcounts | fatal | pass | 0 | 0 |  |
| values | negative_in_layer logcounts | fatal | pass | 0 | 0 |  |
| values | cells_with_zero_total | warning | pass | 0 | 0 |  |
| values | cells_without_donor | fatal | pass | 0 | 0 |  |
| orientation | shape_is_cells_by_features | fatal | pass | 8569 x 20125 | 8569 x 20125 |  |
| orientation | feature_axis_holds_genes | fatal | pass | 6 of 6 marker genes on the feature axis | all 6 |  |
| expression_metadata | donor_column_matches_cell_id | fatal | pass | 0 disagreements | 0 |  |
| expression_metadata | layer_counts_same_nonzero_cells_as_X | fatal | pass | 0 positions differ | 0 |  |
| expression_metadata | layer_logcounts_same_nonzero_cells_as_X | fatal | pass | 0 positions differ | 0 |  |
| expression_metadata | recorded_cells_total_counts_matches_matrix | fatal | pass | 0 mismatches | 0 |  |
| expression_metadata | recorded_cells_total_features_matches_matrix | fatal | pass | 0 mismatches | 0 |  |
| expression_metadata | recorded_features_total_counts_matches_matrix | fatal | pass | 0 mismatches | 0 |  |
| expression_metadata | recorded_features_n_cells_counts_matches_matrix | fatal | pass | 0 mismatches | 0 |  |
| annotation | annotation_ids_are_unique | fatal | pass | 0 | 0 |  |
| annotation | annotation_ids_match_cell_ids | fatal | pass | 0 cells without annotation, 0 annotations without a cell | 0 and 0 |  |
| annotation | annotation_donor_matches_cell | fatal | pass | 0 | 0 |  |
| annotation | annotation_label_matches_cell | fatal | pass | 0 | 0 |  |
| annotation | cells_without_author_label | fatal | pass | 0 | 0 |  |
| annotation | label_map_lists_every_label | fatal | pass | 0 unlisted | 0 |  |
| annotation | marker_genes_peak_in_expected_class | fatal | pass | 6 of 6 correct | 6 of 6 | INS=beta; GCG=alpha; SST=delta; PPY=gamma; PRSS1=acinar; KRT19=ductal |
