# Integrity report: muraro

Checks run: 33. Fatal failures: 0. Warnings to review: 3.

Features flagged for review are listed in `muraro_feature_issues.tsv`.

| group | check | severity | result | observed | expected | detail |
| --- | --- | --- | --- | --- | --- | --- |
| cell_ids | duplicate_cell_ids | fatal | pass | 0 | 0 |  |
| cell_ids | blank_cell_ids | fatal | pass | 0 | 0 |  |
| orientation | cell_axis_holds_cell_ids | fatal | pass | 0 malformed | all match ^D\d+-\d+_\d+$ |  |
| gene_ids | duplicate_feature_ids | fatal | pass | 0 | 0 |  |
| gene_ids | ambiguous_gene_names | warning | review | 59 | 0 | Name points to several NCBI genes, so it receives no Gene ID. ABP1__chr7, ADC__chr1, AGPAT9__chr4, AIM1__chr6, AZI1__chr17 |
| gene_ids | unmatched_gene_names | warning | review | 722 | 0 | Name is unknown to NCBI gene_info. AGSK1__chr15, CRIPAK__chr4, DKFZP686I15217__chr6, DKFZp434J0226__chr19, FLJ10038__chr15 |
| gene_ids | features_sharing_a_gene_id | warning | review | 98 | 0 | Several features resolve to one Gene ID. AGAP4__chr10, AGAP8__chr10, AKAP17A__chrX, AKAP17A__chrY, AKAP2__chr9 |
| values | nonfinite_in_X | fatal | pass | 0 | 0 |  |
| values | negative_in_X | fatal | pass | 0 | 0 |  |
| values | non_integer_in_X | fatal | pass | 0 | 0 | unit is declared as umi_counts |
| values | nonfinite_in_layer transcripts | fatal | pass | 0 | 0 |  |
| values | negative_in_layer transcripts | fatal | pass | 0 | 0 |  |
| values | nonfinite_in_layer umi | fatal | pass | 0 | 0 |  |
| values | negative_in_layer umi | fatal | pass | 0 | 0 |  |
| values | non_integer_in_layer umi | fatal | pass | 0 | 0 | unit is declared as umi_counts |
| values | cells_with_zero_total | warning | pass | 0 | 0 |  |
| values | cells_without_donor | fatal | pass | 0 | 0 |  |
| orientation | shape_is_cells_by_features | fatal | pass | 3072 x 19140 | 3072 x 19140 |  |
| orientation | feature_axis_holds_genes | fatal | pass | 6 of 6 marker genes on the feature axis | all 6 |  |
| expression_metadata | donor_column_matches_cell_id | fatal | pass | 0 disagreements | 0 |  |
| expression_metadata | layer_transcripts_same_nonzero_cells_as_X | fatal | pass | 0 positions differ | 0 |  |
| expression_metadata | layer_umi_same_nonzero_cells_as_X | fatal | pass | 0 positions differ | 0 |  |
| orientation | deposited_table_rows_are_features_and_columns_are_cells | fatal | pass | table 19140 rows x 3072 columns | 19140 features x 3072 cells |  |
| expression_metadata | deposited_table_totals_match_matrix | fatal | pass | 0 cell totals and 0 gene totals differ | 0 and 0 |  |
| expression_metadata | umi_recovery_gives_integers | fatal | pass | 0.999887 integer, 1408 at the saturation cap, 0 unexplained | >= 0.999 integer and 0 unexplained |  |
| expression_metadata | umi_layer_equals_recovered_values | fatal | pass | True | True |  |
| annotation | annotation_ids_are_unique | fatal | pass | 0 | 0 |  |
| annotation | annotation_ids_match_cell_ids | fatal | pass | 0 cells without annotation, 0 annotations without a cell | 0 and 0 |  |
| annotation | annotation_donor_matches_cell | fatal | pass | 0 | 0 |  |
| annotation | annotation_label_matches_cell | fatal | pass | 0 | 0 |  |
| annotation | cells_without_author_label | fatal | pass | 946 | 946 |  |
| annotation | label_map_lists_every_label | fatal | pass | 0 unlisted | 0 |  |
| annotation | marker_genes_peak_in_expected_class | fatal | pass | 6 of 6 correct | 6 of 6 | INS=beta; GCG=alpha; SST=delta; PPY=gamma; PRSS1=acinar; KRT19=ductal |
