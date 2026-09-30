# Notes and decisions

## Program definitions (fixed on 2026-10-01)
Each of the 29 program genes belongs to exactly one program. Earlier runs scored the T-cell senescence program with seven genes including CDKN1A (also in MDM2/DDR) in some analyses and with six genes in others; all cohorts were re-scored with six genes. Changes: T-cell senescence now falls in 3/3 discovery patients and more strongly in GSE297038 (-1.04, q=0.0098, was -0.64, q=0.062); TCGA p-values changed because an earlier run failed to look up gene IDs and scored only 3 to 4 genes per program (all 29 genes are now scored).

## GSE297038 (17 pairs deposited, 12 used)
The merged count file `GSE297038_17pairs_counts_merged.tsv.gz` has 34 columns but five batch-1 cases (848331, 848746, 848914, 848915, 849729) appear with a duplicated column name (suffix `.1`) and one time point missing each. The duplicated columns correlate at r = 0.91 to 0.96 (log CPM), so they are likely technical duplicates of one time point. These cases cannot be paired and are excluded. Only the 12 batch-2 cases with one preRT and one 3weeks column are analyzed. Consider asking the data authors for the correct sample sheet.

## Leakage checks
- Original scripts fitted the scaler and SVD on all cells; fitting them within each training fold changes mean AUC from 0.814 to 0.817 (no material leakage from this step).
- Flipping the treatment labels of a held-out patient lowers that patient's AUC to 0.11-0.22, so the classifier learns a treatment direction shared across patients.
- Treatment remains confounded with sample/batch in GSE236738 (each condition is a separate library).

## Not included
Visium and Stereo-seq analyses, the pharmacological triage composite score, and GSE3578, GSE190075, GSE52903, GSE224327: not used for any claim in the manuscript.
