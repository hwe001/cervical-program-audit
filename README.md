# cervical-program-audit

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23070746.svg)](https://doi.org/10.5281/zenodo.23070746)

Code and aggregate results for a patient-aware audit of five candidate therapeutic-resistance programs in cervical cancer (ACKR2/TGF-beta, MDM2/DNA-damage response, T-cell senescence, hypoxia, CAF/ECM). The audit asks three separate questions of each program in cohorts chosen for that question: does it change with **treatment exposure**, does it track **tumor progression**, and does it separate **clinical outcome**. It also quantifies how cell-level validation overstates performance when patients are few.

Manuscript (in preparation): *Cell-level validation overstates computational resistance biomarkers: a patient-aware audit in cervical cancer.* [Add preprint link.]

## Main results (all reproducible from `results/`)

| Question | Cohorts | Result |
|---|---|---|
| Treatment exposure | GSE236738 (3 patients, single cell), GSE297038 (12 evaluable pairs, RNA-seq) | MDM2/DDR up in 3/3 patients and 12/12 pairs (p53 target genes); T-cell senescence down in 3/3 and 11/12 |
| Progression | GSE63514 (n=128) | hypoxia rho 0.64, T-cell senescence 0.48 (q < 1e-7) |
| Clinical outcome | GSE168009 (5 vs 4), GSE56363 (12 vs 9), GSE70035 (6 vs 6) | no program significant after correction (lowest q = 0.20 over the 15 tests); signs conflict between the two CCRT cohorts; cohorts detect only large effects |
| Prognosis (not treatment-specific) | TCGA-CESC (291 patients, 72 deaths) | hypoxia HR 1.87 per SD (q = 1e-4); T-cell/CDKN2A HR 0.71 (q = 0.006) |
| Leakage and power | simulation (independent and correlated genes), four public single-cell cohorts, four model families; arbitrary labels: random cell-level AUC 0.78-0.91 vs patient hold-out 0.52-0.56 in three cohorts of 6-10 patients | with batch offsets and no treatment effect, random cell-level CV gives AUC 0.97 (3 patients) vs 0.49 for leave-one-patient-out; in GSE236738 the gap is 0.86 vs 0.80 |

The full program-by-cohort table is `results/cross_cohort_table.csv`; figures are in `results/figures/`.

## Layout

```
code/         analysis scripts (Python)
results/      aggregate outputs (JSON/CSV) and figures; no individual-level data
work/         created by code/download_data.py (raw public data and intermediate files; git-ignored)
documentation/ notes on decisions and known problems
```

## Running

Python 3.10+ and the packages in `requirements.txt`. Run everything from the repository root.

```bash
pip install -r requirements.txt
python code/download_data.py            # public data into ./work
python code/analyze_gse236738.py        # single-cell scores and cell annotations
python code/gene_level_ai.py            # 55-gene leave-one-patient-out classifier
python code/leakage_check_infold.py     # in-fold preprocessing and label-flip control
python code/real_random_vs_lopo.py      # random cell-level CV vs leave-one-patient-out
python code/leakage_simulation.py       # simulation of cell-level leakage
python code/stability_analysis.py       # importance and stability
python code/paired_effects.py
python code/sample_level_benchmark.py
python code/composition_program_ablation.py
python code/analyze_gse168009.py
python code/analyze_gse56363.py
python code/analyze_gse70035.py
python code/analyze_gse297038.py
python code/analyze_tcga_survival.py
python code/tcga_ph_check.py            # proportional-hazards check (two-period interaction)
python code/analyze_tcga_adjusted.py      # Cox adjusted for age, FIGO stage, histology (needs clinical_covariates.json from GDC)
python code/score_gse63514.py
python code/score_gse6213.py
python code/benchmark_programs.py
python code/gene_level_gse63514.py
python code/build_cross_cohort_table.py
python code/validation_strategies.py    # random cell, sample split, LOPO, pseudobulk, label flips, effective sample size
python code/second_cohort_null_labels.py # second cohort (GSE224327): leakage with arbitrary labels
python code/model_family_benchmark.py   # logistic regression, random forest, gradient boosting, neural network
python code/correlated_simulation.py    # power with correlated genes and program-level effects
python code/additional_cohorts_benchmark.py  # GSE228499, GSE292163 (arbitrary labels), GSE237425 (paired, real labels); set CERVIX_EXTRA_DATA to the folder holding the three unpacked cohorts
python code/gse197461_benchmark.py      # cervical cohort GSE197461: arbitrary and real (histology, HPV) patient-level labels
python code/robustness_analyses.py     # CDKN2A split, GSE63514 by dissection method, TCGA Cox
python code/power_simulation.py         # power of leave-one-patient-out validation
python code/make_fig_leakage.py
python code/make_fig_heatmap_schematic.py
```

Programs are defined once per script from the same gene lists (29 genes, each gene in one program; the T-cell senescence program has six genes and CDKN1A belongs only to MDM2/DDR).

## Verified and not verified

- The analysis scripts were run in the original workspace, and the cohort scripts plus `build_cross_cohort_table.py` and the two figure scripts were re-run from this repository layout and reproduce `results/cross_cohort_table.csv` exactly.
- `code/download_data.py` was assembled from the original download scripts and has **not** been run end to end.
- The single-cell scripts (`analyze_gse236738.py`, `gene_level_ai.py`, `stability_analysis.py`, `paired_effects.py`, `sample_level_benchmark.py`, `composition_program_ablation.py`) use hard-coded random seeds but were not re-run from this layout.
- Spatial (Visium, Stereo-seq) scripts from the original project are not included: their results are not part of the audit.

## Known limitations

- Discovery cohort GSE236738 has three patients; treatment is confounded with sample and batch.
- GSE297038: 12 of 17 cases are usable. Five batch-1 cases have duplicated column names and a missing time point in the deposited count file and are excluded; see `documentation/notes.md`.
- GSE6213 has no associated publication on GEO; the GSE168009 source paper has not been confirmed.
- Program scores use five or six genes each, and program definitions come from the literature, some from studies used for interpretation.

## Data

Only public data are used (GEO accessions GSE236738, GSE297038, GSE6213, GSE63514, GSE168009, GSE56363, GSE70035; TCGA-CESC from GDC/Xena). Nothing here is patient-identifiable and no raw data are redistributed.

## Licence and citation

Code: MIT (see `LICENSE`). Please cite the manuscript and this repository (`CITATION.cff`).
