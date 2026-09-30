"""Assemble the cross-cohort table (results/cross_cohort_table.csv) from the per-cohort result files.

Run from anywhere after the per-cohort analyses. Reads work/analysis_* and work/validation outputs.
"""
import json
from pathlib import Path

import pandas as pd
from statsmodels.stats.multitest import multipletests

W = Path(__file__).resolve().parents[1] / 'work'
OUT = Path(__file__).resolve().parents[1] / 'results' / 'cross_cohort_table.csv'
P = ['ACKR2_TGFb', 'MDM2_DDR', 'Tcell_senescence', 'Hypoxia', 'CAF_ECM']
rows = []


def add(cohort, endpoint, n, direction, program, effect, p, q, role):
    rows.append(dict(cohort=cohort, endpoint=endpoint, n=n, direction=direction, program=program, effect=effect, p=p, BH_q=q, role=role))


for cohort, path, k_eff, k_p, endpoint, n, direction, role in [
        ('GSE168009', 'analysis_gse168009', 'mean_NDB_minus_DCB', 'exact_permutation_p_two_sided', 'no durable benefit vs durable (CCRT)', '4 vs 5', 'NDB-DCB', 'outcome'),
        ('GSE56363', 'analysis_gse56363', 'mean_difference_NCR_minus_CR', 'exact_permutation_p', 'non-complete vs complete response (CRT)', '9 vs 12', 'NCR-CR', 'outcome'),
        ('GSE70035', 'analysis_gse70035', 'mean_NR_minus_R', 'exact_permutation_p_two_sided', 'non-responder vs responder (neoadjuvant chemo)', '6 vs 6', 'NR-R', 'context control')]:
    g = json.load(open(W / path / 'program_response_summary.json'))['programs']
    q = multipletests([g[p][k_p] for p in P], method='fdr_bh')[1]
    for p, qq in zip(P, q):
        add(cohort, endpoint, n, direction, p, g[p][k_eff], g[p][k_p], qq, role)

for r in json.load(open(W / 'validation' / 'gse63514_gene_level_progression.json'))['program_level_bh_corrected']:
    add('GSE63514', 'ordinal normal/CIN/cancer (Spearman rho)', '128', 'rho', r['program'], r['rho'], r['p'], r['q_bh_across_5_programs'], 'tumour progression')

t = json.load(open(W / 'analysis_tcga_cesc' / 'program_outcome_summary.json'))['programs']
q = multipletests([t[p]['logrank']['p_value'] for p in P], method='fdr_bh')[1]
for p, qq in zip(P, q):
    add('TCGA-CESC', 'OS median-split log-rank', '134 (74 events)', 'high vs low', p, None, t[p]['logrank']['p_value'], qq, 'descriptive')

s = json.load(open(W / 'analysis_gse297038' / 'summary.json'))['primary_clean_pairs']['programs']
for p in P:
    add('GSE297038', 'paired pre-RT vs week-3 CRT', '12 pairs', '3wk-pre', p, s[p]['mean_delta_3wk_minus_pre'], s[p]['exact_signflip_p'], s[p]['BH_q_five_programs'], 'exposure (paired)')

df = pd.DataFrame(rows)
df['BH_q_all_tests'] = multipletests(df.p, method='fdr_bh')[1]
df.to_csv(OUT, index=False)
o = df[df.cohort.isin(['GSE168009', 'GSE56363', 'GSE70035'])]
print('rows', len(df), '| lowest q across the 15 outcome tests:', round(multipletests(o.p, method='fdr_bh')[1].min(), 3))
