import itertools, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

ROOT = (Path(__file__).resolve().parents[1] / 'work')
OUT = ROOT / 'analysis_gse297038'; OUT.mkdir(exist_ok=True)
P = {'ACKR2_TGFb': ['ACKR2', 'TGFB1', 'TGFB2', 'TGFB3', 'TGFBR1', 'TGFBR2'], 'MDM2_DDR': ['MDM2', 'CDKN1A', 'GADD45A', 'DDB2', 'BBC3'],
     'Tcell_senescence': ['CD8A', 'CD8B', 'CDKN2A', 'TOX', 'TIGIT', 'LAG3'], 'Hypoxia': ['HIF1A', 'EPAS1', 'VEGFA', 'CA9', 'SLC2A1', 'LDHA'],
     'CAF_ECM': ['COL1A1', 'COL1A2', 'DCN', 'LUM', 'FAP', 'PDGFRA']}
m = json.load(open(ROOT / 'data_gse297038' / 'symbol2ensg.json'))
d = pd.read_csv(ROOT / 'data_gse297038' / 'GSE297038_17pairs_counts_merged.tsv.gz', sep='\t', index_col=0)
d.index = d.index.str.split('.').str[0]
cpm = np.log2(d / d.sum() * 1e6 + 1)
g = cpm.loc[[m[s] for s in m]]; g.index = list(m)
cols = pd.DataFrame({'col': d.columns})
cols['case'] = cols.col.str.extract(r'700514_(\d+)_')[0]
cols['time'] = cols.col.str.extract(r'_(preRT|3weeks)')[0]
cols['dup'] = cols.col.str.endswith('.1')
info = cols.groupby(['case', 'time']).size().unstack(fill_value=0)
complete = info[(info.preRT == 1) & (info['3weeks'] == 1)].index.tolist()
broken = info[~info.index.isin(complete)].index.tolist()
print('complete clean pairs:', len(complete), 'broken/duplicated:', len(broken), broken)
# diagnostic for the duplicated batch-1 columns
dup_cases = {}
for c in broken:
    ccols = cols[cols.case == c]
    v = [cpm[x].values for x in ccols.col]
    dup_cases[c] = {'columns': ccols.col.tolist(), 'pearson_log_cpm': float(np.corrcoef(v[0], v[1])[0, 1])}
z = g.T.copy(); z = (z - z.mean()) / z.std(ddof=1)  # within-gene standardisation across ALL columns used below


def run(case_list, tag):
    sel = cols[cols.case.isin(case_list) & ~cols.dup]
    sc = g[sel.col].T; sc = (sc - sc.mean()) / sc.std(ddof=1)
    res = {}
    for p, gl in P.items():
        s = sc[gl].mean(axis=1)
        pre = np.array([s[sel[(sel.case == c) & (sel.time == 'preRT')].col.iloc[0]] for c in case_list])
        post = np.array([s[sel[(sel.case == c) & (sel.time == '3weeks')].col.iloc[0]] for c in case_list])
        diff = post - pre
        obs = diff.mean()
        signs = np.array(list(itertools.product([1, -1], repeat=len(diff))))
        null = (signs * diff).mean(axis=1)
        pval = float((np.abs(null) >= abs(obs) - 1e-12).mean())
        res[p] = {'mean_delta_3wk_minus_pre': float(obs), 'n_up': int((diff > 0).sum()), 'n_pairs': len(diff), 'exact_signflip_p': pval, 'sd_delta': float(diff.std(ddof=1)), 'deltas': diff.tolist()}
    q = multipletests([res[p]['exact_signflip_p'] for p in P], method='fdr_bh')[1]
    for p, qq in zip(P, q): res[p]['BH_q_five_programs'] = float(qq)
    # gene-level for the MDM2 axis
    gene = {}
    for gn in ['MDM2', 'CDKN1A', 'GADD45A', 'DDB2', 'BBC3', 'CDKN2A']:
        pre = np.array([sc.loc[sel[(sel.case == c) & (sel.time == 'preRT')].col.iloc[0], gn] for c in case_list])
        post = np.array([sc.loc[sel[(sel.case == c) & (sel.time == '3weeks')].col.iloc[0], gn] for c in case_list])
        diff = post - pre; signs = np.array(list(itertools.product([1, -1], repeat=len(diff)))); null = (signs * diff).mean(axis=1)
        gene[gn] = {'mean_delta': float(diff.mean()), 'n_up': int((diff > 0).sum()), 'p': float((np.abs(null) >= abs(diff.mean()) - 1e-12).mean())}
    return {'set': tag, 'n_pairs': len(case_list), 'programs': res, 'genes': gene}


out = {'complete_pairs': complete, 'unresolved_batch1_cases': dup_cases, 'primary_clean_pairs': run(complete, 'clean pairs only'),
       'note': 'Batch 1 cases have duplicated column names and are missing one time point each in the merged count file; they cannot be paired reliably and are excluded.'}
# batch-adjusted check: are results consistent within batch 2 only (all clean pairs are batch 2)
(OUT / 'summary.json').write_text(json.dumps(out, indent=2))
r = out['primary_clean_pairs']
print(json.dumps(dup_cases, indent=1))
for p, v in r['programs'].items(): print(p, round(v['mean_delta_3wk_minus_pre'], 3), f"{v['n_up']}/{v['n_pairs']} up", 'p=%.4f q=%.4f' % (v['exact_signflip_p'], v['BH_q_five_programs']))
print(r['genes'])
