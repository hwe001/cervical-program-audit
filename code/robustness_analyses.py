"""Robustness analyses requested after review:
1. T-cell senescence program split into CDKN2A alone vs the T-cell genes (CD8A, CD8B, TOX, TIGIT, LAG3).
2. GSE63514 progression associations stratified by dissection method.
3. TCGA-CESC continuous Cox models (one tumour sample per patient).
"""
import gzip, io, itertools, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.stats.multitest import multipletests

W = Path(__file__).resolve().parents[1] / 'work'
OUT = Path(__file__).resolve().parents[1] / 'results' / 'robustness_analyses.json'
sys.path.insert(0, str(W / 'validation'))
import gene_level_gse63514 as gl

TC = ['CD8A', 'CD8B', 'TOX', 'TIGIT', 'LAG3']
res = {}

# ---- 1. split of the T-cell senescence program
expr, stage, samples = gl.load_expr_and_stage()
stage = stage.dropna(); samples = [s for s in samples if s in stage.index]
z = lambda df: (df.T - df.T.mean()) / df.T.std(ddof=1)
E = expr.loc[:, samples]
out63514 = {}
for name, genes in [('CDKN2A_only', ['CDKN2A']), ('T_cell_genes_without_CDKN2A', TC), ('T_cell_senescence_full', ['CD8A', 'CD8B', 'CDKN2A', 'TOX', 'TIGIT', 'LAG3'])]:
    sc = z(E.loc[[g for g in genes if g in E.index]]).mean(axis=1)
    r, p = spearmanr(stage.loc[samples], sc.loc[samples]); out63514[name] = {'rho': float(r), 'p': float(p)}
res['split_GSE63514_stage'] = out63514

m = json.load(open(W / 'data_gse297038' / 'symbol2ensg.json'))
d = pd.read_csv(W / 'data_gse297038' / 'GSE297038_17pairs_counts_merged.tsv.gz', sep='\t', index_col=0)
d.index = d.index.str.split('.').str[0]
cpm = np.log2(d / d.sum() * 1e6 + 1)
g = cpm.loc[[m[s] for s in m]]; g.index = list(m)
cols = pd.DataFrame({'col': d.columns}); cols['case'] = cols.col.str.extract(r'700514_(\d+)_')[0]; cols['time'] = cols.col.str.extract(r'_(preRT|3weeks)')[0]; cols['dup'] = cols.col.str.endswith('.1')
info = cols.groupby(['case', 'time']).size().unstack(fill_value=0)
cases = info[(info.preRT == 1) & (info['3weeks'] == 1)].index.tolist()
sel = cols[cols.case.isin(cases) & ~cols.dup]
sc_all = g[sel.col].T; sc_all = (sc_all - sc_all.mean()) / sc_all.std(ddof=1)
out297038 = {}
for name, genes in [('CDKN2A_only', ['CDKN2A']), ('T_cell_genes_without_CDKN2A', TC)]:
    s = sc_all[genes].mean(axis=1)
    pre = np.array([s[sel[(sel.case == c) & (sel.time == 'preRT')].col.iloc[0]] for c in cases]); post = np.array([s[sel[(sel.case == c) & (sel.time == '3weeks')].col.iloc[0]] for c in cases])
    diff = post - pre; signs = np.array(list(itertools.product([1, -1], repeat=len(diff)))); null = (signs * diff).mean(axis=1)
    out297038[name] = {'mean_delta': float(diff.mean()), 'n_up': int((diff > 0).sum()), 'n_pairs': len(diff), 'exact_signflip_p': float((np.abs(null) >= abs(diff.mean()) - 1e-12).mean())}
res['split_GSE297038_paired'] = out297038

# ---- 2. GSE63514 by dissection method
with gzip.open(W / 'validation' / 'GSE63514_series_matrix.txt.gz', 'rt', errors='replace') as f:
    lines = f.readlines()
begin = next(i for i, x in enumerate(lines) if x.startswith('!series_matrix_table_begin'))
hdr = lines[begin + 1].rstrip('\n').split('\t')[1:]; hdr = [h.strip('"') for h in hdr]
diss = {}
for line in lines[:begin]:
    if line.startswith('!Sample_characteristics_ch1') and 'dissection:' in line:
        vals = [x.strip('"').strip() for x in line.rstrip('\n').split('\t')[1:]]
        diss = {s: v.split(':', 1)[-1].strip().replace('laser-captured', 'laser captured') for s, v in zip(hdr, vals)}
meta = pd.DataFrame({'stage': stage.loc[samples], 'diss': pd.Series(diss).reindex(samples)})
res['GSE63514_stage_by_dissection_counts'] = {str(k): int(v) for k, v in meta.groupby(['diss', 'stage']).size().items()}
progs = {'ACKR2_TGFb': ['ACKR2', 'TGFB1', 'TGFB2', 'TGFB3', 'TGFBR1', 'TGFBR2'], 'MDM2_DDR': ['MDM2', 'CDKN1A', 'GADD45A', 'DDB2', 'BBC3'],
         'Tcell_senescence': ['CD8A', 'CD8B', 'CDKN2A', 'TOX', 'TIGIT', 'LAG3'], 'Hypoxia': ['HIF1A', 'EPAS1', 'VEGFA', 'CA9', 'SLC2A1', 'LDHA'], 'CAF_ECM': ['COL1A1', 'COL1A2', 'DCN', 'LUM', 'FAP', 'PDGFRA']}
byd = {}
for pn, genes in progs.items():
    sc = z(E.loc[[x for x in genes if x in E.index]]).mean(axis=1).loc[samples]
    row = {}
    for dv, sub in meta.groupby('diss'):
        if sub.stage.nunique() > 2 and len(sub) >= 10:
            r, p = spearmanr(sub.stage, sc.loc[sub.index]); row[str(dv)] = {'n': int(len(sub)), 'rho': float(r), 'p': float(p)}
    # rank regression of program score on stage and dissection method
    X = np.column_stack([np.ones(len(meta)), rankdata(meta.stage), pd.get_dummies(meta.diss, drop_first=True).astype(float).values])
    yv = rankdata(sc.loc[meta.index].values); beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ beta; sig2 = resid @ resid / (len(yv) - X.shape[1]); cov = sig2 * np.linalg.inv(X.T @ X)
    row['adjusted_rank_regression_stage_beta'] = float(beta[1]); row['adjusted_se'] = float(np.sqrt(cov[1, 1]))
    byd[pn] = row
res['GSE63514_by_dissection'] = byd

# ---- 3. TCGA-CESC continuous Cox
root = W / 'data_tcga_cesc'
e2s = {v.upper(): k for k, v in m.items()}
expr_t = {}; samp = None; codes = None
with gzip.open(root / 'TCGA-CESC.star_tpm.tsv.gz', 'rt', encoding='utf8') as f:
    for line in f:
        if line.startswith('#'): continue
        p = line.rstrip('\n').split('\t')
        if samp is None: samp = p[1:]; continue
        gname = p[0].split('|')[0].split('.')[0].upper()
        if gname in e2s: expr_t[e2s[gname]] = np.array([float(x) if x not in ('NA', '') else 0. for x in p[1:]], float)
clin = {x['submitter_id']: x for x in json.loads((root / 'clinical.json').read_text())['data']['hits']}
seen = set(); keep = []; time = []; event = []; drop_nontumour = 0; drop_dup = 0
for i, s in enumerate(samp):
    if len(s) >= 15 and s[13:15] != '01': drop_nontumour += 1; continue
    pat = s[:12]
    if pat in seen: drop_dup += 1; continue
    c = clin.get(pat)
    if not c: continue
    dm = c.get('demographic', {}); dx = (c.get('diagnoses') or [{}])[0]
    t = dm.get('days_to_death') or dx.get('days_to_last_follow_up')
    if t is None or float(t) <= 0: continue
    seen.add(pat); keep.append(i); time.append(float(t)); event.append(1 if str(dm.get('vital_status', '')).lower() == 'dead' else 0)
time = np.array(time); event = np.array(event)
cox = {'n_patients': len(keep), 'events': int(event.sum()), 'dropped_non_primary_samples': drop_nontumour, 'dropped_duplicate_patient_samples': drop_dup, 'programs': {}}
pl = []
for pn, genes in progs.items():
    sc = np.vstack([np.log2(expr_t[x][keep] + 1) for x in genes if x in expr_t]).mean(0); sc = (sc - sc.mean()) / sc.std(ddof=1)
    fit = PHReg(time, sc.reshape(-1, 1), status=event).fit()
    hr = float(np.exp(fit.params[0])); ci = np.exp(fit.conf_int()[0]); p = float(fit.pvalues[0])
    cox['programs'][pn] = {'HR_per_SD': hr, 'CI95': [float(ci[0]), float(ci[1])], 'p': p}; pl.append(p)
for pn, q in zip(progs, multipletests(pl, method='fdr_bh')[1]): cox['programs'][pn]['BH_q'] = float(q)
res['TCGA_cox'] = cox
OUT.write_text(json.dumps(res, indent=2)); print(json.dumps(res, indent=2))
