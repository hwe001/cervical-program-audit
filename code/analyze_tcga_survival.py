"""TCGA-CESC overall survival vs the five programs, using the Xena survival table (OS, OS.time) for all patients.

Replaces the earlier analysis, which took follow-up times from a partial GDC clinical pull (only 132 of 304 patients had a time,
mostly patients who had died) and therefore over-represented deaths.
Primary tumour samples only (barcode type 01), one per patient, OS.time > 0.
"""
import gzip, json, math
from pathlib import Path
import numpy as np, pandas as pd
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.stats.multitest import multipletests

W = Path(__file__).resolve().parents[1] / 'work'
OUT = W / 'analysis_tcga_cesc' / 'program_survival_summary.json'
OUT.parent.mkdir(exist_ok=True)
PROGRAMS = {'ACKR2_TGFb': ['ACKR2', 'TGFB1', 'TGFB2', 'TGFB3', 'TGFBR1', 'TGFBR2'], 'MDM2_DDR': ['MDM2', 'CDKN1A', 'GADD45A', 'DDB2', 'BBC3'],
            'Tcell_senescence': ['CD8A', 'CD8B', 'CDKN2A', 'TOX', 'TIGIT', 'LAG3'], 'Hypoxia': ['HIF1A', 'EPAS1', 'VEGFA', 'CA9', 'SLC2A1', 'LDHA'],
            'CAF_ECM': ['COL1A1', 'COL1A2', 'DCN', 'LUM', 'FAP', 'PDGFRA']}


def logrank(time, event, group):
    order = np.argsort(time); t = time[order]; e = event[order]; g = group[order]; u = 0.; v = 0.
    for tt in np.unique(t[e == 1]):
        risk = t >= tt; deaths = (t == tt) & (e == 1); d = int(deaths.sum()); n = int(risk.sum()); ng = int((risk & g).sum()); dg = int((deaths & g).sum())
        if n > 0:
            u += dg - d * ng / n; v += d * (ng / n) * (1 - ng / n) * ((n - d) / (n - 1) if n > 1 else 0)
    chi = float(u * u / v) if v > 0 else 0.
    return {'chi2': chi, 'p_value': float(math.erfc(math.sqrt(chi / 2))), 'observed_minus_expected_high': float(u)}


m = json.load(open(W / 'data_gse297038' / 'symbol2ensg.json')); e2s = {v.upper(): k for k, v in m.items()}
expr = {}; samp = None
with gzip.open(W / 'data_tcga_cesc' / 'TCGA-CESC.star_tpm.tsv.gz', 'rt', encoding='utf8') as f:
    for line in f:
        if line.startswith('#'): continue
        p = line.rstrip('\n').split('\t')
        if samp is None: samp = p[1:]; continue
        gname = p[0].split('|')[0].split('.')[0].upper()
        if gname in e2s: expr[e2s[gname]] = np.array([float(x) if x not in ('NA', '') else 0. for x in p[1:]], float)
surv = pd.read_csv(W / 'data_tcga_cesc' / 'surv.tsv.gz', sep='\t').set_index('sample')
keep, time, event, seen = [], [], [], set()
for i, s in enumerate(samp):
    if s[13:15] != '01' or s[:12] in seen or s not in surv.index: continue
    t = float(surv.loc[s, 'OS.time'])
    if not t > 0: continue
    seen.add(s[:12]); keep.append(i); time.append(t); event.append(int(surv.loc[s, 'OS']))
time = np.array(time); event = np.array(event)
res = {'n_patients': len(keep), 'events': int(event.sum()), 'median_follow_up_days_all': float(np.median(time)), 'programs': {}}
pm, pc = [], []
for pn, gs in PROGRAMS.items():
    sc = np.vstack([np.log2(expr[g][keep] + 1) for g in gs if g in expr]).mean(0)
    hi = sc >= np.median(sc); lr = logrank(time, event, hi)
    z = ((sc - sc.mean()) / sc.std(ddof=1)).reshape(-1, 1); fit = PHReg(time, z, status=event).fit(); ci = np.exp(fit.conf_int()[0])
    res['programs'][pn] = {'genes_found': [g for g in gs if g in expr], 'logrank_median_split': lr, 'cox_HR_per_SD': float(np.exp(fit.params[0])), 'cox_CI95': [float(ci[0]), float(ci[1])], 'cox_p': float(fit.pvalues[0])}
    pm.append(lr['p_value']); pc.append(float(fit.pvalues[0]))
for pn, a, b in zip(PROGRAMS, multipletests(pm, method='fdr_bh')[1], multipletests(pc, method='fdr_bh')[1]):
    res['programs'][pn]['logrank_BH_q'] = float(a); res['programs'][pn]['cox_BH_q'] = float(b)
OUT.write_text(json.dumps(res, indent=2))
print(res['n_patients'], res['events'])
for pn, v in res['programs'].items():
    print(pn, 'HR %.2f (%.2f-%.2f) p=%.4f q=%.4f | logrank p=%.3f' % (v['cox_HR_per_SD'], *v['cox_CI95'], v['cox_p'], v['cox_BH_q'], v['logrank_median_split']['p_value']))
