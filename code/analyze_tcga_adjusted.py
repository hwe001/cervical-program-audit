"""TCGA-CESC Cox models adjusted for age, FIGO stage group and histology (complete cases), one primary tumour per patient.
Treatment is not available in the pulled fields, so the models remain prognostic, not treatment-specific."""
import gzip, json
from pathlib import Path
import numpy as np, pandas as pd
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.stats.multitest import multipletests

W = Path(__file__).resolve().parents[1] / 'work'
RES = Path(__file__).resolve().parents[1] / 'results'
PROGRAMS = {'ACKR2_TGFb': ['ACKR2', 'TGFB1', 'TGFB2', 'TGFB3', 'TGFBR1', 'TGFBR2'], 'MDM2_DDR': ['MDM2', 'CDKN1A', 'GADD45A', 'DDB2', 'BBC3'],
            'Tcell_senescence': ['CD8A', 'CD8B', 'CDKN2A', 'TOX', 'TIGIT', 'LAG3'], 'Hypoxia': ['HIF1A', 'EPAS1', 'VEGFA', 'CA9', 'SLC2A1', 'LDHA'],
            'CAF_ECM': ['COL1A1', 'COL1A2', 'DCN', 'LUM', 'FAP', 'PDGFRA']}
m = json.load(open(W / 'data_gse297038' / 'symbol2ensg.json')); e2s = {v.upper(): k for k, v in m.items()}
expr = {}; samp = None
with gzip.open(W / 'data_tcga_cesc' / 'TCGA-CESC.star_tpm.tsv.gz', 'rt', encoding='utf8') as f:
    for line in f:
        if line.startswith('#'): continue
        p = line.rstrip('\n').split('\t')
        if samp is None: samp = p[1:]; continue
        g = p[0].split('|')[0].split('.')[0].upper()
        if g in e2s: expr[e2s[g]] = np.array([float(x) if x not in ('NA', '') else 0. for x in p[1:]], float)
surv = pd.read_csv(W / 'data_tcga_cesc' / 'surv.tsv.gz', sep='\t').set_index('sample')
clin = {h['submitter_id']: h for h in json.load(open(W / 'data_tcga_cesc' / 'clinical_covariates.json'))['data']['hits']}


def stage_group(x):
    if not x: return np.nan
    x = x.replace('Stage ', '').upper()
    return 0 if x.startswith('I') and not x.startswith('II') and not x.startswith('IV') else (1 if x.startswith('II') and not x.startswith('III') else 2)


rows = []; seen = set()
for i, s in enumerate(samp):
    if s[13:15] != '01' or s[:12] in seen or s not in surv.index: continue
    t = float(surv.loc[s, 'OS.time'])
    if not t > 0: continue
    c = clin.get(s[:12], {}); dx = (c.get('diagnoses') or [{}])[0]; age = (c.get('demographic') or {}).get('age_at_index')
    hist = 1.0 if 'squamous' in (dx.get('primary_diagnosis') or '').lower() else 0.0
    seen.add(s[:12]); rows.append(dict(i=i, time=t, event=int(surv.loc[s, 'OS']), age=age, stage=stage_group(dx.get('figo_stage')), squamous=hist))
df = pd.DataFrame(rows).dropna(subset=['age', 'stage']).reset_index(drop=True)
idx = df.i.values
out = {'n_patients': int(len(df)), 'events': int(df.event.sum()), 'covariates': 'age (per 10 y), FIGO stage group (I, II, III-IV as 0/1/2), squamous histology', 'programs': {}}
pvals = []
for pn, gs in PROGRAMS.items():
    sc = np.vstack([np.log2(expr[g][idx] + 1) for g in gs if g in expr]).mean(0); z = (sc - sc.mean()) / sc.std(ddof=1)
    X = np.column_stack([z, df.age.values / 10.0, df.stage.values, df.squamous.values])
    fit = PHReg(df.time.values, X, status=df.event.values).fit(); ci = np.exp(fit.conf_int()[0])
    out['programs'][pn] = {'adjusted_HR_per_SD': float(np.exp(fit.params[0])), 'CI95': [float(ci[0]), float(ci[1])], 'p': float(fit.pvalues[0])}
    pvals.append(float(fit.pvalues[0]))
for pn, q in zip(PROGRAMS, multipletests(pvals, method='fdr_bh')[1]): out['programs'][pn]['BH_q'] = float(q)
out['note'] = 'Proportional hazards were not formally tested; treatment and HPV status were not available.'
(RES / 'tcga_cox_adjusted.json').write_text(json.dumps(out, indent=2))
print(out['n_patients'], out['events'])
for pn, v in out['programs'].items(): print(pn, 'HR %.2f (%.2f-%.2f) p=%.4f q=%.4f' % (v['adjusted_HR_per_SD'], *v['CI95'], v['p'], v['BH_q']))
