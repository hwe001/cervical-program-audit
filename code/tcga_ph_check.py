"""Proportional-hazards check for the adjusted TCGA-CESC Cox models: episode split at the median event time,
with a program x late-period interaction (counting-process form). A significant interaction means the hazard ratio changes over follow-up."""
import gzip, json
from pathlib import Path
import numpy as np, pandas as pd
from statsmodels.duration.hazard_regression import PHReg

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
    seen.add(s[:12]); rows.append(dict(i=i, time=t, event=int(surv.loc[s, 'OS']), age=age, stage=stage_group(dx.get('figo_stage')), squamous=1.0 if 'squamous' in (dx.get('primary_diagnosis') or '').lower() else 0.0))
df = pd.DataFrame(rows).dropna(subset=['age', 'stage']).reset_index(drop=True); idx = df.i.values
tc = float(np.median(df.time[df.event == 1]))
out = {'n_patients': int(len(df)), 'events': int(df.event.sum()), 'split_time_days': tc, 'programs': {}}
for pn, gs in PROGRAMS.items():
    sc = np.vstack([np.log2(expr[g][idx] + 1) for g in gs if g in expr]).mean(0); z = (sc - sc.mean()) / sc.std(ddof=1)
    base = np.column_stack([z, df.age.values / 10.0, df.stage.values, df.squamous.values])
    # early episode [0, min(T, tc)], late episode [tc, T] for those at risk after tc
    t1 = np.minimum(df.time.values, tc); e1 = np.where(df.time.values <= tc, df.event.values, 0)
    late = df.time.values > tc
    t2 = df.time.values[late]; e2 = df.event.values[late]
    X = np.vstack([np.column_stack([base, np.zeros(len(df))]), np.column_stack([base[late], np.ones(late.sum())])])
    X[len(df):, 4] = 1.0
    Xi = np.column_stack([X[:, :4], X[:, 4], X[:, 0] * X[:, 4]])   # base covariates, late indicator, program x late
    time = np.concatenate([t1, t2]); status = np.concatenate([e1, e2]); entry = np.concatenate([np.zeros(len(df)), np.full(late.sum(), tc)])
    fit = PHReg(time, Xi, status=status, entry=entry).fit()
    out['programs'][pn] = {'interaction_program_x_late_beta': float(fit.params[5]), 'interaction_p': float(fit.pvalues[5]), 'HR_early_per_SD': float(np.exp(fit.params[0])), 'HR_late_per_SD': float(np.exp(fit.params[0] + fit.params[5]))}
(RES / 'tcga_ph_check.json').write_text(json.dumps(out, indent=2))
print(out['n_patients'], out['events'], 'split day', round(tc))
for pn, v in out['programs'].items(): print(pn, 'early HR %.2f late HR %.2f interaction p=%.3f' % (v['HR_early_per_SD'], v['HR_late_per_SD'], v['interaction_p']))
