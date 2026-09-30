import gzip, json
from pathlib import Path
import numpy as np
from scipy.io import mmread
from scipy.sparse import vstack
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

ROOT = Path(r'C:\Users\pc\Documents\Codex\2026-09-23\for\work'); MAT = ROOT / 'GSE236738_matrix'; OUT = ROOT / 'analysis_gse236738'
sample_info = {'GSM7574777_FXW': ('P1', 'TN'), 'GSM7574778_FXW2': ('P1', 'CCRT'), 'GSM7574779_SBJ': ('P2', 'TN'), 'GSM7574780_SBJ2': ('P2', 'CCRT'), 'GSM7574781_WGX': ('P3', 'TN'), 'GSM7574782_WGX2': ('P3', 'CCRT')}
genes_keep = set('ACKR2 TGFB1 TGFB2 TGFB3 TGFBR1 TGFBR2 MDM2 CDKN1A GADD45A DDB2 BBC3 CD8A CD8B CDKN2A TOX TIGIT LAG3 HIF1A EPAS1 VEGFA CA9 SLC2A1 LDHA COL1A1 COL1A2 DCN LUM FAP PDGFRA EPCAM KRT8 KRT18 KRT19 KRT14 KRT17 PAX8 CD3D CD3E TRBC1 NKG7 GNLY MS4A1 CD79A LST1 TYROBP FCER1G LILRB1 CTSS AIF1 PECAM1 VWF EMCN KDR KIT TPSAB1'.split())
parts = []; y = []; groups = []; samples = []
for sample, (patient, trt) in sample_info.items():
    m = mmread(MAT / f'{sample}.matrix.mtx.gz').tocsr()
    with gzip.open(MAT / f'{sample}.features.tsv.gz', 'rt', encoding='utf-8') as f:
        genes = [x.rstrip('\n').split('\t')[1] for x in f]
    idx = {g.upper(): i for i, g in enumerate(genes)}
    names = sorted(g for g in genes_keep if g in idx)
    x = m[[idx[g] for g in names], :].tocsr()
    lib = np.asarray(m.sum(axis=0)).ravel()
    x = x.multiply(1e4 / np.maximum(lib, 1)).tocsr(); x.data = np.log1p(x.data)
    parts.append(x.T); y += [1 if trt == 'CCRT' else 0] * x.shape[1]; groups += [patient] * x.shape[1]; samples += [sample] * x.shape[1]
X = vstack(parts).tocsr(); y = np.array(y); groups = np.array(groups); samples = np.array(samples)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
import itertools, pandas as pd
RES = Path(__file__).resolve().parents[1] / 'results'
def mdl(): return make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=300, C=0.1))
def auc_fit(Xtr, ytr, Xte, yte): return roc_auc_score(yte, mdl().fit(Xtr, ytr).predict_proba(Xte)[:, 1])
out = {'model': 'logistic regression (L2, C=0.1) on the 55-gene panel', 'n_cells': int(X.shape[0]), 'n_samples': 6, 'n_patients': 3}
def rand_cv(y_):
    return float(np.mean([auc_fit(X[tr], y_[tr], X[te], y_[te]) for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y_)]))
out['random_cell_5fold'] = rand_cv(y)
# hold out one sample per patient (test has both classes); the other sample of each patient stays in training
pats = ['P1', 'P2', 'P3']; sp = []
for hold in itertools.product([0, 1], repeat=3):   # label (0=TN,1=CCRT) of the held-out sample for each patient
    if len(set(hold)) < 2: continue
    te = np.zeros(len(y), bool)
    for p, h in zip(pats, hold): te |= (groups == p) & (y == h)
    sp.append(float(auc_fit(X[~te], y[~te], X[te], y[te])))
out['sample_split_one_sample_per_patient'] = {'splits': len(sp), 'auc_mean': float(np.mean(sp)), 'auc_min': float(min(sp)), 'auc_max': float(max(sp))}
lo = {}
for tr, te in LeaveOneGroupOut().split(X, y, groups): lo[groups[te][0]] = float(auc_fit(X[tr], y[tr], X[te], y[te]))
out['leave_one_patient_out'] = {'folds': lo, 'auc_mean': float(np.mean(list(lo.values())))}
# pseudobulk: one row per sample
sam = sorted(set(samples)); PB = np.vstack([np.asarray(X[samples == s_].mean(axis=0)).ravel() for s_ in sam]); ypb = np.array([y[samples == s_][0] for s_ in sam]); gpb = np.array([groups[samples == s_][0] for s_ in sam])
pbf = {}
for tr, te in LeaveOneGroupOut().split(PB, ypb, gpb):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=0.1)).fit(PB[tr], ypb[tr]); pbf[gpb[te][0]] = float(roc_auc_score(ypb[te], m.predict_proba(PB[te])[:, 1]))
out['pseudobulk_leave_one_patient_out'] = {'folds': pbf, 'auc_mean': float(np.mean(list(pbf.values()))), 'note': 'each held-out patient contributes two samples, so each fold AUC is 0, 0.5 or 1'}
# label-flip controls: flipping all labels of selected patients; random cell CV vs leave-one-patient-out
flips = []
for flip in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]:
    yp = y.copy()
    for k, p in enumerate(pats):
        if flip[k]: yp[groups == p] = 1 - yp[groups == p]
    l = float(np.mean([auc_fit(X[tr], yp[tr], X[te], yp[te]) for tr, te in LeaveOneGroupOut().split(X, yp, groups)]))
    flips.append({'flipped': [p for k, p in enumerate(pats) if flip[k]], 'random_cell_5fold': rand_cv(yp), 'leave_one_patient_out': l})
out['label_flip_controls'] = flips
out['label_flip_summary'] = {'random_cell_5fold_range': [min(f['random_cell_5fold'] for f in flips), max(f['random_cell_5fold'] for f in flips)], 'lopo_range': [min(f['leave_one_patient_out'] for f in flips), max(f['leave_one_patient_out'] for f in flips)]}
# effective sample size of program scores (design effect from within-sample correlation)
ca = pd.read_csv(ROOT / 'analysis_gse236738' / 'cell_annotations.csv'); progs = ['ACKR2_TGFb', 'MDM2_DDR', 'Tcell_senescence', 'Hypoxia', 'CAF_ECM']
ess = {}
for p in progs:
    g_ = ca.groupby('sample')[p]; k = g_.ngroups; N = len(ca); m_ = N / k
    msb = (g_.count() * (g_.mean() - ca[p].mean()) ** 2).sum() / (k - 1); msw = ((ca[p] - g_.transform('mean')) ** 2).sum() / (N - k)
    icc = float((msb - msw) / (msb + (m_ - 1) * msw)); deff = 1 + (m_ - 1) * icc
    ess[p] = {'icc_by_sample': icc, 'design_effect': float(deff), 'effective_n_cells': float(N / deff)}
out['effective_sample_size'] = {'cells': int(len(ca)), 'samples': 6, 'patients': 3, 'cells_per_sample_mean': float(len(ca) / 6), 'programs': ess}
(RES / 'validation_strategies.json').write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=2))
