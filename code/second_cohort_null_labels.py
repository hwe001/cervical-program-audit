"""Second real single-cell cohort (GSE224327: 6 patients, one sample each, ~50k cells) used only to demonstrate validation leakage.

GEO provides no sample-level response labels for this series, so every balanced 3-vs-3 assignment of patients to two arbitrary classes is a null labelling
(10 distinct labelings). Random cell-level cross-validation is compared with validation that holds out one patient per class.
"""
import gzip, itertools, json
from pathlib import Path
import numpy as np
from scipy.io import mmread
from scipy.sparse import vstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1] / 'work'
RES = Path(__file__).resolve().parents[1] / 'results'
D = ROOT / 'data_gse224327' / 'extracted'
samples = {'PT1': 'GSM7019487', 'PT2': 'GSM7019488', 'PT3': 'GSM7019489', 'PT4': 'GSM7019490', 'PT5': 'GSM7019491', 'PT6': 'GSM7019492'}
panel = set('ACKR2 TGFB1 TGFB2 TGFB3 TGFBR1 TGFBR2 MDM2 CDKN1A GADD45A DDB2 BBC3 CD8A CD8B CDKN2A TOX TIGIT LAG3 HIF1A EPAS1 VEGFA CA9 SLC2A1 LDHA COL1A1 COL1A2 DCN LUM FAP PDGFRA EPCAM KRT8 KRT18 KRT19 KRT14 KRT17 PAX8 CD3D CD3E TRBC1 NKG7 GNLY MS4A1 CD79A LST1 TYROBP FCER1G LILRB1 CTSS AIF1 PECAM1 VWF EMCN KDR KIT TPSAB1'.split())
parts, pat = [], []
for p, g in samples.items():
    m = mmread(D / f'{g}_{p}_matrix.mtx.gz').tocsr()
    with gzip.open(D / f'{g}_{p}_features.tsv.gz', 'rt', encoding='utf-8') as f:
        genes = [x.rstrip('\n').split('\t')[1] for x in f]
    idx = {x.upper(): i for i, x in enumerate(genes)}; names = sorted(x for x in panel if x in idx)
    lib = np.asarray(m.sum(axis=0)).ravel(); x = m[[idx[n] for n in names], :].tocsr()
    x = x.multiply(1e4 / np.maximum(lib, 1)).tocsr(); x.data = np.log1p(x.data)
    parts.append(x.T); pat += [p] * x.shape[1]
X = vstack(parts).tocsr(); pat = np.array(pat); pts = list(samples)


def mdl(): return make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=300, C=0.1))


res = []
for a in itertools.combinations(pts, 3):
    if 'PT1' not in a: continue          # label symmetry: 10 distinct labelings
    y = np.isin(pat, a).astype(int)
    rc = float(np.mean([roc_auc_score(y[te], mdl().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]) for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y)]))
    A = list(a); B = [p for p in pts if p not in a]; f = []
    for ha, hb in itertools.product(A, B):
        te = np.isin(pat, [ha, hb]); f.append(roc_auc_score(y[te], mdl().fit(X[~te], y[~te]).predict_proba(X[te])[:, 1]))
    res.append({'class1': A, 'random_cell_5fold_auc': rc, 'hold_out_one_patient_per_class_auc': float(np.mean(f))})
    print(res[-1], flush=True)
out = {'cohort': 'GSE224327', 'n_patients': 6, 'n_cells': int(X.shape[0]), 'labelings': res,
       'random_cell_mean': float(np.mean([r['random_cell_5fold_auc'] for r in res])), 'patient_holdout_mean': float(np.mean([r['hold_out_one_patient_per_class_auc'] for r in res])),
       'note': 'labels are arbitrary by construction; any AUC above 0.5 is leakage or chance'}
(RES / 'second_cohort_null_labels.json').write_text(json.dumps(out, indent=2))
print(out['random_cell_mean'], out['patient_holdout_mean'])
