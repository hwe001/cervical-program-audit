"""Model-family benchmark: does cell-level inflation depend on the model?

Models: L2 logistic regression, random forest, histogram gradient boosting, small neural network (one hidden layer).
Cohorts: GSE224327 (6 patients, arbitrary labels; every balanced 3-vs-3 labeling) and GSE236738 (true pre/post labels, plus a label-flip control).
Cells are subsampled to 3,000 per patient or sample (seed 0) for speed.
"""
import gzip, itertools, json
from pathlib import Path
import numpy as np
from scipy.io import mmread
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1] / 'work'
RES = Path(__file__).resolve().parents[1] / 'results'
panel = set('ACKR2 TGFB1 TGFB2 TGFB3 TGFBR1 TGFBR2 MDM2 CDKN1A GADD45A DDB2 BBC3 CD8A CD8B CDKN2A TOX TIGIT LAG3 HIF1A EPAS1 VEGFA CA9 SLC2A1 LDHA COL1A1 COL1A2 DCN LUM FAP PDGFRA EPCAM KRT8 KRT18 KRT19 KRT14 KRT17 PAX8 CD3D CD3E TRBC1 NKG7 GNLY MS4A1 CD79A LST1 TYROBP FCER1G LILRB1 CTSS AIF1 PECAM1 VWF EMCN KDR KIT TPSAB1'.split())
rng0 = np.random.default_rng(0)


def load(mtx, feat, n=3000):
    m = mmread(mtx).tocsr()
    with gzip.open(feat, 'rt', encoding='utf-8') as f:
        genes = [x.rstrip('\n').split('\t')[1] for x in f]
    idx = {g.upper(): i for i, g in enumerate(genes)}; names = sorted(g for g in panel if g in idx)
    lib = np.asarray(m.sum(axis=0)).ravel(); x = m[[idx[g] for g in names], :].tocsr()
    x = x.multiply(1e4 / np.maximum(lib, 1)).tocsr(); x.data = np.log1p(x.data)
    x = x.T.toarray(); keep = rng0.choice(x.shape[0], size=min(n, x.shape[0]), replace=False)
    return x[keep], names


MODELS = {
    'logistic_regression': lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, C=0.1)),
    'random_forest': lambda: RandomForestClassifier(n_estimators=100, min_samples_leaf=10, n_jobs=-1, random_state=0),
    'gradient_boosting': lambda: HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=0),
    'neural_network': lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-2, max_iter=60, early_stopping=True, random_state=0)),
}


def auc(mk, Xtr, ytr, Xte, yte): return roc_auc_score(yte, mk().fit(Xtr, ytr).predict_proba(Xte)[:, 1])


def rand_cv(mk, X, y): return float(np.mean([auc(mk, X[tr], y[tr], X[te], y[te]) for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y)]))


out = {}
# ---- GSE224327 arbitrary labels
D = ROOT / 'data_gse224327' / 'extracted'
samples = {'PT1': 'GSM7019487', 'PT2': 'GSM7019488', 'PT3': 'GSM7019489', 'PT4': 'GSM7019490', 'PT5': 'GSM7019491', 'PT6': 'GSM7019492'}
parts, pat = [], []
for p, g in samples.items():
    x, _ = load(D / f'{g}_{p}_matrix.mtx.gz', D / f'{g}_{p}_features.tsv.gz'); parts.append(x); pat += [p] * len(x)
X = np.vstack(parts); pat = np.array(pat); pts = list(samples)
labelings = [a for a in itertools.combinations(pts, 3) if 'PT1' in a]
pair_rng = np.random.default_rng(1)
res = {k: {'random_cell': [], 'patient_holdout': []} for k in MODELS}
for a in labelings:
    y = np.isin(pat, a).astype(int); A = list(a); B = [q for q in pts if q not in a]
    pairs = [(ha, hb) for ha in A for hb in B]; chosen = [pairs[i] for i in pair_rng.choice(len(pairs), size=3, replace=False)]
    for k, mk in MODELS.items():
        res[k]['random_cell'].append(rand_cv(mk, X, y))
        f = []
        for ha, hb in chosen:
            te = np.isin(pat, [ha, hb]); f.append(auc(mk, X[~te], y[~te], X[te], y[te]))
        res[k]['patient_holdout'].append(float(np.mean(f)))
    print('labeling', a, {k: (round(res[k]['random_cell'][-1], 2), round(res[k]['patient_holdout'][-1], 2)) for k in MODELS}, flush=True)
out['GSE224327_arbitrary_labels'] = {k: {'random_cell_mean': float(np.mean(v['random_cell'])), 'random_cell_range': [float(min(v['random_cell'])), float(max(v['random_cell']))],
                                         'patient_holdout_mean': float(np.mean(v['patient_holdout'])), 'patient_holdout_range': [float(min(v['patient_holdout'])), float(max(v['patient_holdout']))]} for k, v in res.items()}
# ---- GSE236738 true labels
M = ROOT / 'GSE236738_matrix'
info = {'GSM7574777_FXW': ('P1', 0), 'GSM7574778_FXW2': ('P1', 1), 'GSM7574779_SBJ': ('P2', 0), 'GSM7574780_SBJ2': ('P2', 1), 'GSM7574781_WGX': ('P3', 0), 'GSM7574782_WGX2': ('P3', 1)}
parts, y, g = [], [], []
for pre, (p, t) in info.items():
    x, _ = load(M / f'{pre}.matrix.mtx.gz', M / f'{pre}.features.tsv.gz'); parts.append(x); y += [t] * len(x); g += [p] * len(x)
X = np.vstack(parts); y = np.array(y); g = np.array(g)
r2 = {}
for k, mk in MODELS.items():
    lopo = {p: auc(mk, X[g != p], y[g != p], X[g == p], y[g == p]) for p in ['P1', 'P2', 'P3']}
    yf = y.copy(); yf[g == 'P1'] = 1 - yf[g == 'P1']
    lopo_f = float(np.mean([auc(mk, X[g != p], yf[g != p], X[g == p], yf[g == p]) for p in ['P2', 'P3']]))   # flipped patient stays in training
    r2[k] = {'random_cell': rand_cv(mk, X, y), 'lopo_per_patient': {p: float(v) for p, v in lopo.items()}, 'lopo_mean': float(np.mean(list(lopo.values()))),
             'random_cell_label_flipped_P1': rand_cv(mk, X, yf), 'lopo_label_flipped_P1_held_out_P2_P3': lopo_f}
    print(k, {a: (round(b, 2) if isinstance(b, float) else b) for a, b in r2[k].items()}, flush=True)
out['GSE236738_true_labels'] = r2
out['note'] = 'cells subsampled to 3,000 per patient (GSE224327) or per sample (GSE236738); GSE224327 patient holdout uses 3 random one-patient-per-class pairs per labeling'
(RES / 'model_family_benchmark.json').write_text(json.dumps(out, indent=2))
print('done')
