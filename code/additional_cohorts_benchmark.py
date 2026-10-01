"""Additional public single-cell cohorts for the leakage demonstration (same 55-gene panel and four model families as model_family_benchmark.py).

GSE228499 (9 breast-cancer patients) and GSE292163 (10 SiNET patients; scRNA-seq and snRNA-seq mixed): arbitrary patient-level labels, so any apparent
performance is leakage or chance. GSE237425 (4 renal-carcinoma patients, tumour and adjacent normal): real labels, paired design.
Cells are subsampled to 2,000 per sample (seed 0).
"""
import gzip, itertools, json, os
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

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('CERVIX_EXTRA_DATA', REPO / 'work' / 'additional_cohorts'))
RES = REPO / 'results'
N_CELLS, N_LABELINGS, N_PAIRS = 2000, 8, 3
panel = set('ACKR2 TGFB1 TGFB2 TGFB3 TGFBR1 TGFBR2 MDM2 CDKN1A GADD45A DDB2 BBC3 CD8A CD8B CDKN2A TOX TIGIT LAG3 HIF1A EPAS1 VEGFA CA9 SLC2A1 LDHA COL1A1 COL1A2 DCN LUM FAP PDGFRA EPCAM KRT8 KRT18 KRT19 KRT14 KRT17 PAX8 CD3D CD3E TRBC1 NKG7 GNLY MS4A1 CD79A LST1 TYROBP FCER1G LILRB1 CTSS AIF1 PECAM1 VWF EMCN KDR KIT TPSAB1'.split())
rng0 = np.random.default_rng(0)


def load(folder, stem):
    mtx = next((folder).glob(f'*_{stem}_matrix.mtx.gz')); feat = next(folder.glob(f'*_{stem}_features.tsv.gz'))
    m = mmread(mtx).tocsr()
    with gzip.open(feat, 'rt', encoding='utf-8') as f:
        rows = [x.rstrip('\n').split('\t') for x in f]
    genes = [r[1] if len(r) > 1 else r[0] for r in rows]
    idx = {g.upper(): i for i, g in enumerate(genes)}; names = sorted(g for g in panel if g in idx)
    lib = np.asarray(m.sum(axis=0)).ravel(); x = m[[idx[g] for g in names], :].tocsr()
    x = x.multiply(1e4 / np.maximum(lib, 1)).tocsr(); x.data = np.log1p(x.data)
    x = x.T.toarray(); keep = rng0.choice(x.shape[0], size=min(N_CELLS, x.shape[0]), replace=False)
    return x[keep], names


MODELS = {
    'logistic_regression': lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, C=0.1)),
    'random_forest': lambda: RandomForestClassifier(n_estimators=100, min_samples_leaf=10, n_jobs=-1, random_state=0),
    'gradient_boosting': lambda: HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=0),
    'neural_network': lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-2, max_iter=60, early_stopping=True, random_state=0)),
}


def auc(mk, Xtr, ytr, Xte, yte): return roc_auc_score(yte, mk().fit(Xtr, ytr).predict_proba(Xte)[:, 1])


def rand_cv(mk, X, y): return float(np.mean([auc(mk, X[tr], y[tr], X[te], y[te]) for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y)]))


out = {'settings': {'cells_per_sample': N_CELLS, 'labelings': N_LABELINGS, 'holdout_pairs_per_labeling': N_PAIRS, 'panel_genes': len(panel)}}


def arbitrary(acc, folder, stems):
    parts, pat = [], []
    for s in stems:
        x, names = load(folder, s); parts.append(x); pat += [s] * len(x)
    X = np.vstack(parts); pat = np.array(pat); pts = list(stems); k = len(pts) // 2
    r = np.random.default_rng(5); labs = set()
    while len(labs) < N_LABELINGS: labs.add(tuple(sorted(r.choice(len(pts), size=k, replace=False))))
    res = {m: {'random_cell': [], 'patient_holdout': []} for m in MODELS}
    prng = np.random.default_rng(1)
    for lab in sorted(labs):
        A = [pts[i] for i in lab]; B = [p for p in pts if p not in A]; y = np.isin(pat, A).astype(int)
        pairs = [(a, b) for a in A for b in B]; chosen = [pairs[i] for i in prng.choice(len(pairs), size=N_PAIRS, replace=False)]
        for m, mk in MODELS.items():
            res[m]['random_cell'].append(rand_cv(mk, X, y))
            f = []
            for a, b in chosen:
                te = np.isin(pat, [a, b]); f.append(auc(mk, X[~te], y[~te], X[te], y[te]))
            res[m]['patient_holdout'].append(float(np.mean(f)))
        print(acc, lab, {m: (round(res[m]['random_cell'][-1], 2), round(res[m]['patient_holdout'][-1], 2)) for m in MODELS}, flush=True)
    out[acc] = {'n_patients': len(pts), 'n_cells': int(len(X)), 'models': {m: {'random_cell_mean': float(np.mean(v['random_cell'])), 'random_cell_range': [float(min(v['random_cell'])), float(max(v['random_cell']))],
                                                                              'patient_holdout_mean': float(np.mean(v['patient_holdout'])), 'patient_holdout_range': [float(min(v['patient_holdout'])), float(max(v['patient_holdout']))]} for m, v in res.items()}}


arbitrary('GSE228499', DATA / 'GSE228499', ['BC03', 'BC_05', 'BC_06', 'BC_08', 'BC_11', 'BC_12', 'BC_14', 'BC_15', 'BC_17'])
arbitrary('GSE292163', DATA / 'GSE292163', [f'sinet{i}' for i in range(1, 11)])
# real labels, paired design
folder = DATA / 'GSE237425'; pats = ['RAM5', 'RAM12', 'RAM13', 'RAM15']; parts, y, g = [], [], []
for p in pats:
    for lab, stem in ((0, f'{p}_NAT'), (1, f'{p}_Tumor')):
        x, _ = load(folder, stem); parts.append(x); y += [lab] * len(x); g += [p] * len(x)
X = np.vstack(parts); y = np.array(y); g = np.array(g); r3 = {}
for m, mk in MODELS.items():
    lopo = {p: auc(mk, X[g != p], y[g != p], X[g == p], y[g == p]) for p in pats}
    yf = y.copy(); yf[g == 'RAM5'] = 1 - yf[g == 'RAM5']
    lopo_f = float(np.mean([auc(mk, X[g != p], yf[g != p], X[g == p], yf[g == p]) for p in pats if p != 'RAM5']))
    r3[m] = {'random_cell': rand_cv(mk, X, y), 'lopo_per_patient': {p: float(v) for p, v in lopo.items()}, 'lopo_mean': float(np.mean(list(lopo.values()))),
             'random_cell_label_flipped_RAM5': rand_cv(mk, X, yf), 'lopo_label_flipped_RAM5_other_patients': lopo_f}
    print('GSE237425', m, {a: (round(b, 2) if isinstance(b, float) else b) for a, b in r3[m].items()}, flush=True)
out['GSE237425_real_labels'] = {'n_patients': 4, 'n_cells': int(len(X)), 'models': r3}
(RES / 'additional_cohorts_benchmark.json').write_text(json.dumps(out, indent=2)); print('done')
