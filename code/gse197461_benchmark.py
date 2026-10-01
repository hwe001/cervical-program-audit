"""GSE197461: 8 cervical-cancer patients (3 squamous, 5 adenocarcinoma; 6 HPV-positive, 2 HPV-negative), 10x 5' scRNA-seq.
(1) arbitrary 4-vs-4 patient labelings; (2) real patient-level labels (histology, HPV status). Same 55-gene panel, four model families, 2,000 cells per sample.
Patient hold-out: one patient per class is held out together and AUC is computed on the held-out cells."""
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
DATA = Path(os.environ.get('CERVIX_EXTRA_DATA', REPO / 'work' / 'additional_cohorts')) / 'GSE197461'
RES = REPO / 'results'
N_CELLS = 2000
panel = set('ACKR2 TGFB1 TGFB2 TGFB3 TGFBR1 TGFBR2 MDM2 CDKN1A GADD45A DDB2 BBC3 CD8A CD8B CDKN2A TOX TIGIT LAG3 HIF1A EPAS1 VEGFA CA9 SLC2A1 LDHA COL1A1 COL1A2 DCN LUM FAP PDGFRA EPCAM KRT8 KRT18 KRT19 KRT14 KRT17 PAX8 CD3D CD3E TRBC1 NKG7 GNLY MS4A1 CD79A LST1 TYROBP FCER1G LILRB1 CTSS AIF1 PECAM1 VWF EMCN KDR KIT TPSAB1'.split())
rng0 = np.random.default_rng(0)
pts = ['SCC_1', 'SCC_2', 'SCC_3', 'ADC_1', 'ADC_2', 'ADC_3', 'ADC_4', 'ADC_5']
histology = {p: (1 if p.startswith('ADC') else 0) for p in pts}             # ADC = 1
hpv = {p: (0 if p in ('ADC_4', 'ADC_5') else 1) for p in pts}               # HPV-positive = 1


def load(stem):
    m = mmread(next(DATA.glob(f'*_{stem}_matrix.mtx.gz'))).tocsr()
    with gzip.open(next(DATA.glob(f'*_{stem}_features.tsv.gz')), 'rt', encoding='utf-8') as f:
        genes = [(x.rstrip('\n').split('\t') + [''])[1] for x in f]
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


parts, pat, genes_used = [], [], None
for p in pts:
    x, names = load(p); parts.append(x); pat += [p] * len(x); genes_used = names
X = np.vstack(parts); pat = np.array(pat)
out = {'n_patients': 8, 'n_cells': int(len(X)), 'panel_genes_present': len(genes_used), 'cells_per_sample': N_CELLS}


def run(label_of, pairs):
    y = np.array([label_of[p] for p in pat]); res = {}
    for m, mk in MODELS.items():
        r = rand_cv(mk, X, y); f = []
        for a, b in pairs:
            te = np.isin(pat, [a, b]); f.append(auc(mk, X[~te], y[~te], X[te], y[te]))
        res[m] = {'random_cell': r, 'patient_holdout_mean': float(np.mean(f)), 'patient_holdout_range': [float(min(f)), float(max(f))], 'n_holdout_pairs': len(pairs)}
    return res


for name, lab in (('histology_SCC_vs_ADC', histology), ('hpv_positive_vs_negative', hpv)):
    zeros = [p for p in pts if lab[p] == 0]; ones = [p for p in pts if lab[p] == 1]
    out[name] = run(lab, [(a, b) for a in zeros for b in ones]); print(name, {m: (round(v['random_cell'], 2), round(v['patient_holdout_mean'], 2)) for m, v in out[name].items()}, flush=True)
# arbitrary 4-vs-4 labelings
r = np.random.default_rng(5); labs = set()
while len(labs) < 8:
    c = tuple(sorted(r.choice(8, size=4, replace=False)))
    if 0 in c: labs.add(c)
prng = np.random.default_rng(1); arb = {m: {'random_cell': [], 'patient_holdout': []} for m in MODELS}
for c in sorted(labs):
    A = [pts[i] for i in c]; B = [p for p in pts if p not in A]; lab = {p: int(p in A) for p in pts}; y = np.array([lab[p] for p in pat])
    pairs = [(a, b) for a in A for b in B]; chosen = [pairs[i] for i in prng.choice(len(pairs), size=3, replace=False)]
    for m, mk in MODELS.items():
        arb[m]['random_cell'].append(rand_cv(mk, X, y)); f = []
        for a, b in chosen:
            te = np.isin(pat, [a, b]); f.append(auc(mk, X[~te], y[~te], X[te], y[te]))
        arb[m]['patient_holdout'].append(float(np.mean(f)))
    print('arbitrary', c, {m: (round(arb[m]['random_cell'][-1], 2), round(arb[m]['patient_holdout'][-1], 2)) for m in MODELS}, flush=True)
out['arbitrary_labels'] = {m: {'random_cell_mean': float(np.mean(v['random_cell'])), 'random_cell_range': [float(min(v['random_cell'])), float(max(v['random_cell']))],
                               'patient_holdout_mean': float(np.mean(v['patient_holdout'])), 'patient_holdout_range': [float(min(v['patient_holdout'])), float(max(v['patient_holdout']))]} for m, v in arb.items()}
(RES / 'gse197461_benchmark.json').write_text(json.dumps(out, indent=2)); print('done')
