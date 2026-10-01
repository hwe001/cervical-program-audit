"""GSE173682: 11 patients with endometrial (n=5) or ovarian (n=6) cancers, single-nucleus RNA (10x multiome RNA files only).
(A) real patient-level label (tumour site: endometrium vs ovary); (B) arbitrary labelings. Same 55-gene panel, four model families, 2,000 nuclei per patient."""
import gzip, json, os
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
DATA = Path(os.environ.get('CERVIX_EXTRA_DATA', REPO / 'work' / 'additional_cohorts')) / 'GSE173682'
RES = REPO / 'results'
N_CELLS = 2000
# patient -> (GSM, file stem, tumour site); from the GEO sample records
samples = {'P1': ('GSM5276933', '3533EL', 0), 'P2': ('GSM5276934', '3571DL', 0), 'P3': ('GSM5276935', '36186L', 0), 'P4': ('GSM5276936', '36639L', 0), 'P5': ('GSM5276937', '366C5L', 0),
           'P6': ('GSM5276938', '37EACL', 1), 'P7': ('GSM5276939', '38FE7L', 1), 'P8': ('GSM5276940', '3BAE2L', 1), 'P10': ('GSM5276941', '3CCF1L', 1), 'P11': ('GSM5276942', '3E4D1L', 1), 'P9': ('GSM5276943', '3E5CFL', 1)}
panel = set('ACKR2 TGFB1 TGFB2 TGFB3 TGFBR1 TGFBR2 MDM2 CDKN1A GADD45A DDB2 BBC3 CD8A CD8B CDKN2A TOX TIGIT LAG3 HIF1A EPAS1 VEGFA CA9 SLC2A1 LDHA COL1A1 COL1A2 DCN LUM FAP PDGFRA EPCAM KRT8 KRT18 KRT19 KRT14 KRT17 PAX8 CD3D CD3E TRBC1 NKG7 GNLY MS4A1 CD79A LST1 TYROBP FCER1G LILRB1 CTSS AIF1 PECAM1 VWF EMCN KDR KIT TPSAB1'.split())
rng0 = np.random.default_rng(0)


def load(gsm, stem):
    m = mmread(DATA / f'{gsm}_matrix-{stem}.mtx.gz').tocsr()
    with gzip.open(DATA / f'{gsm}_features-{stem}.tsv.gz', 'rt', encoding='utf-8') as f:
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
def auc(mk, a, b, c, d): return roc_auc_score(d, mk().fit(a, b).predict_proba(c)[:, 1])
def rand_cv(mk, X, y): return float(np.mean([auc(mk, X[tr], y[tr], X[te], y[te]) for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y)]))


parts, pat = [], []
for p, (gsm, stem, _) in samples.items():
    x, names = load(gsm, stem); parts.append(x); pat += [p] * len(x)
X = np.vstack(parts); pat = np.array(pat); pts = list(samples)
out = {'n_patients': 11, 'n_nuclei_used': int(len(X)), 'cells_per_patient_cap': N_CELLS, 'panel_genes_present': len(names)}
lab = {p: samples[p][2] for p in pts}; y = np.array([lab[p] for p in pat]); zeros = [p for p in pts if lab[p] == 0]; ones = [p for p in pts if lab[p] == 1]
pairs = [(a, b) for a in zeros for b in ones]; A = {}
for m, mk in MODELS.items():
    f = []
    for a, b in pairs:
        te = np.isin(pat, [a, b]); f.append(auc(mk, X[~te], y[~te], X[te], y[te]))
    A[m] = {'random_cell': rand_cv(mk, X, y), 'patient_holdout_mean': float(np.mean(f)), 'patient_holdout_range': [float(min(f)), float(max(f))], 'n_pairs': len(pairs)}
    print('tumour site', m, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in A[m].items()}, flush=True)
out['real_label_tumour_site_endometrium_vs_ovary'] = A
r = np.random.default_rng(5); labs = set()
while len(labs) < 8: labs.add(tuple(sorted(r.choice(11, size=5, replace=False))))
prng = np.random.default_rng(1); B = {m: {'random_cell': [], 'patient_holdout': []} for m in MODELS}
for c in sorted(labs):
    Aset = [pts[i] for i in c]; Bset = [p for p in pts if p not in Aset]; ya = np.isin(pat, Aset).astype(int)
    pp = [(a, b) for a in Aset for b in Bset]; chosen = [pp[i] for i in prng.choice(len(pp), size=3, replace=False)]
    for m, mk in MODELS.items():
        B[m]['random_cell'].append(rand_cv(mk, X, ya)); f = []
        for a, b in chosen:
            te = np.isin(pat, [a, b]); f.append(auc(mk, X[~te], ya[~te], X[te], ya[te]))
        B[m]['patient_holdout'].append(float(np.mean(f)))
out['arbitrary_labels'] = {m: {'random_cell_mean': float(np.mean(v['random_cell'])), 'random_cell_range': [float(min(v['random_cell'])), float(max(v['random_cell']))],
                               'patient_holdout_mean': float(np.mean(v['patient_holdout'])), 'patient_holdout_range': [float(min(v['patient_holdout'])), float(max(v['patient_holdout']))]} for m, v in B.items()}
print('arbitrary', {m: (round(v['random_cell_mean'], 2), round(v['patient_holdout_mean'], 2)) for m, v in out['arbitrary_labels'].items()}, flush=True)
(RES / 'gse173682_benchmark.json').write_text(json.dumps(out, indent=2)); print('done')
