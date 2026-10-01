"""GSE300897: nine treatment-naive high-grade serous ovarian cancer patients (4 chemo-refractory, 5 chemo-sensitive), 19,156 cells with author cell-type annotation.
(A) real patient-level label (refractory vs sensitive): random cell-level vs patient hold-out validation, four model families;
(B) arbitrary 4-vs-5 labelings; (C) the five resistance programs at patient level (pseudobulk, exact permutation over the 126 labelings; all cells and cancer cells only)."""
import gzip, itertools, json, os
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('CERVIX_EXTRA_DATA', REPO / 'work' / 'additional_cohorts')) / 'GSE300897'
RES = REPO / 'results'
N_CELLS = 2000
PROGRAMS = {'ACKR2_TGFb': ['ACKR2', 'TGFB1', 'TGFB2', 'TGFB3', 'TGFBR1', 'TGFBR2'], 'MDM2_DDR': ['MDM2', 'CDKN1A', 'GADD45A', 'DDB2', 'BBC3'],
            'Tcell_senescence': ['CD8A', 'CD8B', 'CDKN2A', 'TOX', 'TIGIT', 'LAG3'], 'Hypoxia': ['HIF1A', 'EPAS1', 'VEGFA', 'CA9', 'SLC2A1', 'LDHA'],
            'CAF_ECM': ['COL1A1', 'COL1A2', 'DCN', 'LUM', 'FAP', 'PDGFRA']}
panel = set('ACKR2 TGFB1 TGFB2 TGFB3 TGFBR1 TGFBR2 MDM2 CDKN1A GADD45A DDB2 BBC3 CD8A CD8B CDKN2A TOX TIGIT LAG3 HIF1A EPAS1 VEGFA CA9 SLC2A1 LDHA COL1A1 COL1A2 DCN LUM FAP PDGFRA EPCAM KRT8 KRT18 KRT19 KRT14 KRT17 PAX8 CD3D CD3E TRBC1 NKG7 GNLY MS4A1 CD79A LST1 TYROBP FCER1G LILRB1 CTSS AIF1 PECAM1 VWF EMCN KDR KIT TPSAB1'.split())
ann = pd.read_csv(DATA / 'GSE300897_annotation_HGSC.tsv.gz', sep='\t')
cells = ann.cell_name.tolist(); expr = {}
with gzip.open(DATA / 'GSE300897_UMIcounts_HGSC.tsv.gz', 'rt') as f:
    header = f.readline().rstrip('\n').split('\t')[1:]
    assert header == cells, 'cell order differs from annotation'
    for line in f:
        g, rest = line.split('\t', 1)
        if g.upper() in panel and g.upper() not in expr: expr[g.upper()] = np.array(rest.rstrip('\n').split('\t'), dtype=float)
genes = sorted(expr); print('panel genes found', len(genes), 'of', len(panel))
lib = ann.nCount_RNA.values.astype(float)
M = np.column_stack([np.log1p(expr[g] / lib * 1e4) for g in genes])      # cells x genes
pat = ann.patient_id.values; status = ann.status.values; subtype = ann.cell_subtype.values
pts = sorted(set(pat)); label = {p: int(ann.loc[ann.patient_id == p, 'status'].iloc[0] == 'Refractory') for p in pts}
print({p: (label[p], int((pat == p).sum())) for p in pts})
rng0 = np.random.default_rng(0); keep = np.concatenate([rng0.choice(np.where(pat == p)[0], size=min(N_CELLS, (pat == p).sum()), replace=False) for p in pts])
X = M[keep]; P = pat[keep]
MODELS = {
    'logistic_regression': lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, C=0.1)),
    'random_forest': lambda: RandomForestClassifier(n_estimators=100, min_samples_leaf=10, n_jobs=-1, random_state=0),
    'gradient_boosting': lambda: HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=0),
    'neural_network': lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-2, max_iter=60, early_stopping=True, random_state=0)),
}
def auc(mk, a, b, c, d): return roc_auc_score(d, mk().fit(a, b).predict_proba(c)[:, 1])
def rand_cv(mk, X, y): return float(np.mean([auc(mk, X[tr], y[tr], X[te], y[te]) for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y)]))
out = {'n_patients': 9, 'n_cells_total': int(len(ann)), 'n_cells_used_in_models': int(len(X)), 'panel_genes_found': len(genes), 'cells_per_patient_cap': N_CELLS}
# A: real label
y = np.array([label[p] for p in P]); ref = [p for p in pts if label[p] == 1]; sen = [p for p in pts if label[p] == 0]; pairs = [(a, b) for a in ref for b in sen]; A = {}
for m, mk in MODELS.items():
    f = []
    for a, b in pairs:
        te = np.isin(P, [a, b]); f.append(auc(mk, X[~te], y[~te], X[te], y[te]))
    A[m] = {'random_cell': rand_cv(mk, X, y), 'patient_holdout_mean': float(np.mean(f)), 'patient_holdout_range': [float(min(f)), float(max(f))], 'n_pairs': len(pairs)}
    print('real label', m, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in A[m].items()}, flush=True)
out['real_label_refractory_vs_sensitive'] = A
# B: arbitrary labelings
r = np.random.default_rng(5); labs = set()
while len(labs) < 8: labs.add(tuple(sorted(r.choice(9, size=4, replace=False))))
prng = np.random.default_rng(1); B = {m: {'random_cell': [], 'patient_holdout': []} for m in MODELS}
for c in sorted(labs):
    Aset = [pts[i] for i in c]; Bset = [p for p in pts if p not in Aset]; ya = np.isin(P, Aset).astype(int)
    pp = [(a, b) for a in Aset for b in Bset]; chosen = [pp[i] for i in prng.choice(len(pp), size=3, replace=False)]
    for m, mk in MODELS.items():
        B[m]['random_cell'].append(rand_cv(mk, X, ya)); f = []
        for a, b in chosen:
            te = np.isin(P, [a, b]); f.append(auc(mk, X[~te], ya[~te], X[te], ya[te]))
        B[m]['patient_holdout'].append(float(np.mean(f)))
out['arbitrary_labels'] = {m: {'random_cell_mean': float(np.mean(v['random_cell'])), 'random_cell_range': [float(min(v['random_cell'])), float(max(v['random_cell']))],
                               'patient_holdout_mean': float(np.mean(v['patient_holdout'])), 'patient_holdout_range': [float(min(v['patient_holdout'])), float(max(v['patient_holdout']))]} for m, v in B.items()}
print('arbitrary', {m: (round(v['random_cell_mean'], 2), round(v['patient_holdout_mean'], 2)) for m, v in out['arbitrary_labels'].items()}, flush=True)
# C: programs at patient level
def program_test(mask, tag):
    ok = [p for p in pts if (mask & (pat == p)).sum() >= 50]
    if len(ok) < 6 or sum(label[p] for p in ok) < 2 or sum(1 - label[p] for p in ok) < 2: return {'patients_used': ok, 'note': 'too few patients with >=50 cells'}
    pb = pd.DataFrame({g: [M[mask & (pat == p), j].mean() for p in ok] for j, g in enumerate(genes)}, index=ok)
    z = (pb - pb.mean()) / pb.std(ddof=1); yy = np.array([label[p] for p in ok]); res = {}
    for name, gs in PROGRAMS.items():
        sc = z[[g for g in gs if g in z.columns]].mean(axis=1).values; obs = sc[yy == 1].mean() - sc[yy == 0].mean(); k = int(yy.sum()); null = []
        for comb in itertools.combinations(range(len(ok)), k):
            m_ = np.zeros(len(ok), bool); m_[list(comb)] = True; null.append(sc[m_].mean() - sc[~m_].mean())
        res[name] = {'refractory_minus_sensitive': float(obs), 'exact_permutation_p': float((np.abs(null) >= abs(obs) - 1e-12).mean()), 'n_permutations': len(null), 'genes_present': [g for g in gs if g in z.columns]}
    for n_, q in zip(res, multipletests([res[n]['exact_permutation_p'] for n in res], method='fdr_bh')[1]): res[n_]['BH_q'] = float(q)
    return {'patients_used': ok, 'n_refractory': int(yy.sum()), 'n_sensitive': int((1 - yy).sum()), 'programs': res}
out['programs_all_cells'] = program_test(np.ones(len(ann), bool), 'all')
out['programs_cancer_cells_only'] = program_test(subtype == 'Ovarian.cancer.cell', 'cancer')
for k in ('programs_all_cells', 'programs_cancer_cells_only'):
    print(k, {n: (round(v['refractory_minus_sensitive'], 2), round(v['exact_permutation_p'], 3)) for n, v in out[k].get('programs', {}).items()} or out[k].get('note'), flush=True)
(RES / 'gse300897_benchmark.json').write_text(json.dumps(out, indent=2)); print('done')
