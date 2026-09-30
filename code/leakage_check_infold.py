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

ROOT = (Path(__file__).resolve().parents[1] / 'work'); MAT = ROOT / 'GSE236738_matrix'; OUT = ROOT / 'analysis_gse236738'
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
rng = np.random.default_rng(0)

def run(X, y, groups, infold, seed=42):
    aucs = {}
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        if infold:
            sc = StandardScaler(with_mean=False).fit(X[tr]); svd = TruncatedSVD(20, random_state=seed).fit(sc.transform(X[tr]))
            Ztr = svd.transform(sc.transform(X[tr])); Zte = svd.transform(sc.transform(X[te]))
        else:
            sc = StandardScaler(with_mean=False).fit(X); svd = TruncatedSVD(20, random_state=seed).fit(sc.transform(X))
            Ztr = svd.transform(sc.transform(X[tr])); Zte = svd.transform(sc.transform(X[te]))
        clf = RandomForestClassifier(300, min_samples_leaf=10, class_weight='balanced_subsample', random_state=seed, n_jobs=-1).fit(Ztr, y[tr])
        aucs[groups[te][0]] = float(roc_auc_score(y[te], clf.predict_proba(Zte)[:, 1]))
    return aucs

res = {'original_global_fit': run(X, y, groups, False), 'infold_fit': run(X, y, groups, True)}
# negative control 1: shuffle treatment labels at SAMPLE level within patients (swap pre/post for a random subset of patients)
ctrl = []
for flip in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]:
    yp = y.copy()
    for k, p in enumerate(['P1', 'P2', 'P3']):
        if flip[k]:
            mk = groups == p; yp[mk] = 1 - yp[mk]
    a = run(X, yp, groups, True); ctrl.append({'flipped_patients': [p for k, p in enumerate(['P1', 'P2', 'P3']) if flip[k]], 'aucs': a, 'mean': float(np.mean(list(a.values())))})
res['sample_label_flip_control_infold'] = ctrl
for k in ('original_global_fit', 'infold_fit'):
    res[k + '_mean'] = float(np.mean(list(res[k].values())))
(OUT / 'leakage_check_infold.json').write_text(json.dumps(res, indent=2)); print(json.dumps(res, indent=2))
