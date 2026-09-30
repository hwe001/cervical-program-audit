import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = (Path(__file__).resolve().parents[1] / 'results' / 'leakage_simulation.json')
G, CELLS, SB, REPS = 55, 300, 0.5, 20


def make(n_pat, delta, rng):
    X, y, g = [], [], []
    eff = np.zeros(G); eff[:10] = delta
    for p in range(n_pat):
        for t in (0, 1):
            off = rng.normal(0, SB, G)  # sample-specific (batch/library) offset
            x = rng.normal(0, 1, (CELLS, G)) + off + t * eff
            X.append(x); y += [t] * CELLS; g += [p] * CELLS
    return np.vstack(X), np.array(y), np.array(g)


def mdl(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=0.1))


def cell_cv(X, y, rng):
    a = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=int(rng.integers(1e9))).split(X, y):
        m = mdl().fit(X[tr], y[tr]); a.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    return float(np.mean(a))


def lopo(X, y, g):
    a = []
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        m = mdl().fit(X[tr], y[tr]); a.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    return float(np.mean(a))


rng = np.random.default_rng(1); res = []
for delta, label in [(0.0, 'no treatment effect'), (0.15, 'small true effect')]:
    for n in [3, 5, 10, 20]:
        c, l = [], []
        for _ in range(REPS):
            X, y, g = make(n, delta, rng); c.append(cell_cv(X, y, rng)); l.append(lopo(X, y, g))
        res.append({'scenario': label, 'patients': n, 'cell_level_cv_auc_mean': float(np.mean(c)), 'cell_level_cv_auc_sd': float(np.std(c)),
                    'leave_one_patient_out_auc_mean': float(np.mean(l)), 'leave_one_patient_out_auc_sd': float(np.std(l))})
        print(res[-1], flush=True)
OUT.write_text(json.dumps({'params': {'genes': G, 'cells_per_sample': CELLS, 'sample_offset_sd': SB, 'reps': REPS, 'model': 'logistic regression'}, 'results': res}, indent=2))
