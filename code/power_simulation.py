import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parents[1] / 'results' / 'power_simulation.json'
G, CELLS, SB, REPS = 55, 300, 0.5, 20


def make(n_pat, delta, rng):
    X, y, g = [], [], []
    eff = np.zeros(G); eff[:10] = delta
    for p in range(n_pat):
        for t in (0, 1):
            off = rng.normal(0, SB, G)
            X.append(rng.normal(0, 1, (CELLS, G)) + off + t * eff); y += [t] * CELLS; g += [p] * CELLS
    return np.vstack(X), np.array(y), np.array(g)


def lopo(X, y, g):
    a = []
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=0.1)).fit(X[tr], y[tr]); a.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    return float(np.mean(a))


rng = np.random.default_rng(7); res = []
for delta in [0.0, 0.15, 0.3, 0.5, 1.0]:
    for n in [3, 5, 10, 20]:
        a = np.array([lopo(*make(n, delta, rng)) for _ in range(REPS)])
        res.append({'effect_sd_on_10_genes': delta, 'patients': n, 'lopo_auc_mean': float(a.mean()), 'lopo_auc_sd': float(a.std()), 'fraction_auc_above_0.6': float((a > 0.6).mean())})
        print(res[-1], flush=True)
OUT.write_text(json.dumps({'params': {'genes': G, 'cells_per_sample': CELLS, 'sample_offset_sd': SB, 'reps': REPS}, 'results': res}, indent=2))
