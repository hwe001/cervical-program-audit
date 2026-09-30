"""Leakage and power simulation with correlated genes and program-level effect sizes.

55 genes: five blocks of ten genes with within-block correlation 0.5, plus five independent genes. Cell noise ~ N(0, Sigma); each sample adds an offset ~ N(0, 0.25 Sigma).
The treatment effect shifts the ten genes of block 0 by a common amount chosen so that the program score (mean of the block) moves by d cell-level SDs of that score.
"""
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parents[1] / 'results' / 'correlated_simulation.json'
G, CELLS, RHO, REPS = 55, 300, 0.5, 20
Sigma = np.eye(G)
for b in range(5):
    blk = slice(b * 10, b * 10 + 10); Sigma[blk, blk] = RHO + (1 - RHO) * np.eye(10)
L = np.linalg.cholesky(Sigma)
sd_score = float(np.sqrt(Sigma[:10, :10].mean()))          # cell-level SD of the block-mean score


def make(n_pat, d, rng):
    X, y, g = [], [], []
    shift = np.zeros(G); shift[:10] = d * sd_score
    for p in range(n_pat):
        for t in (0, 1):
            off = 0.5 * (L @ rng.normal(size=G))
            X.append(rng.normal(size=(CELLS, G)) @ L.T + off + t * shift); y += [t] * CELLS; g += [p] * CELLS
    return np.vstack(X), np.array(y), np.array(g)


def mdl(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=0.1))


def lopo(X, y, g):
    return float(np.mean([roc_auc_score(y[te], mdl().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]) for tr, te in LeaveOneGroupOut().split(X, y, g)]))


def cellcv(X, y, rng):
    return float(np.mean([roc_auc_score(y[te], mdl().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]) for tr, te in StratifiedKFold(5, shuffle=True, random_state=int(rng.integers(1e9))).split(X, y)]))


rng = np.random.default_rng(11); res = []
for d in [0.0, 0.25, 0.5, 1.0, 2.0]:
    for n in [3, 5, 10, 20]:
        a, c = [], []
        for _ in range(REPS):
            X, y, g = make(n, d, rng); a.append(lopo(X, y, g)); c.append(cellcv(X, y, rng))
        res.append({'program_effect_sd': d, 'patients': n, 'lopo_auc_mean': float(np.mean(a)), 'lopo_auc_sd': float(np.std(a)), 'cell_cv_auc_mean': float(np.mean(c)), 'cell_cv_auc_sd': float(np.std(c)),
                    'fraction_lopo_above_0.6': float(np.mean(np.array(a) > 0.6))})
        print(res[-1], flush=True)
OUT.write_text(json.dumps({'params': {'genes': G, 'cells_per_sample': CELLS, 'within_block_correlation': RHO, 'sample_offset_scale': 0.5, 'reps': REPS, 'cell_level_sd_of_program_score': sd_score}, 'results': res}, indent=2))
