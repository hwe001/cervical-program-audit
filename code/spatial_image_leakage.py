"""Spatial leakage test on three Visium cervical-cancer slides (Patient5-7; Su et al., Front Immunol 2025; Zenodo 10.5281/zenodo.16917924).

Classical ML: hand-built H&E tile features (colour, haematoxylin/eosin, texture) predict (a) epithelial-high and (b) fibroblast-high spots (expression-derived, above the slide median)
and (c) spatially smooth arbitrary labels that carry no biology. Validation: random-spot 5-fold, spatial-block 5-fold, and leave-one-slide-out.
"""
import gzip, json, os
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
from scipy.io import mmread
from skimage.color import rgb2hed, rgb2hsv, rgb2gray
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

Image.MAX_IMAGE_PIXELS = None
REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('CERVIX_VISIUM_DIR', REPO / 'work' / 'visium'))
RES = REPO / 'results'
PATIENTS = ['Patient5', 'Patient6', 'Patient7']
EPI = ['EPCAM', 'KRT8', 'KRT18', 'KRT19', 'KRT14', 'KRT17', 'MUC1', 'PAX8']
FIB = ['COL1A1', 'COL1A2', 'DCN', 'LUM', 'COL3A1', 'FAP']
PATCH = 12   # half-width in hires pixels (24 x 24 patch, about 1.6 spot diameters)


def load_slide(p):
    folder = DATA / p; sf = json.load(open(folder / 'spatial' / 'scalefactors_json.json'))
    pos = pd.read_csv(folder / 'tissue_positions.csv'); pos = pos[pos.in_tissue == 1].reset_index(drop=True)
    m = mmread(folder / 'filtered_feature_bc_matrix' / 'matrix.mtx.gz').tocsc()
    with gzip.open(folder / 'filtered_feature_bc_matrix' / 'features.tsv.gz', 'rt') as f: genes = [x.split('\t')[1].upper() for x in f]
    with gzip.open(folder / 'filtered_feature_bc_matrix' / 'barcodes.tsv.gz', 'rt') as f: bcs = [x.strip() for x in f]
    col = {b: i for i, b in enumerate(bcs)}; idx = np.array([col[b] for b in pos.barcode]); m = m[:, idx]
    lib = np.asarray(m.sum(axis=0)).ravel(); gi = {g: i for i, g in enumerate(genes)}
    def score(gs):
        rows = [gi[g] for g in gs if g in gi]; x = m[rows, :].toarray() / np.maximum(lib, 1) * 1e4
        return np.log1p(x).mean(axis=0)
    pos['epi'] = score(EPI); pos['fib'] = score(FIB)
    img = np.asarray(Image.open(folder / 'spatial' / 'tissue_hires_image.png').convert('RGB'))
    s = sf['tissue_hires_scalef']; pos['x'] = pos.pxl_col_in_fullres * s; pos['y'] = pos.pxl_row_in_fullres * s
    return pos, img


def tile_features(img, hed, hsv, gray, x, y):
    x0, y0 = int(round(x)), int(round(y)); h, w = gray.shape
    xs, xe, ys, ye = max(0, x0 - PATCH), min(w, x0 + PATCH), max(0, y0 - PATCH), min(h, y0 + PATCH)
    t = img[ys:ye, xs:xe].reshape(-1, 3) / 255.0; th = hed[ys:ye, xs:xe].reshape(-1, 3); tv = hsv[ys:ye, xs:xe].reshape(-1, 3); g = gray[ys:ye, xs:xe]
    f = list(t.mean(0)) + list(t.std(0)) + list(tv.mean(0)) + list(tv.std(0)) + list(th.mean(0)) + list(th.std(0))
    gy, gx = np.gradient(g); f.append(np.hypot(gx, gy).mean())
    g8 = (g * 255).astype(np.uint8); lbp = local_binary_pattern(g8, 8, 1, 'uniform'); f += list(np.bincount(lbp.astype(int).ravel(), minlength=10)[:10] / lbp.size)
    gl = graycomatrix(g8 // 8, [1], [0], levels=32, symmetric=True, normed=True)
    f += [graycoprops(gl, k)[0, 0] for k in ('contrast', 'homogeneity', 'energy', 'correlation')]
    return f


def neighbours(pos):
    key = {(r, c): i for i, (r, c) in enumerate(zip(pos.array_row, pos.array_col))}; nb = []
    for r, c in zip(pos.array_row, pos.array_col):
        nb.append([key[k] for k in ((r, c - 2), (r, c + 2), (r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)) if k in key])
    return nb


def smooth_noise(nb, n, seed, iters=6):
    z = np.random.default_rng(seed).normal(size=n)
    for _ in range(iters): z = np.array([(z[i] + sum(z[j] for j in nb[i])) / (1 + len(nb[i])) for i in range(n)])
    return z


MODELS = {
    'logistic_regression': lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=0.1)),
    'random_forest': lambda: RandomForestClassifier(n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=0),
    'gradient_boosting': lambda: HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=0),
    'neural_network': lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-2, max_iter=100, early_stopping=True, random_state=0)),
}


def auc(mk, a, b, c, d): return roc_auc_score(d, mk().fit(a, b).predict_proba(c)[:, 1])


slides = {}
for p in PATIENTS:
    pos, img = load_slide(p); hed = rgb2hed(img); hsv = rgb2hsv(img); gray = rgb2gray(img)
    F = np.array([tile_features(img, hed, hsv, gray, x, y) for x, y in zip(pos.x, pos.y)]); nb = neighbours(pos)
    labels = {'epithelial_high': (pos.epi > pos.epi.median()).astype(int).values, 'fibroblast_high': (pos.fib > pos.fib.median()).astype(int).values}
    for sd in (1, 2, 3):
        z = smooth_noise(nb, len(pos), sd); labels[f'arbitrary_smooth_{sd}'] = (z > np.median(z)).astype(int)
    # autocorrelation of the expression-derived scores: correlation of a spot with the mean of its neighbours
    ac = {k: float(np.corrcoef(v, [np.mean([v[j] for j in nb[i]]) if nb[i] else v[i] for i in range(len(v))])[0, 1]) for k, v in (('epi', pos.epi.values), ('fib', pos.fib.values))}
    # spatial blocks: 5 x 5 grid over the array coordinates
    rb = pd.cut(pos.array_row, 5, labels=False).values; cb = pd.cut(pos.array_col, 5, labels=False).values
    slides[p] = dict(F=F, labels=labels, block=rb * 5 + cb, n=len(pos), autocorr=ac)
    print(p, 'spots', len(pos), 'features', F.shape[1], 'neighbour correlation', {k: round(v, 2) for k, v in ac.items()}, flush=True)

out = {'patients': PATIENTS, 'spots': {p: slides[p]['n'] for p in PATIENTS}, 'neighbour_correlation_of_scores': {p: slides[p]['autocorr'] for p in PATIENTS}, 'feature_count': int(slides[PATIENTS[0]]['F'].shape[1]), 'results': {}}
label_names = ['epithelial_high', 'fibroblast_high', 'arbitrary_smooth_1', 'arbitrary_smooth_2', 'arbitrary_smooth_3']
for lab in label_names:
    out['results'][lab] = {}
    for m, mk in MODELS.items():
        rnd, blk = [], []
        for p in PATIENTS:
            S = slides[p]; y = S['labels'][lab]; F = S['F']
            rnd.append(float(np.mean([auc(mk, F[a], y[a], F[b], y[b]) for a, b in KFold(5, shuffle=True, random_state=0).split(F)])))
            blk.append(float(np.mean([auc(mk, F[a], y[a], F[b], y[b]) for a, b in GroupKFold(5).split(F, y, S['block']) if len(set(y[b])) == 2])))
        loso, loso_z = [], []
        for hold in PATIENTS:
            tr = [p for p in PATIENTS if p != hold]
            Xtr = np.vstack([slides[p]['F'] for p in tr]); ytr = np.concatenate([slides[p]['labels'][lab] for p in tr]); Xte = slides[hold]['F']; yte = slides[hold]['labels'][lab]
            loso.append(float(auc(mk, Xtr, ytr, Xte, yte)))
            zs = lambda A: (A - A.mean(0)) / (A.std(0) + 1e-9)
            Xtr_z = np.vstack([zs(slides[p]['F']) for p in tr]); loso_z.append(float(auc(mk, Xtr_z, ytr, zs(Xte), yte)))
        out['results'][lab][m] = {'random_spot_per_slide': rnd, 'random_spot_mean': float(np.mean(rnd)), 'spatial_block_per_slide': blk, 'spatial_block_mean': float(np.mean(blk)),
                                  'leave_one_slide_out_per_slide': loso, 'leave_one_slide_out_mean': float(np.mean(loso)), 'leave_one_slide_out_slide_zscored_mean': float(np.mean(loso_z))}
        print(lab, m, 'random %.2f | block %.2f | LOSO %.2f | LOSO z %.2f' % (np.mean(rnd), np.mean(blk), np.mean(loso), np.mean(loso_z)), flush=True)
(RES / 'spatial_image_leakage.json').write_text(json.dumps(out, indent=2)); print('done')
