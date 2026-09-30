"""
Leave-one-patient-out feature stability, subsample-based stability selection,
and pathway-level stability for the GSE236738 55-gene resistance panel.

Addresses manuscript_v1.md's "Remaining analyses before submission":
"quantify leave-one-patient-out feature stability, bootstrap stability, and
pathway-level stability".

Works directly on the 55 curated genes (no SVD latent step) so that feature
and pathway importances are directly interpretable, unlike
ai_robustness_benchmark.py's SVD-component loadings.

Two complementary questions this answers:
  1. Are the SAME individual genes important regardless of which patient is
     held out / which cell subsample is drawn? (gene-level stability)
  2. Is the answer more stable if we ask the same question one level up, at
     the resistance-PROGRAM level rather than the individual-gene level?
     (this directly supports the manuscript's framing that AI's contribution
     is patient-aware integration across programs, not gene-level discovery)
"""
import gzip
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.io import mmread
from scipy.sparse import vstack
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut

ROOT = (Path(__file__).resolve().parents[1] / 'work')
MAT = ROOT / 'GSE236738_matrix'
OUT = ROOT / 'analysis_gse236738'
OUT.mkdir(exist_ok=True)

SAMPLE_INFO = {
    'GSM7574777_FXW': ('P1', 'TN'), 'GSM7574778_FXW2': ('P1', 'CCRT'),
    'GSM7574779_SBJ': ('P2', 'TN'), 'GSM7574780_SBJ2': ('P2', 'CCRT'),
    'GSM7574781_WGX': ('P3', 'TN'), 'GSM7574782_WGX2': ('P3', 'CCRT'),
}
GENES_KEEP = set(
    'ACKR2 TGFB1 TGFB2 TGFB3 TGFBR1 TGFBR2 MDM2 CDKN1A GADD45A DDB2 BBC3 '
    'CD8A CD8B CDKN2A TOX TIGIT LAG3 HIF1A EPAS1 VEGFA CA9 SLC2A1 LDHA '
    'COL1A1 COL1A2 DCN LUM FAP PDGFRA EPCAM KRT8 KRT18 KRT19 KRT14 KRT17 '
    'PAX8 CD3D CD3E TRBC1 NKG7 GNLY MS4A1 CD79A LST1 TYROBP FCER1G LILRB1 '
    'CTSS AIF1 PECAM1 VWF EMCN KDR KIT TPSAB1'.split()
)
# Same 5 resistance programs used throughout (analyze_gse236738.py,
# gene_level_ai.py). Only genes that are both in GENES_KEEP and in a program
# contribute to that program's aggregate importance.
PROGRAMS = {
    'ACKR2_TGFb': ['ACKR2', 'TGFB1', 'TGFB2', 'TGFB3', 'TGFBR1', 'TGFBR2'],
    'MDM2_DDR': ['MDM2', 'CDKN1A', 'GADD45A', 'DDB2', 'BBC3'],
    'Tcell_senescence': ['CD8A', 'CD8B', 'CDKN2A', 'TOX', 'TIGIT', 'LAG3'],
    'Hypoxia': ['HIF1A', 'EPAS1', 'VEGFA', 'CA9', 'SLC2A1', 'LDHA'],
    'CAF_ECM': ['COL1A1', 'COL1A2', 'DCN', 'LUM', 'FAP', 'PDGFRA'],
}
TOP_K = 15
N_SUBSAMPLES = 200
SUBSAMPLE_FRAC = 0.5
RNG_SEED = 42


def load_data():
    parts, labels, groups, feature_names = [], [], [], None
    for sample, (patient, treatment) in SAMPLE_INFO.items():
        m = mmread(MAT / f'{sample}.matrix.mtx.gz').tocsr()
        with gzip.open(MAT / f'{sample}.features.tsv.gz', 'rt', encoding='utf-8') as f:
            genes = [x.rstrip('\n').split('\t')[1] for x in f]
        idx = {g.upper(): i for i, g in enumerate(genes)}
        names = sorted(g for g in GENES_KEEP if g in idx)
        feature_names = names
        rows = [idx[g] for g in names]
        x = m[rows, :].tocsr()
        lib = np.asarray(m.sum(axis=0)).ravel()
        x = x.multiply(1e4 / np.maximum(lib, 1))
        x.data = np.log1p(x.data)
        parts.append(x.T)
        labels.extend([1 if treatment == 'CCRT' else 0] * x.shape[1])
        groups.extend([patient] * x.shape[1])
    X = vstack(parts).tocsr().toarray()  # 55 features is small enough to densify
    return X, np.array(labels), np.array(groups), feature_names


def gene_to_program_matrix(feature_names):
    """(n_genes, n_programs) 0/1 matrix mapping genes to programs they belong to."""
    prog_names = list(PROGRAMS)
    M = np.zeros((len(feature_names), len(prog_names)))
    for j, p in enumerate(prog_names):
        for g in PROGRAMS[p]:
            if g in feature_names:
                M[feature_names.index(g), j] = 1.0
    return M, prog_names


def rank_stability(importance_matrix):
    """Given (n_runs, n_features) importances, return mean pairwise Spearman
    rank correlation and mean pairwise top-K Jaccard overlap across runs."""
    n_runs = importance_matrix.shape[0]
    if n_runs < 2:
        return {'mean_spearman': None, 'mean_top_k_jaccard': None}
    spearmans, jaccards = [], []
    k = min(TOP_K, importance_matrix.shape[1])
    for i, j in combinations(range(n_runs), 2):
        rho, _ = spearmanr(importance_matrix[i], importance_matrix[j])
        spearmans.append(rho)
        top_i = set(np.argsort(importance_matrix[i])[::-1][:k])
        top_j = set(np.argsort(importance_matrix[j])[::-1][:k])
        jaccards.append(len(top_i & top_j) / len(top_i | top_j))
    return {'mean_spearman': float(np.mean(spearmans)), 'mean_top_k_jaccard': float(np.mean(jaccards)), 'n_pairs': len(spearmans)}


def selection_frequency(importance_matrix, names, k=TOP_K):
    """Fraction of runs in which each feature/program appears in the top-k
    by importance — the core "stability selection" statistic."""
    n_runs, n_feat = importance_matrix.shape
    k = min(k, n_feat)
    counts = np.zeros(n_feat)
    for row in importance_matrix:
        top = np.argsort(row)[::-1][:k]
        counts[top] += 1
    freq = counts / n_runs
    ci_lo = np.percentile(importance_matrix, 2.5, axis=0)
    ci_hi = np.percentile(importance_matrix, 97.5, axis=0)
    order = np.argsort(freq)[::-1]
    return [
        {
            'name': names[i], 'selection_frequency': float(freq[i]),
            'mean_importance': float(importance_matrix[:, i].mean()),
            'importance_95ci': [float(ci_lo[i]), float(ci_hi[i])],
        }
        for i in order
    ]


def fit_importance(X, y, seed):
    clf = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=10, class_weight='balanced_subsample',
        random_state=seed, n_jobs=-1,
    )
    clf.fit(X, y)
    return clf.feature_importances_


def main():
    X, y, groups, feature_names = load_data()
    gene_to_prog, prog_names = gene_to_program_matrix(feature_names)

    # 1. Leave-one-patient-out gene-level and program-level importances.
    logo = LeaveOneGroupOut()
    lopo_gene_importances = []
    lopo_fold_meta = []
    for train, test in logo.split(X, y, groups):
        imp = fit_importance(X[train], y[train], seed=42)
        lopo_gene_importances.append(imp)
        lopo_fold_meta.append({'held_out_patient': str(groups[test][0]), 'n_train': int(len(train))})
    lopo_gene_importances = np.vstack(lopo_gene_importances)
    lopo_program_importances = lopo_gene_importances @ gene_to_prog

    # 2. Subsample-based stability selection (Meinshausen-Buhlmann style):
    # repeatedly fit on a random ~50% subsample of the pooled cells, track
    # how often each gene/program lands in the top-K by importance.
    rng = np.random.default_rng(RNG_SEED)
    n = X.shape[0]
    sub_gene_importances = np.zeros((N_SUBSAMPLES, X.shape[1]))
    for r in range(N_SUBSAMPLES):
        idx = rng.choice(n, size=int(n * SUBSAMPLE_FRAC), replace=False)
        sub_gene_importances[r] = fit_importance(X[idx], y[idx], seed=int(rng.integers(0, 2**31 - 1)))
    sub_program_importances = sub_gene_importances @ gene_to_prog

    result = {
        'n_cells': int(X.shape[0]),
        'n_genes': int(X.shape[1]),
        'n_programs': len(prog_names),
        'lopo_folds': lopo_fold_meta,
        'gene_level': {
            'lopo_rank_stability': rank_stability(lopo_gene_importances),
            'subsample_rank_stability': rank_stability(sub_gene_importances),
            'subsample_selection_frequency_top20': selection_frequency(sub_gene_importances, feature_names)[:20],
            'subsample_selection_frequency_all': selection_frequency(sub_gene_importances, feature_names),
        },
        'program_level': {
            'lopo_rank_stability': rank_stability(lopo_program_importances),
            'subsample_rank_stability': rank_stability(sub_program_importances),
            'subsample_selection_frequency': selection_frequency(sub_program_importances, prog_names, k=len(prog_names)),
        },
        'n_subsamples': N_SUBSAMPLES,
        'subsample_fraction': SUBSAMPLE_FRAC,
        'top_k_for_selection_frequency_and_jaccard': TOP_K,
        'caveat': (
            'LOPO stability is computed across only 3 folds (3 patients), so '
            'its rank-correlation/Jaccard estimates are themselves noisy point '
            'estimates, not a well-powered statistic — report alongside the '
            'subsample-based stability selection (200 runs), which is the '
            'better-powered of the two. Subsampling draws cells from the pooled '
            'pool without patient stratification, so it reflects sampling '
            'variability, not patient-to-patient generalization; LOPO is the '
            'complementary check for that.'
        ),
    }
    (OUT / 'stability_analysis.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('gene_level',)}, indent=2))
    print('\nTop-5 genes by subsample selection frequency:')
    for row in result['gene_level']['subsample_selection_frequency_top20'][:5]:
        print(f"  {row['name']:10s} freq={row['selection_frequency']:.2f}  importance={row['mean_importance']:.4f}  95%CI={row['importance_95ci']}")
    print(f"\nGene-level LOPO mean Spearman:     {result['gene_level']['lopo_rank_stability']['mean_spearman']:.3f}")
    print(f"Program-level LOPO mean Spearman:  {result['program_level']['lopo_rank_stability']['mean_spearman']:.3f}")
    print(f"Gene-level subsample mean Spearman:    {result['gene_level']['subsample_rank_stability']['mean_spearman']:.3f}")
    print(f"Program-level subsample mean Spearman: {result['program_level']['subsample_rank_stability']['mean_spearman']:.3f}")


if __name__ == '__main__':
    main()
