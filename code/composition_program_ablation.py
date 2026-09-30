"""
Composition-only vs program-only vs combined ablation.

sample_level_benchmark.py already fits a model on the COMBINED feature set
(program-score means + cell-type composition proportions). This script asks
the ablation question the manuscript's "remaining analyses" list calls for:
how much of that combined model's signal comes from compositional shifts
(cell-type proportions changing after treatment) versus program-score shifts
(the same cell types expressing resistance programs differently), and does
combining them actually help over either alone?

Same six-sample, patient-held-out (leave-one-patient-out) design, same two
model families, and the same identical 500-permutation null construction as
sample_level_benchmark.py, so the three feature sets are directly comparable.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = (Path(__file__).resolve().parents[1] / 'work')
df = pd.read_csv(ROOT / 'analysis_gse236738' / 'cell_annotations.csv')
programs = ['ACKR2_TGFb', 'MDM2_DDR', 'Tcell_senescence', 'Hypoxia', 'CAF_ECM']

means = df.groupby('sample')[programs].mean()
props = pd.crosstab(df['sample'], df['cell_type'], normalize='index')
meta = df[['sample', 'patient', 'treatment']].drop_duplicates().set_index('sample')

FEATURE_SETS = {
    'composition_only': props,
    'program_only': means,
    'combined': means.join(props, how='left').fillna(0),
}


def run_feature_set(X, y, groups):
    logo = LeaveOneGroupOut()
    models = {
        'logistic_regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
        'random_forest': RandomForestClassifier(n_estimators=150, min_samples_leaf=1, class_weight='balanced', random_state=42, n_jobs=2),
    }
    results = []
    for name, template in models.items():
        for train, test in logo.split(X, y, groups):
            model = template
            model.fit(X.iloc[train], y[train])
            p = model.predict_proba(X.iloc[test])[:, 1]
            results.append({'model': name, 'held_out_patient': str(groups[test][0]), 'auc': float(roc_auc_score(y[test], p))})

    rng = np.random.default_rng(42)
    null = []
    for rep in range(500):
        yp = rng.permutation(y)
        fold = []
        for train, test in logo.split(X, yp, groups):
            if len(np.unique(yp[test])) < 2:
                continue
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
            model.fit(X.iloc[train], yp[train])
            fold.append(roc_auc_score(yp[test], model.predict_proba(X.iloc[test])[:, 1]))
        if fold:
            null.append(float(np.mean(fold)))

    return {
        'n_features': int(X.shape[1]),
        'features': list(X.columns),
        'model_comparison': results,
        'observed_mean_auc': {m: float(np.mean([r['auc'] for r in results if r['model'] == m])) for m in ['logistic_regression', 'random_forest']},
        'permutation_null_mean': float(np.mean(null)),
        'permutation_null_95th': float(np.quantile(null, 0.95)),
    }


def main():
    summary = {}
    for name, X in FEATURE_SETS.items():
        X = X.loc[meta.index]
        y = (meta.treatment == 'CCRT').astype(int).to_numpy()
        groups = meta.patient.to_numpy()
        summary[name] = run_feature_set(X, y, groups)

    summary['caveat'] = (
        'Only six biological samples are available; every feature set here shares the same '
        'leakage-control-benchmark limitation already stated for sample_level_benchmark.json. '
        'This ablation is about relative signal contribution between feature sets, not an '
        'independent claim of predictive validity for any single one.'
    )
    out = ROOT / 'analysis_gse236738' / 'composition_program_ablation.json'
    out.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print(f"{'feature_set':18s} {'n_feat':7s} {'LR AUC':8s} {'RF AUC':8s} {'null mean':10s} {'null p95':10s}")
    for name in FEATURE_SETS:
        s = summary[name]
        print(f"{name:18s} {s['n_features']:<7d} {s['observed_mean_auc']['logistic_regression']:<8.3f} {s['observed_mean_auc']['random_forest']:<8.3f} {s['permutation_null_mean']:<10.3f} {s['permutation_null_95th']:<10.3f}")


if __name__ == '__main__':
    main()
