"""
Gene-level (not program-mean) Spearman association with cervical disease
progression stage in GSE63514, for all 29 pharmacological-ranking candidate
genes. Addresses reviewer critique: the pharmacological ranking currently
inherits one program-level rho for every gene in a program (e.g. all six
Hypoxia genes get rho=0.639), so within-program ranking is driven entirely
by classifier importance, not independent cross-cohort evidence. This
computes a real per-gene rho/p, then Benjamini-Hochberg-corrects across all
29 tests (not just the 5 program-level tests), so pharmacological_ranking.py
can use gene-specific reproducibility evidence and note which genes survive
multiple-testing correction.

Reuses the same probe->gene mapping and series-matrix loading as
score_gse63514.py; do not duplicate that file, this is the gene-level
extension of it.
"""
import gzip
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

ROOT = (Path(__file__).resolve().parents[1] / 'work' / 'validation')

PROGRAMS = {
    'ACKR2_TGFb': ['ACKR2', 'TGFB1', 'TGFB2', 'TGFB3', 'TGFBR1', 'TGFBR2'],
    'MDM2_DDR': ['MDM2', 'CDKN1A', 'GADD45A', 'DDB2', 'BBC3'],
    'Tcell_senescence': ['CD8A', 'CD8B', 'CDKN2A', 'TOX', 'TIGIT', 'LAG3'],
    'Hypoxia': ['HIF1A', 'EPAS1', 'VEGFA', 'CA9', 'SLC2A1', 'LDHA'],
    'CAF_ECM': ['COL1A1', 'COL1A2', 'DCN', 'LUM', 'FAP', 'PDGFRA'],
}
ALL_GENES = sorted({g for genes in PROGRAMS.values() for g in genes})
STAGE_MAP = {
    'normal cervical epithelium': 0,
    'cervical intraepithelial neoplasm, low grade lesion': 1,
    'cervical intraepithelial neoplasm, moderate grade lesion': 2,
    'cervical intraepithelial neoplasm, high grade lesion': 3,
    'cervical squamous epithelial cancer': 4,
}


def load_expr_and_stage():
    mapping = {}
    with gzip.open(ROOT / 'GPL570.annot.gz', 'rt', errors='replace') as f:
        for line in f:
            if line.startswith('!platform_table_begin'):
                header = next(f).rstrip('\n').split('\t')
                ii, si = header.index('ID'), header.index('Gene symbol')
                break
        for line in f:
            if line.startswith('!platform_table_end'):
                break
            p = line.rstrip('\n').split('\t')
            if len(p) > max(ii, si):
                sym = p[si].split(' /// ')[0].strip()
                if sym and sym != '---':
                    mapping[p[ii]] = sym

    with gzip.open(ROOT / 'GSE63514_series_matrix.txt.gz', 'rt', errors='replace') as f:
        lines = f.readlines()
    begin = next(i for i, x in enumerate(lines) if x.startswith('!series_matrix_table_begin'))
    end = next(i for i in range(begin + 1, len(lines)) if lines[i].startswith('!series_matrix_table_end'))
    df = pd.read_csv(io.StringIO(''.join(lines[begin + 1:end])), sep='\t')
    df.rename(columns={df.columns[0]: 'probe'}, inplace=True)
    df['gene'] = df['probe'].map(mapping)
    df = df[df.gene.notna()]
    samples = [c for c in df.columns if c not in ['probe', 'gene']]
    expr = df.groupby('gene')[samples].mean()

    classes = {}
    for line in lines[:begin]:
        if line.startswith('!Sample_characteristics_ch1') and 'tissue type:' in line:
            vals = [x.strip('"') for x in line.rstrip('\n').split('\t')[1:]]
            classes.update({s: v.split(':', 1)[-1].strip() for s, v in zip(samples, vals)})
    stage = pd.Series({s: STAGE_MAP.get(classes.get(s, ''), np.nan) for s in samples})
    return expr, stage, samples


def main():
    expr, stage, samples = load_expr_and_stage()
    stage = stage.dropna()
    samples = [s for s in samples if s in stage.index]

    gene_to_program = {g: p for p, genes in PROGRAMS.items() for g in genes if g in genes}

    rows = []
    for gene in ALL_GENES:
        program = next(p for p, genes in PROGRAMS.items() if gene in genes)
        if gene not in expr.index:
            rows.append({'gene': gene, 'program': program, 'in_platform': False, 'rho': None, 'p': None, 'n': len(samples)})
            continue
        vals = expr.loc[gene, samples]
        rho, p = spearmanr(stage.loc[samples], vals)
        rows.append({'gene': gene, 'program': program, 'in_platform': True, 'rho': float(rho), 'p': float(p), 'n': len(samples)})

    testable = [r for r in rows if r['p'] is not None]
    reject, qvals, _, _ = multipletests([r['p'] for r in testable], method='fdr_bh')
    for r, q, rej in zip(testable, qvals, reject):
        r['q_bh_across_29_genes'] = float(q)
        r['significant_at_q0.05'] = bool(rej)

    # also BH-correct the 5 program-level tests separately, since that's what
    # sections 3.2/3.4 report and the reviewer specifically asked for BH
    # correction "across programs" as a distinct, smaller-family test
    program_rows = []
    for program, genes in PROGRAMS.items():
        present = [g for g in genes if g in expr.index]
        prog_score = expr.loc[present, samples].mean(axis=0)
        rho, p = spearmanr(stage.loc[samples], prog_score)
        program_rows.append({'program': program, 'rho': float(rho), 'p': float(p), 'n': len(samples)})
    preject, pqvals, _, _ = multipletests([r['p'] for r in program_rows], method='fdr_bh')
    for r, q, rej in zip(program_rows, pqvals, preject):
        r['q_bh_across_5_programs'] = float(q)
        r['significant_at_q0.05'] = bool(rej)

    out = {
        'gene_level': rows,
        'program_level_bh_corrected': program_rows,
        'note': (
            'Gene-level rho/p computed independently per gene (not inherited from the program mean), '
            'to give the pharmacological ranking gene-specific cross-cohort evidence instead of one '
            'shared program-level value. Genes not on the GPL570 platform (in_platform=false) have no '
            'rho/p and should not be silently treated as non-significant. GSE6213 is deliberately not '
            'BH-corrected or given formal p-values here: with pretreatment arms as small as n=2 '
            '(carbon-ion) and n=18-20 for the other two modalities, only directional consistency is '
            'reported for that cohort (sections 3.2/3.4), not formal significance.'
        ),
    }
    (ROOT / 'gse63514_gene_level_progression.json').write_text(json.dumps(out, indent=2), encoding='utf-8')

    print('Program-level (BH-corrected across 5 programs):')
    for r in sorted(program_rows, key=lambda x: x['p']):
        print(f"  {r['program']:18s} rho={r['rho']:.3f}  p={r['p']:.2e}  q={r['q_bh_across_5_programs']:.2e}  sig={r['significant_at_q0.05']}")
    print('\nGene-level (BH-corrected across 29 genes), sorted by p:')
    for r in sorted(testable, key=lambda x: x['p']):
        print(f"  {r['gene']:8s} ({r['program']:18s}) rho={r['rho']:.3f}  p={r['p']:.2e}  q={r['q_bh_across_29_genes']:.2e}  sig={r['significant_at_q0.05']}")
    missing = [r['gene'] for r in rows if not r['in_platform']]
    if missing:
        print(f'\nNot on GPL570 (no gene-level rho available): {missing}')


if __name__ == '__main__':
    main()
