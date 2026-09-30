import gzip, itertools, json
from pathlib import Path
import numpy as np

PROGRAMS = {
    'ACKR2_TGFb': ['ACKR2','TGFB1','TGFB2','TGFB3','TGFBR1','TGFBR2'],
    'MDM2_DDR': ['MDM2','CDKN1A','GADD45A','DDB2','BBC3'],
    'Tcell_senescence': ['CD8A','CD8B','CDKN2A','TOX','TIGIT','LAG3'],
    'Hypoxia': ['HIF1A','EPAS1','VEGFA','CA9','SLC2A1','LDHA'],
    'CAF_ECM': ['COL1A1','COL1A2','DCN','LUM','FAP','PDGFRA'],
}
ROOT = Path('work/external_public/GSE168009')
OUT = Path('work/analysis_gse168009')
OUT.mkdir(parents=True, exist_ok=True)

# The family XML provides the authoritative sample labels; the count matrix uses the same labels.
labels = ['NDB'] * 4 + ['DCB'] * 5
samples = ['NDB1','NDB2','NDB3','NDB4','DCB1','DCB2','DCB3','DCB4','DCB5']
wanted = set(sum(PROGRAMS.values(), []))
raw = {}
with gzip.open(ROOT/'GSE168009_Raw_count.txt.gz', 'rt', encoding='utf-8') as f:
    header = f.readline().rstrip('\n').split('\t')[1:]
    for line in f:
        p = line.rstrip('\n').split('\t')
        if p[0] in wanted:
            raw[p[0]] = np.array([float(x) for x in p[1:]], dtype=float)
if header != samples:
    raise ValueError(f'Unexpected sample order: {header}')

# Counts -> log2 CPM; each gene is then standardized within this external cohort,
# matching the existing cohort-level program score implementation.
all_counts = np.zeros(len(samples))
with gzip.open(ROOT/'GSE168009_Raw_count.txt.gz', 'rt', encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\n').split('\t')
        all_counts += np.array([float(x) for x in p[1:]], dtype=float)
scores = {}
for name, genes in PROGRAMS.items():
    z = []
    for gene in genes:
        if gene in raw:
            x = np.log2((raw[gene] / all_counts) * 1e6 + 1)
            z.append((x - x.mean()) / (x.std() or 1.0))
    scores[name] = np.mean(z, axis=0)

groups = np.array(labels)
ndb = groups == 'NDB'
dcb = groups == 'DCB'
rng = np.random.default_rng(20260930)
result = {
    'dataset': 'GSE168009',
    'design': 'pretreatment RNA-seq; no durable benefit (NDB) vs durable clinical benefit (DCB) after platinum-based CCRT',
    'n_samples': len(samples), 'n_ndb': int(ndb.sum()), 'n_dcb': int(dcb.sum()),
    'genes_found': sorted(raw), 'programs': {}
}
for name, x in scores.items():
    observed = float(x[ndb].mean() - x[dcb].mean())
    # Exact 4-vs-5 label permutation test (126 possible assignments).
    extreme = total = 0
    for nd_idx in itertools.combinations(range(len(x)), int(ndb.sum())):
        mask = np.zeros(len(x), dtype=bool); mask[list(nd_idx)] = True
        diff = float(x[mask].mean() - x[~mask].mean())
        extreme += abs(diff) >= abs(observed) - 1e-15
        total += 1
    result['programs'][name] = {
        'genes_found': [g for g in PROGRAMS[name] if g in raw],
        'mean_NDB_minus_DCB': observed,
        'mean_NDB': float(x[ndb].mean()), 'mean_DCB': float(x[dcb].mean()),
        'exact_permutation_p_two_sided': (1 + extreme) / (total + 1),
        'direction': 'higher_NDB' if observed > 0 else 'higher_DCB'
    }

# BH correction across the five prespecified program tests.
items = list(result['programs'].items())
order = sorted(range(len(items)), key=lambda i: items[i][1]['exact_permutation_p_two_sided'])
q = [1.0] * len(items)
for rank, i in enumerate(order, start=1):
    q[i] = min(1.0, items[i][1]['exact_permutation_p_two_sided'] * len(items) / rank)
for i, (name, _) in enumerate(items):
    result['programs'][name]['BH_q_across_five_programs'] = q[i]

(OUT/'program_response_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
