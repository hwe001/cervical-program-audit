import csv, json
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = (Path(__file__).resolve().parents[1] / 'work')
src = ROOT / 'analysis_gse236738' / 'cell_annotations.csv'
out = ROOT / 'analysis_gse236738'
programs = ['ACKR2_TGFb','MDM2_DDR','Tcell_senescence','Hypoxia','CAF_ECM']

rows = []
with src.open(encoding='utf-8') as f:
    for r in csv.DictReader(f):
        for p in programs: r[p] = float(r[p])
        rows.append(r)

groups = defaultdict(list)
for r in rows:
    groups[(r['patient'], r['treatment'], r['cell_type'])].append(r)

means = {}
for key, rs in groups.items():
    means[key] = {p: float(np.mean([r[p] for r in rs])) for p in programs} | {'n_cells': len(rs)}

effect_rows = []
for patient in ['P1','P2','P3']:
    celltypes = sorted({k[2] for k in means if k[0] == patient})
    for ct in celltypes:
        pre = means.get((patient,'TN',ct)); post = means.get((patient,'CCRT',ct))
        if not pre or not post: continue
        row = {'patient': patient, 'cell_type': ct, 'n_pre': pre['n_cells'], 'n_post': post['n_cells']}
        for p in programs:
            row[f'{p}_pre'] = pre[p]; row[f'{p}_post'] = post[p]; row[f'{p}_delta'] = post[p] - pre[p]
        effect_rows.append(row)

with (out / 'paired_celltype_effects.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=effect_rows[0].keys()); writer.writeheader(); writer.writerows(effect_rows)

# Candidate resistant epithelial cells: top decile of the combined MDM2 + ACKR2/TGFb score within each sample.
epi = [r for r in rows if r['cell_type'] == 'Epithelial']
for r in epi:
    r['combined_resistance'] = r['MDM2_DDR'] + r['ACKR2_TGFb'] + r['Tcell_senescence'] + r['Hypoxia']
thresholds = {}
for sample in sorted({r['sample'] for r in epi}):
    vals = [r['combined_resistance'] for r in epi if r['sample'] == sample]
    thresholds[sample] = float(np.quantile(vals, .9)) if vals else None
candidate_summary = []
for sample in sorted(thresholds):
    rs = [r for r in epi if r['sample'] == sample]
    cand = [r for r in rs if r['combined_resistance'] >= thresholds[sample]]
    row = {'sample': sample, 'patient': cand[0]['patient'] if cand else '', 'treatment': cand[0]['treatment'] if cand else '', 'epithelial_cells': len(rs), 'candidate_cells': len(cand), 'candidate_fraction': len(cand)/len(rs) if rs else 0}
    for p in programs: row[p] = float(np.mean([r[p] for r in cand])) if cand else None
    candidate_summary.append(row)
with (out / 'candidate_epithelial_resistance_summary.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=candidate_summary[0].keys()); writer.writeheader(); writer.writerows(candidate_summary)

(out / 'paired_effects_metadata.json').write_text(json.dumps({'n_cells': len(rows), 'n_effect_rows': len(effect_rows), 'note': 'Descriptive paired effects; inferential testing requires patient-level replication and refined annotation.'}, indent=2), encoding='utf-8')
print(json.dumps({'effect_rows': len(effect_rows), 'candidate_summary_rows': len(candidate_summary), 'output': str(out)}, indent=2))
