import csv, gzip, json
from pathlib import Path
import numpy as np
from scipy.io import mmread

ROOT = (Path(__file__).resolve().parents[1] / 'work')
MAT = ROOT / 'GSE236738_matrix'
OUT = ROOT / 'analysis_gse236738'
OUT.mkdir(exist_ok=True)

classes = {
    'T_NK': ['CD3D','CD3E','TRBC1','TRBC2','NKG7','GNLY'],
    'B_cell': ['MS4A1','CD79A','CD37','CD74','HLA-DRA'],
    'Myeloid': ['LST1','TYROBP','FCER1G','LILRB1','CTSS','AIF1'],
    'Fibroblast': ['COL1A1','COL1A2','DCN','LUM','COL3A1','FAP'],
    'Endothelial': ['PECAM1','VWF','EMCN','KDR','ESAM'],
    'Epithelial': ['EPCAM','KRT8','KRT18','KRT19','KRT14','KRT17','MUC1','PAX8'],
    'Mast': ['KIT','TPSAB1','TPSB2','MS4A2'],
}
programs = {
    'ACKR2_TGFb': ['ACKR2','TGFB1','TGFB2','TGFB3','TGFBR1','TGFBR2'],
    'MDM2_DDR': ['MDM2','CDKN1A','GADD45A','DDB2','BBC3'],
    'Tcell_senescence': ['CD8A','CD8B','CDKN2A','TOX','TIGIT','LAG3'],
    'Hypoxia': ['HIF1A','EPAS1','VEGFA','CA9','SLC2A1','LDHA'],
    'CAF_ECM': ['COL1A1','COL1A2','DCN','LUM','FAP','PDGFRA'],
}
sample_info = {
    'GSM7574777_FXW': ('P1','TN'), 'GSM7574778_FXW2': ('P1','CCRT'),
    'GSM7574779_SBJ': ('P2','TN'), 'GSM7574780_SBJ2': ('P2','CCRT'),
    'GSM7574781_WGX': ('P3','TN'), 'GSM7574782_WGX2': ('P3','CCRT'),
}

def read_matrix(prefix):
    mat = mmread(MAT / f'{prefix}.matrix.mtx.gz').tocsr()
    with gzip.open(MAT / f'{prefix}.features.tsv.gz', 'rt', encoding='utf-8') as f:
        genes = [line.rstrip('\n').split('\t')[1] for line in f]
    with gzip.open(MAT / f'{prefix}.barcodes.tsv.gz', 'rt', encoding='utf-8') as f:
        barcodes = [line.rstrip('\n') for line in f]
    return mat, genes, barcodes

rows = []
summary = []
for prefix, (patient, treatment) in sample_info.items():
    mat, genes, barcodes = read_matrix(prefix)
    gene_index = {g.upper(): i for i, g in enumerate(genes)}
    lib = np.asarray(mat.sum(axis=0)).ravel()
    norm = mat.multiply(1e4 / np.maximum(lib, 1)).tocsr()
    lognorm = norm.copy(); lognorm.data = np.log1p(lognorm.data)
    score_vectors = {}
    for name, markers in {**classes, **programs}.items():
        idx = [gene_index[g] for g in markers if g in gene_index]
        score_vectors[name] = np.asarray(lognorm[idx, :].mean(axis=0)).ravel() if idx else np.zeros(mat.shape[1])
    class_names = list(classes)
    class_matrix = np.vstack([score_vectors[n] for n in class_names])
    best = class_matrix.argmax(axis=0)
    sorted_scores = np.sort(class_matrix, axis=0)
    labels = np.array(class_names, dtype=object)[best]
    labels[(sorted_scores[-1] < 0.20) | ((sorted_scores[-1] - sorted_scores[-2]) < 0.05)] = 'Other/ambiguous'
    for i, barcode in enumerate(barcodes):
        row = {'sample': prefix, 'patient': patient, 'treatment': treatment, 'barcode': barcode, 'cell_type': labels[i]}
        for name in programs:
            row[name] = round(float(score_vectors[name][i]), 6)
        rows.append(row)
    for label in list(classes) + ['Other/ambiguous']:
        sel = labels == label
        if not np.any(sel):
            continue
        entry = {'sample': prefix, 'patient': patient, 'treatment': treatment, 'cell_type': label, 'n_cells': int(sel.sum())}
        for name in programs:
            entry[name] = float(np.mean(score_vectors[name][sel]))
        summary.append(entry)

with (OUT / 'cell_annotations.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
with (OUT / 'sample_celltype_summary.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=summary[0].keys()); writer.writeheader(); writer.writerows(summary)
(OUT / 'analysis_metadata.json').write_text(json.dumps({'n_cells': len(rows), 'samples': list(sample_info), 'programs': programs, 'classes': classes}, indent=2), encoding='utf-8')
print(json.dumps({'n_cells': len(rows), 'summary_rows': len(summary), 'output': str(OUT)}, indent=2))
