import gzip, json
from pathlib import Path
import numpy as np
from scipy.io import mmread
from scipy.sparse import vstack
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

ROOT = (Path(__file__).resolve().parents[1] / 'work'); MAT = ROOT / 'GSE236738_matrix'; OUT = ROOT / 'analysis_gse236738'
sample_info = {'GSM7574777_FXW': ('P1', 'TN'), 'GSM7574778_FXW2': ('P1', 'CCRT'), 'GSM7574779_SBJ': ('P2', 'TN'), 'GSM7574780_SBJ2': ('P2', 'CCRT'), 'GSM7574781_WGX': ('P3', 'TN'), 'GSM7574782_WGX2': ('P3', 'CCRT')}
genes_keep = set('ACKR2 TGFB1 TGFB2 TGFB3 TGFBR1 TGFBR2 MDM2 CDKN1A GADD45A DDB2 BBC3 CD8A CD8B CDKN2A TOX TIGIT LAG3 HIF1A EPAS1 VEGFA CA9 SLC2A1 LDHA COL1A1 COL1A2 DCN LUM FAP PDGFRA EPCAM KRT8 KRT18 KRT19 KRT14 KRT17 PAX8 CD3D CD3E TRBC1 NKG7 GNLY MS4A1 CD79A LST1 TYROBP FCER1G LILRB1 CTSS AIF1 PECAM1 VWF EMCN KDR KIT TPSAB1'.split())
parts = []; y = []; groups = []; samples = []
for sample, (patient, trt) in sample_info.items():
    m = mmread(MAT / f'{sample}.matrix.mtx.gz').tocsr()
    with gzip.open(MAT / f'{sample}.features.tsv.gz', 'rt', encoding='utf-8') as f:
        genes = [x.rstrip('\n').split('\t')[1] for x in f]
    idx = {g.upper(): i for i, g in enumerate(genes)}
    names = sorted(g for g in genes_keep if g in idx)
    x = m[[idx[g] for g in names], :].tocsr()
    lib = np.asarray(m.sum(axis=0)).ravel()
    x = x.multiply(1e4 / np.maximum(lib, 1)).tocsr(); x.data = np.log1p(x.data)
    parts.append(x.T); y += [1 if trt == 'CCRT' else 0] * x.shape[1]; groups += [patient] * x.shape[1]; samples += [sample] * x.shape[1]
X = vstack(parts).tocsr(); y = np.array(y); groups = np.array(groups); samples = np.array(samples)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
def mdl(): return make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=300, C=0.1))
a=[]
for tr,te in StratifiedKFold(5,shuffle=True,random_state=0).split(X,y):
    a.append(roc_auc_score(y[te], mdl().fit(X[tr],y[tr]).predict_proba(X[te])[:,1]))
b={}
for tr,te in LeaveOneGroupOut().split(X,y,groups):
    b[groups[te][0]]=roc_auc_score(y[te], mdl().fit(X[tr],y[tr]).predict_proba(X[te])[:,1])
# sample-level split control: hold out one sample (not patient) from each label? cells-of-sample split
res={'model':'logistic regression on 55 genes','random_cell_5fold_auc_mean':float(np.mean(a)),'random_cell_5fold_auc_folds':[float(x) for x in a],'leave_one_patient_out_auc':{k:float(v) for k,v in b.items()},'leave_one_patient_out_auc_mean':float(np.mean(list(b.values())))}
Path(OUT/'real_data_random_vs_patient_cv.json').write_text(json.dumps(res,indent=2)); print(json.dumps(res,indent=2))
