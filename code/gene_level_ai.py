import csv, gzip, json
from pathlib import Path
import numpy as np
from scipy.io import mmread
from scipy.sparse import vstack
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

ROOT=(Path(__file__).resolve().parents[1] / 'work'); MAT=ROOT/'GSE236738_matrix'; OUT=ROOT/'analysis_gse236738'
sample_info={'GSM7574777_FXW':('P1','TN'),'GSM7574778_FXW2':('P1','CCRT'),'GSM7574779_SBJ':('P2','TN'),'GSM7574780_SBJ2':('P2','CCRT'),'GSM7574781_WGX':('P3','TN'),'GSM7574782_WGX2':('P3','CCRT')}
genes_keep=set('ACKR2 TGFB1 TGFB2 TGFB3 TGFBR1 TGFBR2 MDM2 CDKN1A GADD45A DDB2 BBC3 CD8A CD8B CDKN2A TOX TIGIT LAG3 HIF1A EPAS1 VEGFA CA9 SLC2A1 LDHA COL1A1 COL1A2 DCN LUM FAP PDGFRA EPCAM KRT8 KRT18 KRT19 KRT14 KRT17 PAX8 CD3D CD3E TRBC1 NKG7 GNLY MS4A1 CD79A LST1 TYROBP FCER1G LILRB1 CTSS AIF1 PECAM1 VWF EMCN KDR KIT TPSAB1'.split())
parts=[]; labels=[]; groups=[]; feature_names=None
for sample,(patient,treatment) in sample_info.items():
 m=mmread(MAT/f'{sample}.matrix.mtx.gz').tocsr()
 with gzip.open(MAT/f'{sample}.features.tsv.gz','rt',encoding='utf-8') as f: genes=[x.rstrip('\n').split('\t')[1] for x in f]
 idx={g.upper():i for i,g in enumerate(genes)}; names=sorted(g for g in genes_keep if g in idx); feature_names=names
 rows=[idx[g] for g in names]; x=m[rows,:].tocsr(); lib=np.asarray(m.sum(axis=0)).ravel(); x=x.multiply(1e4/np.maximum(lib,1)); x.data=np.log1p(x.data); parts.append(x.T); labels.extend([1 if treatment=='CCRT' else 0]*x.shape[1]); groups.extend([patient]*x.shape[1])
X=vstack(parts).tocsr(); y=np.array(labels); groups=np.array(groups); X=StandardScaler(with_mean=False).fit_transform(X)
svd=TruncatedSVD(n_components=20,random_state=42); Z=svd.fit_transform(X); logo=LeaveOneGroupOut(); folds=[]
for train,test in logo.split(Z,y,groups):
 clf=RandomForestClassifier(n_estimators=300,min_samples_leaf=10,class_weight='balanced_subsample',random_state=42,n_jobs=-1); clf.fit(Z[train],y[train]); p=clf.predict_proba(Z[test])[:,1]; folds.append({'patient':groups[test][0],'auc':float(roc_auc_score(y[test],p)),'n_test':int(len(test))})
meta={'n_cells':int(X.shape[0]),'n_features':int(X.shape[1]),'latent_dimensions':20,'variance_explained':float(svd.explained_variance_ratio_.sum()),'features':feature_names,'folds':folds,'warning':'Gene-level baseline uses a curated marker/resistance panel; final model should expand to data-driven variable genes.'}
(OUT/'gene_level_ai_baseline.json').write_text(json.dumps(meta,indent=2),encoding='utf-8'); print(json.dumps(meta,indent=2))
