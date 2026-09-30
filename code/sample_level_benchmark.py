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

ROOT=(Path(__file__).resolve().parents[1] / 'work')
df=pd.read_csv(ROOT/'analysis_gse236738'/'cell_annotations.csv')
programs=['ACKR2_TGFb','MDM2_DDR','Tcell_senescence','Hypoxia','CAF_ECM']
means=df.groupby('sample')[programs].mean()
props=pd.crosstab(df['sample'],df['cell_type'],normalize='index')
X=means.join(props,how='left').fillna(0)
meta=df[['sample','patient','treatment']].drop_duplicates().set_index('sample').loc[X.index]
y=(meta.treatment=='CCRT').astype(int).to_numpy(); groups=meta.patient.to_numpy()
logo=LeaveOneGroupOut(); results=[]
models={'logistic_regression':make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,C=1.0)), 'random_forest':RandomForestClassifier(n_estimators=150,min_samples_leaf=1,class_weight='balanced',random_state=42,n_jobs=2)}
for name,template in models.items():
    for train,test in logo.split(X,y,groups):
        model=template; model.fit(X.iloc[train],y[train]); p=model.predict_proba(X.iloc[test])[:,1]
        results.append({'model':name,'held_out_patient':groups[test][0],'auc':float(roc_auc_score(y[test],p))})
rng=np.random.default_rng(42); null=[]
for rep in range(500):
    yp=rng.permutation(y); fold=[]
    for train,test in logo.split(X,yp,groups):
        if len(np.unique(yp[test])) < 2:
            continue
        model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,C=1.0)); model.fit(X.iloc[train],yp[train]); fold.append(roc_auc_score(yp[test],model.predict_proba(X.iloc[test])[:,1]))
    if fold:
        null.append(float(np.mean(fold)))
summary={'n_samples':int(len(X)),'n_patients':int(meta.patient.nunique()),'n_features':int(X.shape[1]),'features':list(X.columns),'model_comparison':results,'observed_mean_auc':{m:float(np.mean([r['auc'] for r in results if r['model']==m])) for m in models},'permutation_null_mean':float(np.mean(null)),'permutation_null_95th':float(np.quantile(null,.95)),'caveat':'Only six biological samples are available; this is a leakage-control benchmark, not clinical prediction.'}
out=ROOT/'analysis_gse236738'/'sample_level_benchmark.json'; out.write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
