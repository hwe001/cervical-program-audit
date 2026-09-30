import json
from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr
root=(Path(__file__).resolve().parents[1] / 'work' / 'validation')
programs=['ACKR2_TGFb','MDM2_DDR','Tcell_senescence','Hypoxia','CAF_ECM']
prog=pd.read_csv(root/'GSE63514_program_scores.csv')
stage={'normal cervical epithelium':0,'cervical intraepithelial neoplasm, low grade lesion':1,'cervical intraepithelial neoplasm, moderate grade lesion':2,'cervical intraepithelial neoplasm, high grade lesion':3,'cervical squamous epithelial cancer':4}
prog['stage']=prog.tissue_type.map(stage); progression=[]
for p in programs:
 r,pv=spearmanr(prog.stage,prog[p]); progression.append({'program':p,'rho':float(r),'p':float(pv),'n':len(prog)})
rt=pd.read_csv(root/'GSE6213_program_scores.csv'); rt['modality']=rt.tissue_type.str.replace('prior to treatment, ','',regex=False).str.replace('during treatment, ','',regex=False); rt['timepoint']=rt.tissue_type.str.extract(r'^(prior to|during)',expand=False); treatment=[]
for modality,g in rt.groupby('modality'):
 pre=g[g.timepoint=='prior to']; post=g[g.timepoint=='during']
 if len(pre)==0 or len(post)==0: continue
 for p in programs:
  sd=pd.concat([pre[p],post[p]]).std() or 1; treatment.append({'modality':modality,'program':p,'n_pre':len(pre),'n_during':len(post),'standardized_delta':float((post[p].mean()-pre[p].mean())/sd)})
out={'GSE63514_progression':progression,'GSE6213_treatment':treatment}; (root/'cross_cohort_program_benchmark.json').write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2))
