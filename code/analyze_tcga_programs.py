import gzip,csv,json,math
from pathlib import Path
from urllib.request import urlopen,Request
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
PROGRAMS={'ACKR2_TGFb':['ACKR2','TGFB1','TGFB2','TGFB3','TGFBR1','TGFBR2'],'MDM2_DDR':['MDM2','CDKN1A','GADD45A','DDB2','BBC3'],'Tcell_senescence':['CD8A','CD8B','CDKN2A','TOX','TIGIT','LAG3'],'Hypoxia':['HIF1A','EPAS1','VEGFA','CA9','SLC2A1','LDHA'],'CAF_ECM':['COL1A1','COL1A2','DCN','LUM','FAP','PDGFRA']}
def logrank(time,event,group):
    order=np.argsort(time); t=time[order]; e=event[order]; g=group[order]; u=0.; v=0.
    for tt in np.unique(t[e==1]):
        risk=t>=tt; deaths=(t==tt)&(e==1); d=int(deaths.sum()); n=int(risk.sum()); ng=int((risk&g).sum()); dg=int((deaths&g).sum())
        if n>0: u += dg-d*ng/n; v += d*(ng/n)*(1-ng/n)*((n-d)/(n-1) if n>1 else 0)
    chi=float((u*u)/v) if v>0 else 0.; p=float(math.erfc(math.sqrt(chi/2)))
    return {'chi2':chi,'p_value':p,'observed_minus_expected_high':float(u)}
root=Path('work/data_tcga_cesc'); wanted=set(sum(PROGRAMS.values(),[])); expr={}; samples=None
def fetch_symbol(symbol):
    try:
        req=Request('https://rest.ensembl.org/xrefs/symbol/homo_sapiens/'+symbol+'?content-type=application/json',headers={'Accept':'application/json'})
        hits=json.load(urlopen(req,timeout=8)); return symbol,(hits[0]['id'].upper() if hits else None)
    except Exception: return symbol,None
ens_to_symbol={v.upper():k for k,v in json.load(open('work/data_gse297038/symbol2ensg.json')).items()}
with ThreadPoolExecutor(max_workers=12) as ex:
    pass
with gzip.open(root/'TCGA-CESC.star_tpm.tsv.gz','rt',encoding='utf8') as f:
    for line in f:
        if line.startswith('#'): continue
        p=line.rstrip('\n').split('\t')
        if samples is None: samples=p[1:]; continue
        gene=p[0].split('|')[0].split('.')[0].upper()
        if gene in ens_to_symbol: expr[ens_to_symbol[gene]]=np.array([float(x) if x not in ('NA','') else 0. for x in p[1:]],float)
print('samples',len(samples),'markers',len(expr))
clinical=json.loads((root/'clinical.json').read_text())['data']['hits']; clin={x['submitter_id']:x for x in clinical}
keep=[]; time=[]; event=[]
for i,s in enumerate(samples):
    patient=s[:12]; c=clin.get(patient)
    if not c: continue
    d=c.get('demographic',{}); dx=(c.get('diagnoses') or [{}])[0]
    t=d.get('days_to_death') or dx.get('days_to_last_follow_up')
    if t is None: continue
    keep.append(i); time.append(float(t)); event.append(1 if str(d.get('vital_status','')).lower()=='dead' else 0)
time=np.array(time); event=np.array(event); result={'n_with_expression_and_outcome':len(keep),'events':int(event.sum()),'programs':{}}
for name,gs in PROGRAMS.items():
    arr=np.vstack([np.log2(expr[g][keep]+1) for g in gs if g in expr]).mean(0); med=float(np.median(arr)); hi=arr>=med
    result['programs'][name]={'genes_found':[g for g in gs if g in expr],'median_score':med,'n_high':int(hi.sum()),'events_high':int(event[hi].sum()),'events_low':int(event[~hi].sum()),'median_time_high_days':float(np.median(time[hi])),'median_time_low_days':float(np.median(time[~hi])),'logrank':logrank(time,event,hi)}
out=Path('work/analysis_tcga_cesc'); out.mkdir(parents=True,exist_ok=True); (out/'program_outcome_summary.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
