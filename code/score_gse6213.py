import csv, gzip, io, json
from pathlib import Path
import pandas as pd

ROOT=(Path(__file__).resolve().parents[1] / 'work' / 'validation')
PROGRAMS={'ACKR2_TGFb':['ACKR2','TGFB1','TGFB2','TGFB3','TGFBR1','TGFBR2'],'MDM2_DDR':['MDM2','CDKN1A','GADD45A','DDB2','BBC3'],'Tcell_senescence':['CD8A','CD8B','CDKN2A','TOX','TIGIT','LAG3'],'Hypoxia':['HIF1A','EPAS1','VEGFA','CA9','SLC2A1','LDHA'],'CAF_ECM':['COL1A1','COL1A2','DCN','LUM','FAP','PDGFRA']}

mapping={}
with gzip.open(ROOT/'GPL2895.annot.gz','rt',errors='replace') as f:
    for line in f:
        if line.startswith('!platform_table_begin'):
            header=next(f).rstrip('\n').split('\t'); ii=header.index('ID'); si=header.index('Gene symbol'); break
    for line in f:
        if line.startswith('!platform_table_end'): break
        p=line.rstrip('\n').split('\t')
        if len(p)>max(ii,si):
            sym=p[si].split(' /// ')[0].strip()
            if sym and sym!='---': mapping[p[ii]]=sym

path=ROOT/'GSE6213_series_matrix.txt.gz'
with gzip.open(path,'rt',errors='replace') as f: lines=f.readlines()
begin=next(i for i,x in enumerate(lines) if x.startswith('!series_matrix_table_begin'))
end=next(i for i in range(begin+1,len(lines)) if lines[i].startswith('!series_matrix_table_end'))
df=pd.read_csv(io.StringIO(''.join(lines[begin+1:end])),sep='\t')
df.rename(columns={df.columns[0]:'probe'},inplace=True); df['probe']=df['probe'].astype(str); df['gene']=df['probe'].map(mapping); df=df[df.gene.notna()]
samples=[c for c in df.columns if c not in ['probe','gene']]; expr=df.groupby('gene')[samples].mean()
classes={}
for line in lines[:begin]:
    if line.startswith('!Sample_description') and 'during treatment' in line:
        vals=[x.strip('"') for x in line.rstrip('\n').split('\t')[1:]]
        classes.update({s:v.split(':',1)[-1].strip() for s,v in zip(samples,vals)})
rows=[]
for s in samples:
    r={'sample':s,'tissue_type':classes.get(s,'')}
    for name,genes in PROGRAMS.items():
        g=[x for x in genes if x in expr.index]; r[name]=float(expr.loc[g,s].mean()) if g else None
    rows.append(r)
out=ROOT/'GSE6213_program_scores.csv'; pd.DataFrame(rows).to_csv(out,index=False)
print(json.dumps({'n_probes':len(df),'n_genes':len(expr),'n_samples':len(samples),'tissue_counts':pd.Series([r['tissue_type'] for r in rows]).value_counts().to_dict(),'output':str(out)},indent=2))
