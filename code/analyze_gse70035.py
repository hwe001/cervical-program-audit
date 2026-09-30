import gzip,itertools,json
from pathlib import Path
import numpy as np
ROOT=Path('work/external_public/GSE70035'); OUT=Path('work/analysis_gse70035'); OUT.mkdir(parents=True,exist_ok=True)
PROGRAMS={'ACKR2_TGFb':['ACKR2','TGFB1','TGFB2','TGFB3','TGFBR1','TGFBR2'],'MDM2_DDR':['MDM2','CDKN1A','GADD45A','DDB2','BBC3'],'Tcell_senescence':['CD8A','CD8B','CDKN2A','TOX','TIGIT','LAG3'],'Hypoxia':['HIF1A','EPAS1','VEGFA','CA9','SLC2A1','LDHA'],'CAF_ECM':['COL1A1','COL1A2','DCN','LUM','FAP','PDGFRA']}
wanted=set(sum(PROGRAMS.values(),[])); mapping={}
with gzip.open(ROOT/'GSE70035_family.soft.gz','rt',encoding='utf8',errors='replace') as f:
    in_table=False
    for line in f:
        if line.startswith('!platform_table_begin'): in_table=True; header=next(f).rstrip().split('\t'); continue
        if in_table and line.startswith('!platform_table_end'): break
        if in_table:
            p=line.rstrip('\n').split('\t')
            if len(p)>10:
                for g in p[10].split(' /// '):
                    g=g.strip().upper()
                    if g in wanted: mapping[p[0]]=g
samples=[]; rows={}
with gzip.open(ROOT/'GSE70035_series_matrix.txt.gz','rt',encoding='utf8',errors='replace') as f:
    inside=False
    for line in f:
        if line.startswith('!series_matrix_table_begin'):
            samples=next(f).rstrip().split('\t')[1:]; inside=True; continue
        if line.startswith('!series_matrix_table_end'): break
        if inside:
            p=line.rstrip('\n').split('\t'); rows[p[0].strip('"')]=np.array([float(x.strip('"')) for x in p[1:]])
gene={}
for probe,g in mapping.items():
    if probe in rows: gene.setdefault(g,[]).append(rows[probe])
gene={g:np.mean(v,axis=0) for g,v in gene.items()}
scores={}
for name,gs in PROGRAMS.items():
    z=[]
    for g in gs:
        if g in gene:
            x=gene[g]; z.append((x-x.mean())/(x.std() or 1))
    scores[name]=np.mean(z,axis=0)
labels=np.array(['R']*6+['NR']*6); r=labels=='R'; nr=~r
res={'dataset':'GSE70035','design':'pretreatment expression profiling; RECIST responder vs non-responder after neoadjuvant chemotherapy','n_samples':12,'n_responder':6,'n_non_responder':6,'genes_found':sorted(gene),'programs':{}}
for name,x in scores.items():
    obs=float(x[nr].mean()-x[r].mean()); ex=tot=0
    for idx in itertools.combinations(range(12),6):
        m=np.zeros(12,bool);m[list(idx)]=1;d=float(x[m].mean()-x[~m].mean());ex+=abs(d)>=abs(obs)-1e-15;tot+=1
    res['programs'][name]={'genes_found':[g for g in PROGRAMS[name] if g in gene],'mean_NR_minus_R':obs,'exact_permutation_p_two_sided':(1+ex)/(tot+1),'direction':'higher_NR' if obs>0 else 'higher_R'}
items=list(res['programs'].items()); order=sorted(range(5),key=lambda i:items[i][1]['exact_permutation_p_two_sided'])
for rank,i in enumerate(order,1): items[i][1]['BH_q_across_five_programs']=min(1,items[i][1]['exact_permutation_p_two_sided']*5/rank)
(OUT/'program_response_summary.json').write_text(json.dumps(res,indent=2),encoding='utf8'); print(json.dumps(res,indent=2))
