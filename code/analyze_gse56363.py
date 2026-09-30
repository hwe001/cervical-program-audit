import gzip,itertools,json
from pathlib import Path
import numpy as np
PROGRAMS={'ACKR2_TGFb':['ACKR2','TGFB1','TGFB2','TGFB3','TGFBR1','TGFBR2'],'MDM2_DDR':['MDM2','CDKN1A','GADD45A','DDB2','BBC3'],'Tcell_senescence':['CD8A','CD8B','CDKN2A','TOX','TIGIT','LAG3'],'Hypoxia':['HIF1A','EPAS1','VEGFA','CA9','SLC2A1','LDHA'],'CAF_ECM':['COL1A1','COL1A2','DCN','LUM','FAP','PDGFRA']}
root=Path('work/data_gse56363'); mapping={}; f=gzip.open(root/'GSE56363_family.soft.gz','rt',encoding='utf8')
for line in f:
    if line.startswith('!platform_table_begin'):
        next(f)
        for row in f:
            if row.startswith('!platform_table_end'): break
            p=row.rstrip('\n').split('\t')
            if len(p)>9 and p[9]: mapping[p[0]]=p[9].upper()
        break
f.close(); samples=[]; rows=[]; inmat=False
with gzip.open(root/'GSE56363_series_matrix.txt.gz','rt',encoding='utf8') as f:
    for line in f:
        if line.startswith('!series_matrix_table_begin'):
            samples=[x.strip('"') for x in next(f).rstrip().split('\t')[1:]]; inmat=True; continue
        if line.startswith('!series_matrix_table_end'): break
        if inmat:
            p=line.rstrip().split('\t'); rows.append((p[0].strip('"'),[float(x.strip('"')) for x in p[1:]]))
wanted=set(sum(PROGRAMS.values(),[])); expr={}
for probe,vals in rows:
    g=mapping.get(probe)
    if g in wanted: expr.setdefault(g,[]).append(vals)
gene={g:np.mean(v,axis=0) for g,v in expr.items()}; scores={}
for name,gs in PROGRAMS.items():
    z=[]
    for g in gs:
        if g in gene:
            a=gene[g]; z.append((a-a.mean())/(a.std() or 1))
    scores[name]=np.mean(z,axis=0) if z else np.zeros(len(samples))
cr=np.arange(12); ncr=np.arange(12,21); result={'n_samples':21,'complete_response':12,'non_complete_response':9,'genes_found':sorted(gene),'programs':{}}
for name,a in scores.items():
    obs=float(a[ncr].mean()-a[cr].mean()); extreme=0; total=0
    for comb in itertools.combinations(range(21),9):
        g=np.zeros(21,bool); g[list(comb)]=1; d=float(a[g].mean()-a[~g].mean()); extreme += abs(d)>=abs(obs); total += 1
    result['programs'][name]={'genes_found':[g for g in PROGRAMS[name] if g in gene],'mean_difference_NCR_minus_CR':obs,'mean_CR':float(a[cr].mean()),'mean_NCR':float(a[ncr].mean()),'exact_permutation_p':(1+extreme)/(total+1)}
out=Path('work/analysis_gse56363'); out.mkdir(parents=True,exist_ok=True)
(out/'program_response_summary.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
