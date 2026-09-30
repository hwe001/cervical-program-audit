import os
from pathlib import Path as _P
_REPO=_P(__file__).resolve().parents[1]
os.chdir(_REPO/'work')
import json, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
T=pd.read_csv(_REPO/'results'/'cross_cohort_table.csv'); P=['ACKR2_TGFb','MDM2_DDR','Tcell_senescence','Hypoxia','CAF_ECM']
PL=['ACKR2/TGF-β','MDM2/DDR','T-cell/\nCDKN2A','Hypoxia','CAF/ECM']
tc=None
b=json.load(open('validation/cross_cohort_program_benchmark.json'))['GSE6213_treatment']
g6213={r['program']:r['standardized_delta'] for r in b if r['modality']=='chemoradiotherapy'}
sc={'ACKR2_TGFb':'2/3 up','MDM2_DDR':'3/3 up','Tcell_senescence':'3/3 down','Hypoxia':'1/3 up','CAF_ECM':'2/3 up'}
cols=[('GSE236738\n3 pts, pre/post',None),('GSE297038\n12 pairs, CRT','GSE297038'),('GSE6213\nCRT, unpaired','g6213'),('GSE63514\nstage, n=128','GSE63514'),('GSE168009\n4 vs 5','GSE168009'),('GSE56363\n9 vs 12','GSE56363'),('GSE70035\n6 vs 6, NAC','GSE70035'),('TCGA-CESC\nsurvival, n=291','TCGA-CESC')]
M=np.full((5,len(cols)),np.nan); txt=[['']*len(cols) for _ in P]; sig=np.zeros((5,len(cols)),bool)
for j,(lab,c) in enumerate(cols):
    for i,p in enumerate(P):
        if c is None: txt[i][j]=sc[p]; M[i,j]=0.0 ; continue
        if c=='g6213': txt[i][j]='%+.2f'%g6213[p]; M[i,j]=np.nan; continue
        r=T[(T.cohort==c)&(T.program==p)].iloc[0]
        sgn=1.0
        if c=='TCGA-CESC': sgn=np.sign(np.log(r.effect))
        else: sgn=np.sign(r.effect)
        M[i,j]=sgn*-np.log10(r.p); sig[i,j]=r.BH_q<0.05
        txt[i][j]=('HR %.2f'%r.effect if c=='TCGA-CESC' else '%+.2f'%r.effect)+('\np=%.2g'%r.p)
fig,ax=plt.subplots(figsize=(11,4.4))
cm=plt.get_cmap('RdBu_r').copy(); cm.set_bad('#e8e8e8'); im=ax.imshow(np.ma.masked_invalid(M),cmap=cm,vmin=-4,vmax=4,aspect='auto')
for i in range(5):
    for j in range(len(cols)):
        ax.text(j,i,txt[i][j],ha='center',va='center',fontsize=7.5,fontweight='bold' if sig[i,j] else 'normal',color='black')
        if sig[i,j]: ax.add_patch(plt.Rectangle((j-.5,i-.5),1,1,fill=False,ec='black',lw=2))
ax.set_xticks(range(len(cols))); ax.set_xticklabels([c[0] for c in cols],fontsize=7.5); ax.set_yticks(range(5)); ax.set_yticklabels(PL,fontsize=8.5)
ax.xaxis.tick_top()
for x in (2.5,3.5): ax.axvline(x,color='k',lw=1.2)
ax.text(1,5.05,'Treatment exposure',ha='center',va='top',fontsize=8,transform=ax.transData); ax.text(3,5.05,'Progression',ha='center',va='top',fontsize=8)
ax.text(5.5,5.05,'Outcome (positive = poorer); TCGA = prognosis',ha='center',va='top',fontsize=8)
ax.set_ylim(5.4,-0.5)
cb=plt.colorbar(im,ax=ax,fraction=0.025,pad=0.01); cb.set_label('signed -log10 p (capped at 4)',fontsize=7.5)
plt.tight_layout(); plt.savefig(_REPO/'results'/'figures'/'fig_heatmap.png',dpi=200)
# schematic
fig,ax=plt.subplots(figsize=(10,4.6)); ax.axis('off'); ax.set_xlim(0,100); ax.set_ylim(0,50)
def box(x,y,w,h,t,fc,fs=8.5,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.4',fc=fc,ec='#444',lw=1)); ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=fs,fontweight='bold' if bold else 'normal',wrap=True)
box(1,17,17,14,'Five locked\nprograms\n(literature-derived,\n29 genes; no refitting)','#f2f2f2',8.5,True)
Q=[('1  Does it change with treatment?','GSE236738 (3 pts), GSE297038 (12 pairs), GSE6213','MDM2/DDR up (12/12 pairs)\nT-cell senescence down (11/12)\nverdict: exposure marker','#dbe9f6',31),
   ('2  Does it track tumor progression?','GSE63514 (n=128, normal/CIN/cancer)','Hypoxia rho 0.64\nT-cell senescence rho 0.48\nverdict: progression marker','#e6f2dc',16),
   ('3  Does it separate clinical outcome?','GSE168009, GSE56363, GSE70035 (n=9, 21, 12); TCGA survival','No program significant in the three\noutcome cohorts (lowest q=0.20); signs conflict.\nTCGA (mixed treatment): hypoxia HR 1.87\nverdict: not reproducible for CCRT outcome','#f8e0dc',1)]
for t,c,v,fc,y in Q:
    box(25,y,38,12,t+'\n'+c,fc,7.5); box(68,y,31,12,v,fc,7)
    ax.annotate('',xy=(24.5,y+6),xytext=(19,24),arrowprops=dict(arrowstyle='->',color='#444'))
ax.text(50,48,'Safeguards: patients (not cells) as unit; in-fold preprocessing; exact tests; correction across all cohorts',ha='center',fontsize=8,style='italic')
plt.tight_layout(); plt.savefig(_REPO/'results'/'figures'/'fig_schematic.png',dpi=200)
