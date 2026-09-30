import os
from pathlib import Path as _P
_REPO=_P(__file__).resolve().parents[1]
os.chdir(_REPO/'work')
import json, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
s=json.load(open(_REPO/'results'/'leakage_simulation.json'))['results']; r=json.load(open(_REPO/'results'/'gse236738_random_vs_patient_cv.json'))
fig,ax=plt.subplots(1,2,figsize=(9,3.6),gridspec_kw={'width_ratios':[1.3,1]})
for sc,ls in [('no treatment effect','-'),('small true effect','--')]:
    d=[x for x in s if x['scenario']==sc]; n=[x['patients'] for x in d]
    ax[0].errorbar(n,[x['cell_level_cv_auc_mean'] for x in d],[x['cell_level_cv_auc_sd'] for x in d],color='#c0392b',ls=ls,marker='o',capsize=2,label=f'random cell split, {sc}')
    ax[0].errorbar(n,[x['leave_one_patient_out_auc_mean'] for x in d],[x['leave_one_patient_out_auc_sd'] for x in d],color='#2c6fbb',ls=ls,marker='s',capsize=2,label=f'leave-one-patient-out, {sc}')
ax[0].axhline(0.5,color='grey',lw=0.8,ls=':'); ax[0].set_xscale('log'); ax[0].minorticks_off(); ax[0].set_xticks([3,5,10,20]); ax[0].set_xticklabels([3,5,10,20])
ax[0].set_xlabel('patients (paired pre/post)'); ax[0].set_ylabel('AUC'); ax[0].set_ylim(0.3,1.05); ax[0].set_title('a  Simulation with batch offsets',loc='left',fontsize=10); ax[0].legend(fontsize=6.2,loc='lower left')
lo=r['leave_one_patient_out_auc']; v=[r['random_cell_5fold_auc_mean']]+list(lo.values())+[r['leave_one_patient_out_auc_mean']]
lab=['random\ncell 5-fold','P1','P2','P3','LOPO\nmean']; col=['#c0392b','#9bbbe0','#9bbbe0','#9bbbe0','#2c6fbb']
ax[1].bar(lab,v,color=col); ax[1].axhline(0.5,color='grey',lw=0.8,ls=':'); ax[1].set_ylim(0.4,1.0)
for i,x in enumerate(v): ax[1].text(i,x+0.01,f'{x:.2f}',ha='center',fontsize=8)
ax[1].set_title('b  GSE236738 (55 genes)',loc='left',fontsize=10); ax[1].set_ylabel('AUC')
plt.tight_layout(); plt.savefig(_REPO/'results'/'figures'/'fig_leakage.png',dpi=200)
