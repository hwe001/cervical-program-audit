"""Fig. 2: (a) leakage simulation, (b) GSE236738 random vs leave-one-patient-out, (c) power of leave-one-patient-out validation."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[1] / 'results'
s = json.load(open(R / 'leakage_simulation.json'))['results']
r = json.load(open(R / 'gse236738_random_vs_patient_cv.json'))
pw = json.load(open(R / 'power_simulation.json'))['results']
fig, ax = plt.subplots(1, 3, figsize=(12, 3.7), gridspec_kw={'width_ratios': [1.25, 0.9, 1.15]})
for sc, ls in [('no treatment effect', '-'), ('small true effect', '--')]:
    d = [x for x in s if x['scenario'] == sc]; n = [x['patients'] for x in d]
    ax[0].errorbar(n, [x['cell_level_cv_auc_mean'] for x in d], [x['cell_level_cv_auc_sd'] for x in d], color='#c0392b', ls=ls, marker='o', capsize=2, label='random cell split, ' + sc)
    ax[0].errorbar(n, [x['leave_one_patient_out_auc_mean'] for x in d], [x['leave_one_patient_out_auc_sd'] for x in d], color='#2c6fbb', ls=ls, marker='s', capsize=2, label='leave-one-patient-out, ' + sc)
ax[0].axhline(0.5, color='grey', lw=0.8, ls=':'); ax[0].set_xscale('log'); ax[0].minorticks_off(); ax[0].set_xticks([3, 5, 10, 20]); ax[0].set_xticklabels([3, 5, 10, 20])
ax[0].set_xlabel('patients (paired pre/post)'); ax[0].set_ylabel('AUC'); ax[0].set_ylim(0.3, 1.05); ax[0].set_title('a  Leakage simulation', loc='left', fontsize=10); ax[0].legend(fontsize=5.8, loc='lower left')
lo = r['leave_one_patient_out_auc']; v = [r['random_cell_5fold_auc_mean']] + list(lo.values()) + [r['leave_one_patient_out_auc_mean']]
lab = ['random\ncell 5-fold', 'P1', 'P2', 'P3', 'LOPO\nmean']; col = ['#c0392b', '#9bbbe0', '#9bbbe0', '#9bbbe0', '#2c6fbb']
ax[1].bar(lab, v, color=col); ax[1].axhline(0.5, color='grey', lw=0.8, ls=':'); ax[1].set_ylim(0.4, 1.0); ax[1].tick_params(axis='x', labelsize=7)
for i, x in enumerate(v):
    ax[1].text(i, x + 0.01, f'{x:.2f}', ha='center', fontsize=8)
ax[1].set_title('b  GSE236738 (55 genes)', loc='left', fontsize=10); ax[1].set_ylabel('AUC')
cmap = plt.get_cmap('viridis')
effs = sorted(set(x['effect_sd_on_10_genes'] for x in pw))
for k, e in enumerate(effs):
    d = [x for x in pw if x['effect_sd_on_10_genes'] == e]
    ax[2].errorbar([x['patients'] for x in d], [x['lopo_auc_mean'] for x in d], [x['lopo_auc_sd'] for x in d], marker='o', capsize=2, color=cmap(k / max(1, len(effs) - 1) * 0.9), label=f'effect {e:g} SD')
ax[2].axhline(0.5, color='grey', lw=0.8, ls=':'); ax[2].set_xscale('log'); ax[2].minorticks_off(); ax[2].set_xticks([3, 5, 10, 20]); ax[2].set_xticklabels([3, 5, 10, 20])
ax[2].set_xlabel('patients (paired pre/post)'); ax[2].set_ylabel('leave-one-patient-out AUC'); ax[2].set_ylim(0.3, 1.02); ax[2].set_title('c  Power of patient-level validation', loc='left', fontsize=10); ax[2].legend(fontsize=6.5, loc='upper left')
plt.tight_layout(); plt.savefig(R / 'figures' / 'fig_leakage.png', dpi=200)
