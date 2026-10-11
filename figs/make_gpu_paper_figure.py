"""Rebuild the two paper figures from the bundled reference summaries."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import t
from plot_style import use_style, figure, save, TEXT, PALETTE, BASELINE

P = Path(__file__).resolve().parent
import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', type=Path, default=P)
args = parser.parse_args()
OUTPUT = args.output_dir.resolve()
OUTPUT.mkdir(parents=True, exist_ok=True)
summary = json.loads((P/'data/gpu_summary.json').read_text())
conditions = ['easy', 'recovery', 'friction_shift', 'damping_shift']
labels = ['Easy alignment', 'Recovery', 'Changed friction', 'Changed damping']
use_style()
fig, ax = figure(width=TEXT, ratio=.28)
for i, condition in enumerate(conditions):
    row = next(g for g in summary['groups'] if g['kind']=='world_model'
               and g['width']==512 and g['horizon']==32 and g['condition']==condition)
    lo, hi = [100*x for x in row['paired_gain_over_scorer_95ci']]
    delta = 100*row['paired_gain_over_scorer']
    ax.plot([lo, hi], [i, i], color=PALETTE['blue'], linewidth=1.3)
    ax.plot(delta, i, 'o', color=PALETTE['blue'], markersize=4)
ax.axvline(0, color=BASELINE, linewidth=.8, linestyle='--')
ax.set_yticks(range(4), labels)
ax.set_ylim(3.5,-.5)
ax.set_xlim(-26,12)
ax.set_xticks([-20,-10,0,10])
ax.set_xlabel('World model minus matched scorer (success percentage points)')
ax.set_ylabel('Scenario')
ax.grid(axis='y', visible=False)
ax.tick_params(axis='y',length=0)
save(fig, str(OUTPUT/'gpu_primary_comparison'))

# Refresh the retained navigation figure with the same font as the GPU figure.
# Values and interval definition are unchanged from the original paper.
navigation = json.loads((P/'data/navigation_summary.json').read_text())
methods = ['baseline', 'world_model_a', 'world_model_b', 'direct_scorer']
names = ['Policy', 'WM A', 'WM B', 'Scorer']
colors = ['#757575', PALETTE['blue'], PALETTE['cyan'], PALETTE['orange']]
scenarios = ['open', 'detour', 'key_door', 'rotated_controls']
fig, ax = figure(width=TEXT, ratio=.34)
x = np.arange(4)
for j, method in enumerate(methods):
    values, errors = [], []
    for scenario in scenarios:
        row = next(r for r in navigation if r['scenario']==scenario and r['shots']==0 and r['method']==method)
        a = np.array(row['success_by_training_seed'])*100
        mean = a.mean()
        error = t.ppf(.975,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a))
        values.append(mean)
        errors.append([min(error,mean),min(error,100-mean)])
    bars = ax.bar(x+(j-1.5)*.19, values, .19, label=names[j], color=colors[j],
                  edgecolor='black', linewidth=.3, yerr=np.array(errors).T,
                  error_kw={'elinewidth':.5,'capsize':1})
    ax.bar_label(bars, fmt='%.0f', fontsize=5.5, padding=2)
ax.set_xticks(x, ['Open','Detours','Key + door','Changed controls'], fontsize=7)
ax.set_xlabel('Situation (zero adaptation episodes)',fontsize=7)
ax.set_ylabel('Success (%)',fontsize=7)
ax.set_ylim(0,115)
ax.set_yticks([0,50,100])
ax.legend(loc='upper center',bbox_to_anchor=(.5,1.25),ncol=4,fontsize=6.5)
save(fig, str(OUTPUT/'scenario_success'))

for name in ['scenario_success.svg','gpu_primary_comparison.svg']:
    path = OUTPUT/name
    path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
