"""Paired post-run analysis; two-way bootstrap over training seeds and reset IDs."""
from pathlib import Path
import json
import numpy as np
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results-root', type=Path, default=Path(__file__).resolve().parents[1]/'results')
parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parents[1]/'outputs/analysis')
args=parser.parse_args()
ROOT=args.results_root.resolve()
OUT=args.output_dir.resolve()
if OUT == ROOT or ROOT in OUT.parents:
    parser.error('Analysis output must be separate from input results')
OUT.mkdir(parents=True,exist_ok=True)
rows=[json.loads(x) for x in (ROOT/'navigation-study/episodes.jsonl').read_text().splitlines()]
rows=[r for r in rows if r['split']=='test']
gates=json.loads((ROOT/'navigation-study/gate_results.json').read_text())
methods=['baseline','world_model_a','world_model_b','direct_scorer']
scenarios=['open','detour','key_door','rotated_controls']
seeds=[11,22,33];shots=[0,1,5];resets=sorted({r['reset_seed'] for r in rows})
lookup={(r['training_seed'],r['scenario'],r['shots'],r['reset_seed'],r['method']):r for r in rows}
glookup={(r['training_seed'],r['scenario'],r['shots'],r['reset_seed'],r['lambda_per_second']):r for r in gates}
rng=np.random.default_rng(90210)
seed_draws=rng.integers(0,3,(10000,3));reset_draws=rng.integers(0,len(resets),(10000,len(resets)))
def stats(a):
    # Shape is seed x reset. Conditions, when present, are averaged beforehand.
    estimates=a[seed_draws[:,:,None],reset_draws[:,None,:]].mean((1,2))
    return {'mean':float(a.mean()),'bootstrap_95_ci':np.quantile(estimates,[.025,.975]).tolist()}
def matrix(sc,shot,m,key):
    return np.array([[lookup[s,sc,shot,r,m][key] for r in resets] for s in seeds],float)
comparisons=[]
for sc in scenarios:
    for shot in shots:
        for m in methods[1:]:
            gain=matrix(sc,shot,m,'success')-matrix(sc,shot,'baseline','success')
            cost=matrix(sc,shot,m,'inference_seconds')-matrix(sc,shot,'baseline','inference_seconds')
            comparisons.append({'scenario':sc,'shots':shot,'method':m,'success_gain':stats(gain),'extra_inference_seconds':stats(cost),
                'break_even_lambda_point_estimate':float(gain.mean()/cost.mean()) if gain.mean()>0 and cost.mean()>0 else None})
gate_comparisons=[]
for lam in [0,.1,1,10]:
    for m in methods:
        diffs=[]
        for sc in scenarios:
            for shot in shots:
                g=np.array([[glookup[s,sc,shot,r,lam]['utility'] for r in resets] for s in seeds])
                fixed=matrix(sc,shot,m,'success')-lam*matrix(sc,shot,m,'inference_seconds')
                diffs.append(g-fixed)
        gate_comparisons.append({'lambda_per_second':lam,'compared_to':m,'utility_gain':stats(np.mean(diffs,axis=0))})
out={'bootstrap_replicates':10000,'bootstrap_seed':90210,
     'uncertainty_definition':'Paired crossed bootstrap: resample 3 training seeds and 20 reset IDs, preserving pairing across methods, scenarios, and adaptation levels. Scenarios and shot levels are fixed. Gate intervals condition on the fitted validation gates; no gate refitting. Small seed count limits coverage.',
     'scenario_comparisons':comparisons,'gate_comparisons':gate_comparisons}
(OUT/'navigation_paired_analysis.json').write_text(json.dumps(out,indent=2)+'\n')
for row in comparisons:
    if row['shots']==0 and row['method']!='direct_scorer':print(json.dumps(row))
for row in gate_comparisons:
    if row['lambda_per_second']==1:print(json.dumps(row))
