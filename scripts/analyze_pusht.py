"""Paired analysis of completed Push-T variants only; resample reset seeds as clusters."""
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
P=ROOT/'pusht-pilot'
base=json.loads((P/'baseline/episodes.json').read_text())
key=lambda r:(r['env_seed'],r['condition'])
b={key(r):r for r in base};seeds=sorted({r['env_seed'] for r in base})
rng=np.random.default_rng(42);draws=rng.integers(0,len(seeds),(10000,len(seeds)))
def interval(v):
    return {'mean':float(v.mean()),'paired_reset_cluster_bootstrap_95_ci':np.quantile(v[draws].mean(1),[.025,.975]).tolist()}
summary=[]
for method in ['world_model_a','world_model','direct_scorer']:
    if not (P/method/'summary.json').exists():continue
    rows=json.loads((P/method/'episodes.json').read_text());a={key(r):r for r in rows};assert set(a)==set(b)
    out={'method':method,'episodes':len(rows),'reset_clusters':len(seeds),'successes':sum(r['success'] for r in rows),'baseline_successes':sum(r['success'] for r in base)}
    for name,fn in [('success_gain',lambda r:r['success']),('max_reward_gain',lambda r:r['max_reward']),('extra_inference_seconds_per_episode',lambda r:r['decisions']*r['mean_decision_latency_seconds']),('extra_control_moves',lambda r:r['steps'])]:
        values=np.array([np.mean([fn(a[s,c])-fn(b[s,c]) for c in ['standard','shifted_start']]) for s in seeds])
        out[name]=interval(values)
    out['mean_of_episode_mean_decision_latencies_seconds']=float(np.mean([r['mean_decision_latency_seconds'] for r in rows]))
    out['baseline_mean_of_episode_mean_decision_latencies_seconds']=float(np.mean([r['mean_decision_latency_seconds'] for r in base]))
    out['pooled_mean_decision_latency_seconds']=json.loads((P/method/'summary.json').read_text())['mean_decision_latency_seconds']
    out['baseline_pooled_mean_decision_latency_seconds']=json.loads((P/'baseline/summary.json').read_text())['mean_decision_latency_seconds']
    out['pooled_latency_ratio']=out['pooled_mean_decision_latency_seconds']/out['baseline_pooled_mean_decision_latency_seconds']
    out['strata']={c:{'method_successes':sum(r['success'] for r in rows if r['condition']==c),'baseline_successes':sum(r['success'] for r in base if r['condition']==c),'episodes':12} for c in ['standard','shifted_start']}
    out['discordant_pairs']={'model_only_success':sum(a[k]['success'] and not b[k]['success'] for k in a),'baseline_only_success':sum(b[k]['success'] and not a[k]['success'] for k in a)}
    cost=out['extra_inference_seconds_per_episode']['mean'];gain=out['success_gain']['mean']
    out['break_even_lambda_point_estimate']=gain/cost if gain>0 and cost>0 else None
    summary.append(out)
pairwise=[]
methods=['baseline','world_model_a','world_model','direct_scorer']
all_rows={m:json.loads((P/m/'episodes.json').read_text()) for m in methods if (P/m/'summary.json').exists()}
for target,reference in [('world_model_a','direct_scorer'),('world_model','direct_scorer'),('world_model_a','world_model')]:
    if target not in all_rows or reference not in all_rows:continue
    a={key(r):r for r in all_rows[target]};bref={key(r):r for r in all_rows[reference]}
    row={'target':target,'reference':reference}
    for name,fn in [('success_gain',lambda r:r['success']),('extra_inference_seconds_per_episode',lambda r:r['decisions']*r['mean_decision_latency_seconds'])]:
        v=np.array([np.mean([fn(a[s,c])-fn(bref[s,c]) for c in ['standard','shifted_start']]) for s in seeds]);row[name]=interval(v)
    pairwise.append(row)
figure_data=[]
for method,rows in all_rows.items():
    look={key(r):r for r in rows}
    ss=np.array([np.mean([look[s,c]['success'] for c in ['standard','shifted_start']]) for s in seeds])
    tt=np.array([sum(look[s,c]['decisions']*look[s,c]['mean_decision_latency_seconds'] for c in ['standard','shifted_start']) for s in seeds])
    dd=np.array([sum(look[s,c]['decisions'] for c in ['standard','shifted_start']) for s in seeds])
    point=float(tt.sum()/dd.sum());rep=tt[draws].sum(1)/dd[draws].sum(1)
    measured=json.loads((P/method/'summary.json').read_text())
    assert np.isclose(point,measured['mean_decision_latency_seconds'])
    figure_data.append({'method':method,'success_rate':interval(ss),'pooled_latency_seconds':{'mean':point,'paired_reset_cluster_bootstrap_95_ci':np.quantile(rep,[.025,.975]).tolist()},'mean_inference_seconds_per_episode':float(tt.sum()/len(rows))})
out={'bootstrap_replicates':10000,'bootstrap_seed':42,'uncertainty':'Paired reset-cluster bootstrap, keeping standard and shifted trials together; one policy training seed and one auxiliary seed. Not training uncertainty; no multiplicity adjustment.','inference_time_definition':'Sum of decision latencies reconstructed as decisions times episode mean decision latency; excludes simulation and auxiliary training.','comparisons':summary,'pairwise_controls':pairwise,'figure_data':figure_data}
(OUT/'pusht_paired_analysis.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'comparisons':[{k:r[k] for k in ['method','successes','success_gain','extra_inference_seconds_per_episode','pooled_latency_ratio']} for r in summary],'pairwise_controls':pairwise,'figure_data':figure_data},indent=2))
