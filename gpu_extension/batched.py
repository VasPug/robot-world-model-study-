"""Batched independent episodes, explicit paired RNG, held-out gate evaluation.

DDPM update follows Hugging Face Diffusers 0.27.2 DDPMScheduler.step
(Apache-2.0), specialized to the released epsilon/fixed_small checkpoint.
An independent test compares the update with the pinned upstream implementation.
Batch throughput is never reported as single-robot decision latency.
"""
import csv
import itertools
import json
import time
import numpy as np
import torch
from . import study as s


def coefficients(scheduler):
    assert scheduler.config.prediction_type=='epsilon'
    assert scheduler.config.variance_type=='fixed_small'
    assert not scheduler.config.thresholding
    result=[]
    for t in scheduler.timesteps:
        previous=scheduler.previous_timestep(t)
        a=scheduler.alphas_cumprod[t]
        ap=scheduler.alphas_cumprod[previous] if previous>=0 else scheduler.one
        beta=1-a; bp=1-ap; current=a/ap
        result.append([float(beta.sqrt()),float(a.sqrt()),float(ap.sqrt()*(1-current)/beta),
                       float(current.sqrt()*bp/beta),float(scheduler._get_variance(t).sqrt()) if t>0 else 0.0])
    return torch.tensor(result,device='cuda')


def ddpm_step(sample,epsilon,noise,coef,clip_range):
    x0=(sample-coef[0]*epsilon)/coef[1]
    x0=x0.clamp(-clip_range,clip_range)
    return coef[2]*x0+coef[3]*sample+coef[4]*noise


@torch.inference_mode()
def sample_batch(policy,histories,seeds,count):
    assert policy.obs_as_cond and not policy.pred_action_steps_only
    assert policy.noise_scheduler.config.clip_sample
    batch=len(histories)
    obs=torch.as_tensor(np.ascontiguousarray(histories),device='cuda',dtype=torch.float32)
    cond=policy.normalizer['obs'].normalize(obs).repeat_interleave(count,0)
    scheduler=policy.noise_scheduler
    scheduler.set_timesteps(policy.num_inference_steps)
    if not hasattr(policy,'_extension_ddpm_coefficients'):
        policy._extension_ddpm_coefficients=coefficients(scheduler)
    coef=policy._extension_ddpm_coefficients
    # Draw a whole independent noise stream per episode/candidate on CPU, then
    # transfer once. Changing batch composition cannot change another row's RNG.
    noise=[]
    for seed in seeds:
        for candidate in range(count):
            generator=torch.Generator(device='cpu').manual_seed(int(seed)+candidate*1000000007)
            noise.append(torch.randn(policy.num_inference_steps+1,policy.horizon,policy.action_dim,generator=generator))
    noise=torch.stack(noise,1).to('cuda')
    trajectory=noise[0]
    for k,t in enumerate(scheduler.timesteps):
        output=policy.model(trajectory,t,cond)
        trajectory=ddpm_step(trajectory,output,noise[k+1],coef[k],scheduler.config.clip_sample_range)
    actions=policy.normalizer['action'].unnormalize(trajectory)
    start=policy.n_obs_steps-1
    return actions[:,start:start+8].cpu().numpy().reshape(batch,count,8,2)


def extend(chunks,histories,library):
    actions=np.empty((len(histories),chunks.shape[1],64,2),dtype=np.float32)
    for i,history in enumerate(histories):
        distance=np.mean((library['obs']-history[-1])**2,axis=1)
        near=np.argsort(distance)[:min(64,len(distance))]
        for j,chunk in enumerate(chunks[i]):
            cost=np.mean((library['actions'][near,:8]-chunk)**2,axis=(1,2))
            ref=library['actions'][near[int(np.argmin(cost))]].copy()
            ref[8:]+=(chunk[-1]-ref[7])[None]*np.linspace(1,0,56)[:,None]
            ref[:8]=chunk
            actions[i,j]=np.clip(ref,0,512)
    return actions


@torch.inference_mode()
def score_batch(model,kind,histories,actions,horizon,geo):
    b,k=actions.shape[:2]
    h=np.repeat(np.asarray(histories),k,axis=0)
    x,_=s.features(h,actions.reshape(b*k,64,2),horizon)
    pred=model(x).cpu().numpy()
    if kind=='world_model':
        future=pred.reshape(b,k,64,20)*100+np.asarray(histories)[:,-1,None,None,:]
        return geo.rewards(future[:,:,:horizon]).max(-1),future
    return np.clip(pred[:,0],0,1).reshape(b,k),None


def arm_key(arm):
    return f"{arm['kind']}_w{arm['width']}_s{arm['seed']}_h{arm['horizon']}"


def learned_arms(cfg):
    return [dict(kind=kind,width=width,seed=seed,horizon=horizon)
            for width,seed,horizon,kind in itertools.product(cfg['evaluation_widths'],cfg['training_seeds'],cfg['horizons'],('world_model','direct_scorer'))]


def batch_checks(policy,library,cfg,root):
    env=s.make_env(9981,'recovery'); obs=s.observe(env); env.close()
    histories=np.repeat(np.array([[obs,obs]]),2,axis=0)
    a=sample_batch(policy,histories[:1],[197],1)
    b=sample_batch(policy,histories,[197,198],2)
    # CUDA kernels can differ numerically with batch shape. Subpixel tolerance.
    error=float(np.max(np.abs(a[0,0]-b[0,0])))
    assert error<0.05, ('Batch composition changed candidate zero',error)
    reversed_batch=sample_batch(policy,histories[::-1],[198,197],2)
    order_error=float(np.max(np.abs(b[0]-reversed_batch[1])))
    assert order_error<0.05,order_error
    # Boundary windows remain deterministic and prefix actions are unchanged.
    full=extend(b,histories,library)
    np.testing.assert_allclose(full[:,:,:8],np.clip(b,0,512),atol=1e-6)
    facts=dict(candidate_zero_max_error_pixels=error,batch_order_max_error_pixels=order_error,
               sampler='100-step DDPM epsilon/fixed_small; independent CPU noise per candidate')
    s.save(root/'batch_checks.json',facts);s.emit('batch_checks_passed',**facts)


def benchmark(policy,library,models,cfg,root):
    # Fixed training-only histories. These are latency measurements, not rollouts.
    ids=np.linspace(0,len(library['obs'])-1,cfg['timing_samples'],dtype=int)
    histories=np.stack([np.stack([library['obs'][i],library['obs'][i]]) for i in ids])
    proposal_times={1:[],cfg['candidates']:[]}; proposal_actions=[]
    for count in proposal_times:
        for i,h in enumerate(histories):
            s.synchronize(); tick=time.perf_counter()
            chunks=sample_batch(policy,h[None],[998000+i],count)
            actions=extend(chunks,h[None],library) if count>1 else np.clip(chunks,0,512)
            s.synchronize();proposal_times[count].append(time.perf_counter()-tick)
            if count>1:proposal_actions.append(actions)
    env=s.make_env(9191);geo=s.Geometry(env);env.close()
    records={}
    records['policy_w0_s0_h8']=dict(mean_seconds=float(np.mean(proposal_times[1])),p95_seconds=float(np.quantile(proposal_times[1],.95)),samples=len(histories))
    for arm in learned_arms(cfg):
        model=models[arm['kind'],arm['width'],arm['seed']]
        latencies=[]
        for i,h in enumerate(histories):
            s.synchronize();tick=time.perf_counter()
            score_batch(model,arm['kind'],h[None],proposal_actions[i],arm['horizon'],geo)
            s.synchronize()
            latencies.append(proposal_times[cfg['candidates']][i]+time.perf_counter()-tick)
        records[arm_key(arm)]=dict(mean_seconds=float(np.mean(latencies)),p95_seconds=float(np.quantile(latencies,.95)),samples=len(histories))
    s.save(root/'single_decision_latency.json',records)
    s.emit('latency_benchmark',histories=len(histories),policy_mean_seconds=records['policy_w0_s0_h8']['mean_seconds'],
           selector_proposal_mean_seconds=float(np.mean(proposal_times[cfg['candidates']])),
           note='Single-episode proposal plus scoring timings; not divided batch wall time.')
    return records


def gate_features(observation,initial_reward):
    obs=np.asarray(observation); block=obs[:18].reshape(9,2)
    relative=(obs[18:]-block.mean(0))/256
    return np.array([1,initial_reward,*((block.mean(0)-256)/256),*relative],dtype=np.float64)


def episode_state(seed,condition,split):
    env=s.make_env(seed,condition);obs=s.observe(env);geo=s.Geometry(env)
    reward=float(geo.rewards(obs[None])[0]);assert reward<1
    return dict(env=env,geo=geo,history=[obs.copy(),obs.copy()],past=[],rewards=[],decision=0,
                success=False,initial_reward=reward,env_seed=seed,condition=condition,split=split,
                gate_features=gate_features(obs,reward).tolist(),prediction_errors=[],batch_charges=[])


def evaluate_batch(policy,library,model,arm,jobs,cfg,latency,episode_file,decision_file):
    states=[episode_state(*job) for job in jobs];rows=[];tick_batch=time.perf_counter()
    while True:
        active=[state for state in states if not state['success'] and len(state['past'])<cfg['max_steps']]
        if not active:break
        histories=np.asarray([state['history'] for state in active])
        seeds=[state['env_seed']*1000+state['decision']*2 for state in active]
        count=1 if arm['kind']=='policy' else cfg['candidates']
        s.synchronize();tick=time.perf_counter()
        chunks=sample_batch(policy,histories,seeds,count)
        actions=extend(chunks,histories,library) if count>1 else np.clip(chunks,0,512)
        score=None;future=None;chosen=np.zeros(len(active),dtype=int)
        if model is not None:
            score,future=score_batch(model,arm['kind'],histories,actions,arm['horizon'],active[0]['geo'])
            chosen=np.argmax(score,axis=1)
        s.synchronize();batch_seconds=time.perf_counter()-tick
        for i,state in enumerate(active):
            observed=[]
            for action in actions[i,chosen[i],:cfg['execution_horizon']]:
                full,reward,done,_=state['env'].step(action)
                obs=full[:20].astype(np.float32)
                state['history']=[state['history'][-1],obs];observed.append(obs)
                state['past'].append(action.copy());state['rewards'].append(float(reward))
                state['success']=bool(done)
                if done or len(state['past'])>=cfg['max_steps']:break
            error=None
            if future is not None:
                error=float(np.sqrt(np.mean((future[i,chosen[i],:len(observed),:18]-np.asarray(observed)[:,:18])**2)))
                state['prediction_errors'].append(error)
            decision_file.write(json.dumps(dict(**arm,split=state['split'],env_seed=state['env_seed'],
                condition=state['condition'],decision=state['decision'],chosen=int(chosen[i]),
                predicted_scores=None if score is None else score[i].tolist(),selected_keypoint_rmse=error,
                batch_active_episodes=len(active),batch_inference_seconds=batch_seconds),allow_nan=False)+'\n')
            state['batch_charges'].append(batch_seconds/len(active));state['decision']+=1
        decision_file.flush()
        if active[0]['decision']%8==0:
            s.emit('evaluation_batch_step',arm=arm,active_episodes=len(active),
                   decision=active[0]['decision'],batch_elapsed_seconds=time.perf_counter()-tick_batch)
    elapsed=time.perf_counter()-tick_batch
    for state in states:
        row=dict(**arm,split=state['split'],env_seed=state['env_seed'],condition=state['condition'],
                 success=int(state['success']),steps=len(state['past']),initial_reward=state['initial_reward'],
                 max_reward=max(state['rewards']),decisions=state['decision'],gate_features=state['gate_features'],
                 mean_latency_seconds=latency['mean_seconds'],p95_latency_seconds=latency['p95_seconds'],
                 estimated_serial_inference_seconds=latency['mean_seconds']*state['decision'],
                 amortized_batch_inference_seconds=sum(state['batch_charges']),batch_wall_seconds=elapsed,
                 batch_episodes=len(states),selected_keypoint_rmse=None if not state['prediction_errors'] else float(np.mean(state['prediction_errors'])))
        episode_file.write(json.dumps(row,allow_nan=False)+'\n');episode_file.flush();rows.append(row)
        state['env'].close()
    return rows


def oracle_diagnostics(policy,library,models,cfg,root):
    # Exact initial-state candidate ranking on separate validation resets only.
    records=[]
    for i in range(cfg['oracle_seeds']):
        env_seed=cfg['validation_seed_start']+i
        for condition in cfg['conditions']:
            env=s.make_env(env_seed,condition);obs=s.observe(env);geo=s.Geometry(env);env.close()
            histories=np.array([[obs,obs]])
            chunks=sample_batch(policy,histories,[env_seed*1000],cfg['candidates'])
            actions=extend(chunks,histories,library)[0]
            truths={h:[] for h in cfg['horizons']}
            for candidate in actions:
                e=s.make_env(env_seed,condition);_,reward=s.rollout(e,candidate);e.close()
                for h in truths:truths[h].append(float(reward[:h].max()))
            for arm in learned_arms(cfg):
                model=models[arm['kind'],arm['width'],arm['seed']]
                pred,_=score_batch(model,arm['kind'],histories,actions[None],arm['horizon'],geo)
                truth=np.asarray(truths[arm['horizon']]);chosen=int(np.argmax(pred[0]))
                records.append(dict(**arm,env_seed=env_seed,condition=condition,
                    ranking_regret=float(truth.max()-truth[chosen]),score_mae=float(np.mean(np.abs(truth-pred[0]))),
                    true_candidate_scores=truth.tolist(),predicted_candidate_scores=pred[0].tolist()))
            s.emit('ranking_diagnostic',env_seed=env_seed,condition=condition)
    s.save(root/'candidate_ranking.json',records)


def paired_interval(matrix,samples,rng):
    # Crossed resampling: same reset columns shared by all auxiliary seeds.
    if matrix.ndim==1:matrix=matrix[None]
    boot=[]
    for _ in range(samples):
        si=rng.integers(matrix.shape[0],size=matrix.shape[0])
        ei=rng.integers(matrix.shape[1],size=matrix.shape[1])
        boot.append(float(matrix[si][:,ei].mean()))
    return np.quantile(boot,[.025,.975]).tolist()


def oracle_closed_loop(policy,library,cfg,root):
    rows=[]
    for horizon in cfg['horizons']:
        for i,condition in itertools.product(range(cfg['oracle_seeds']),cfg['conditions']):
            seed=cfg['validation_seed_start']+i
            state=episode_state(seed,condition,'validation')
            timings=[]
            while not state['success'] and len(state['past'])<cfg['max_steps']:
                histories=np.asarray([state['history']])
                s.synchronize();tick=time.perf_counter()
                chunks=sample_batch(policy,histories,[seed*1000+state['decision']*2],cfg['candidates'])
                actions=extend(chunks,histories,library)[0]
                scores=s.oracle_scores(seed,condition,state['past'],actions,horizon)
                chosen=int(np.argmax(scores));s.synchronize();timings.append(time.perf_counter()-tick)
                for action in actions[chosen,:cfg['execution_horizon']]:
                    full,reward,done,_=state['env'].step(action)
                    obs=full[:20].astype(np.float32)
                    state['history']=[state['history'][-1],obs]
                    state['past'].append(action.copy());state['rewards'].append(float(reward));state['success']=bool(done)
                    if done or len(state['past'])>=cfg['max_steps']:break
                state['decision']+=1
            row=dict(kind='oracle',split='validation',env_seed=seed,condition=condition,horizon=horizon,
                     success=int(state['success']),steps=len(state['past']),max_reward=max(state['rewards']),
                     mean_latency_seconds=float(np.mean(timings)),decisions=state['decision'])
            rows.append(row);state['env'].close()
            s.save(root/'oracle_episodes.json',rows);s.emit('oracle_episode',**row)
    return rows


def matrix_for(rows,kind,width,horizon,condition,cfg,metric='success',split='test'):
    rr=[r for r in rows if r['kind']==kind and r['width']==width and r['horizon']==horizon and r['condition']==condition and r['split']==split]
    seeds=[0] if kind=='policy' else cfg['training_seeds']
    resets=sorted(set(r['env_seed'] for r in rr))
    lookup={(r['seed'],r['env_seed']):r for r in rr}
    return np.array([[lookup[seed,reset][metric] for reset in resets] for seed in seeds],dtype=float)


def analyze(rows,cfg,root,oracle_rows=None):
    rng=np.random.default_rng(9181);groups=[];samples=cfg['bootstrap_samples']
    for condition in cfg['conditions']:
        baseline=matrix_for(rows,'policy',0,8,condition,cfg)
        groups.append(dict(kind='policy',width=0,horizon=8,condition=condition,success_rate=float(baseline.mean()),
                           episodes=baseline.size,independent_resets=baseline.shape[1],auxiliary_seeds=0,
                           mean_single_decision_seconds=float(matrix_for(rows,'policy',0,8,condition,cfg,'mean_latency_seconds').mean())))
        for width,horizon in itertools.product(cfg['evaluation_widths'],cfg['horizons']):
            scorer=matrix_for(rows,'direct_scorer',width,horizon,condition,cfg)
            for kind in ('world_model','direct_scorer'):
                successes=matrix_for(rows,kind,width,horizon,condition,cfg)
                group=dict(kind=kind,width=width,horizon=horizon,condition=condition,
                           success_rate=float(successes.mean()),episodes=successes.size,
                           independent_resets=successes.shape[1],auxiliary_seeds=successes.shape[0],
                           paired_gain_over_policy=float((successes-baseline).mean()),
                           paired_gain_over_policy_95ci=paired_interval(successes-baseline,samples,rng),
                           mean_single_decision_seconds=float(matrix_for(rows,kind,width,horizon,condition,cfg,'mean_latency_seconds').mean()))
                if kind=='world_model':
                    group.update(paired_gain_over_scorer=float((successes-scorer).mean()),
                                 paired_gain_over_scorer_95ci=paired_interval(successes-scorer,samples,rng))
                groups.append(group)
    gates=fit_gates(rows,cfg,root)
    summary=dict(stage=cfg['stage'],test_episodes=sum(r['split']=='test' for r in rows),
                 validation_episodes=sum(r['split']=='validation' for r in rows),groups=groups,gates=gates,
                 primary_contrast=dict(width=cfg['primary_width'],horizon=cfg['primary_horizon'],methods=['world_model','direct_scorer']),
                 peak_gpu_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                 limitations=['One released policy training seed. Auxiliary/reset uncertainty only.',
                     'Policy is frozen; explicit prediction is controlled by the data/capacity-matched direct scorer.',
                     'Restricted retrieval-extended action proposals; known rigid geometry and reward.',
                     'Zero-shot shifts; no online adaptation.',
                     'Latency is a separate single-decision benchmark; serial episode costs are estimates.',
                     'Pointwise bootstrap intervals are exploratory and not multiple-comparison corrected.'])
    summary['oracle_validation_groups']=[dict(horizon=h,condition=c,episodes=len(rr),
        success_rate=float(np.mean([r['success'] for r in rr])))
        for h,c in itertools.product(cfg['horizons'],cfg['conditions'])
        if (rr:=[r for r in (oracle_rows or []) if r['horizon']==h and r['condition']==c])]
    s.save(root/'summary.json',summary)
    s.emit('FINAL_SUMMARY',**summary)
    report=['# GPU horizon and physics full study','',
        f"Completed {summary['test_episodes']} held-out and {summary['validation_episodes']} validation episodes.",
        'Three auxiliary seeds and 100 paired test resets per stratum; one released policy seed.', '',
        '## Prespecified medium-size, 32-step comparison','',
        '| Condition | World model success | Direct scorer success | Paired gain (95% interval) |',
        '|---|---:|---:|---:|']
    for condition in cfg['conditions']:
        wm=next(g for g in groups if g['kind']=='world_model' and g['width']==cfg['primary_width'] and g['horizon']==cfg['primary_horizon'] and g['condition']==condition)
        ds=next(g for g in groups if g['kind']=='direct_scorer' and g['width']==cfg['primary_width'] and g['horizon']==cfg['primary_horizon'] and g['condition']==condition)
        lo,hi=wm['paired_gain_over_scorer_95ci']
        report.append(f"| {condition} | {wm['success_rate']:.1%} | {ds['success_rate']:.1%} | {wm['paired_gain_over_scorer']:+.1%} ({lo:+.1%}, {hi:+.1%}) |")
    report+=['','All capacities, horizons, paired intervals and gate results: [summary.json](summary.json).',
             'Raw episodes: [episodes.csv](episodes.csv). Timing: [single_decision_latency.json](single_decision_latency.json).',
             'Consultation weights are fitted on disjoint validation resets only: [gate_models.json](gate_models.json).',
             '', '## Limits','']+['- '+text for text in summary['limitations']]
    (root/'report.md').write_text('\n'.join(report),encoding='utf-8')


def fit_gates(rows,cfg,root):
    records=[];weights=[];rng=np.random.default_rng(1951)
    lookup={(r['split'],r['kind'],r['seed'],r['env_seed'],r['condition']):r for r in rows
            if r['kind']=='policy' or (r['kind']=='world_model' and r['width']==cfg['primary_width'] and r['horizon']==cfg['primary_horizon'])}
    for lam in cfg['consultation_lambdas']:
        for seed in cfg['training_seeds']:
            train=[r for r in rows if r['split']=='validation' and r['kind']=='world_model' and r['seed']==seed and r['width']==cfg['primary_width'] and r['horizon']==cfg['primary_horizon']]
            x=np.asarray([r['gate_features'] for r in train]);y=[]
            for r in train:
                b=lookup['validation','policy',0,r['env_seed'],r['condition']]
                y.append(r['success']-b['success']-lam*(r['estimated_serial_inference_seconds']-b['estimated_serial_inference_seconds']))
            regularizer=np.eye(x.shape[1]);regularizer[0,0]=0
            coef=np.linalg.solve(x.T@x+regularizer,x.T@np.asarray(y))
            weights.append(dict(lambda_per_second=lam,seed=seed,coefficients=coef.tolist(),features=['intercept','initial_coverage','block_x','block_y','relative_pusher_x','relative_pusher_y'],fit_split='validation'))
            for r in rows:
                if r['split']!='test' or r['kind']!='world_model' or r['seed']!=seed or r['width']!=cfg['primary_width'] or r['horizon']!=cfg['primary_horizon']:continue
                b=lookup['test','policy',0,r['env_seed'],r['condition']]
                consult=float(np.dot(r['gate_features'],coef))>0
                chosen=r if consult else b
                records.append(dict(lambda_per_second=lam,seed=seed,env_seed=r['env_seed'],condition=r['condition'],consult=int(consult),
                    success=chosen['success'],utility=chosen['success']-lam*chosen['estimated_serial_inference_seconds'],
                    policy_utility=b['success']-lam*b['estimated_serial_inference_seconds'],
                    model_utility=r['success']-lam*r['estimated_serial_inference_seconds']))
    summary=[]
    for lam,condition in itertools.product(cfg['consultation_lambdas'],cfg['conditions']):
        rr=[r for r in records if r['lambda_per_second']==lam and r['condition']==condition]
        resets=sorted(set(r['env_seed'] for r in rr));mapped={(r['seed'],r['env_seed']):r for r in rr}
        gain=np.array([[mapped[seed,reset]['utility']-mapped[seed,reset]['policy_utility'] for reset in resets] for seed in cfg['training_seeds']])
        summary.append(dict(lambda_per_second=lam,condition=condition,success_rate=float(np.mean([r['success'] for r in rr])),
            consultation_rate=float(np.mean([r['consult'] for r in rr])),utility_gain_over_policy=float(gain.mean()),
            utility_gain_over_policy_95ci=paired_interval(gain,cfg['bootstrap_samples'],rng)))
    s.save(root/'gate_models.json',weights);s.save(root/'gate_test_episodes.json',records)
    return summary


def run_full(policy,library,models,cfg,root):
    batch_checks(policy,library,cfg,root)
    latency=benchmark(policy,library,models,cfg,root)
    oracle_diagnostics(policy,library,models,cfg,root)
    arms=learned_arms(cfg)
    # Block order is fixed and randomized before outcomes are observed.
    np.random.default_rng(777).shuffle(arms)
    arms.insert(0,dict(kind='policy',width=0,seed=0,horizon=8))
    rows=[];start=time.perf_counter()
    total=(cfg['validation_reset_seeds']+cfg['evaluation_seeds'])*len(cfg['conditions'])*len(arms)
    with (root/'episodes.jsonl').open('w',encoding='utf-8') as ef,(root/'decisions.jsonl').open('w',encoding='utf-8') as df:
        for arm in arms:
            model=models.get((arm['kind'],arm['width'],arm['seed']))
            for split,seed_start,count in [('validation',cfg['validation_seed_start'],cfg['validation_reset_seeds']),('test',cfg['evaluation_seed_start'],cfg['evaluation_seeds'])]:
                jobs=[(seed_start+i,condition,split) for i in range(count) for condition in cfg['conditions']]
                for offset in range(0,len(jobs),cfg['parallel_episodes']):
                    batch=jobs[offset:offset+cfg['parallel_episodes']]
                    new=evaluate_batch(policy,library,model,arm,batch,cfg,latency[arm_key(arm)],ef,df)
                    rows.extend(new)
                    progress=dict(completed=len(rows),total=total,elapsed_seconds=time.perf_counter()-start,
                                  latest_arm=arm,split=split,batch_successes=sum(r['success'] for r in new),batch_episodes=len(new))
                    progress['estimated_remaining_seconds']=progress['elapsed_seconds']/len(rows)*(total-len(rows))
                    s.save(root/'progress.json',progress);s.emit('evaluation_progress',**progress)
    s.save(root/'episodes.json',rows)
    with (root/'episodes.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    oracle_rows=oracle_closed_loop(policy,library,cfg,root)
    analyze(rows,cfg,root,oracle_rows)
