"""GPU horizon study; source and configuration define each reproduction run."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import urllib.request
import zipfile

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'vendor'))
import dill
import numpy as np
import pymunk
import shapely
import torch
from torch import nn
import zarr
from omegaconf import OmegaConf
from diffusers import DDPMScheduler
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.policy.diffusion_transformer_lowdim_policy import DiffusionTransformerLowdimPolicy
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from diffusion_policy.env.pusht.pusht_env import pymunk_to_shapely

HMAX = 64
INPUT = 40 + HMAX * 3


def emit(event, **kwargs):
    print(json.dumps(dict(event=event, **kwargs), allow_nan=False), flush=True)


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def synchronize():
    torch.cuda.synchronize()


def inputs(cfg):
    cache = Path(cfg['input_cache'])
    cache.mkdir(parents=True, exist_ok=True)
    assets = json.loads((REPO / 'assets.json').read_text())
    for name, spec in assets.items():
        target = cache / name
        if not target.exists():
            emit('download_start', asset=name)
            temporary = target.with_suffix('.part')
            urllib.request.urlretrieve(spec['url'], temporary)
            assert digest(temporary) == spec['sha256'], name
            temporary.replace(target)
        assert digest(target) == spec['sha256'], name
        emit('input_verified', asset=name, sha256=spec['sha256'])
    data = cache / 'dataset'
    if not (data / '.complete').exists():
        data.mkdir(exist_ok=True)
        with zipfile.ZipFile(cache / 'pusht.zip') as z:
            for entry in z.infolist():
                assert (data / entry.filename).resolve().is_relative_to(data.resolve())
            z.extractall(data)
        (data / '.complete').touch()
    return cache


def load_policy(cache):
    payload = torch.load(cache / 'transformer.ckpt', map_location='cpu', pickle_module=dill)
    original = OmegaConf.to_container(payload['cfg'], resolve=True)
    pc = copy.deepcopy(original['policy'])
    pc.pop('_target_')
    mc = pc.pop('model'); mc.pop('_target_')
    sc = pc.pop('noise_scheduler'); sc.pop('_target_')
    policy = DiffusionTransformerLowdimPolicy(
        model=TransformerForDiffusion(**mc), noise_scheduler=DDPMScheduler(**sc), **pc)
    policy.load_state_dict(payload['state_dicts']['ema_model'], strict=True)
    policy = policy.cuda().eval().requires_grad_(False)
    assert policy.n_action_steps == 8 and policy.num_inference_steps == 100
    emit('policy', parameters=sum(p.numel() for p in policy.parameters()),
         inference_steps=100, released_policy_training_seeds=1)
    return policy, original


class Geometry:
    """Vectorized rigid projection and exact polygon coverage, as in the pilot."""
    def __init__(self, env):
        self.template = env.kp_manager.local_keypoint_map['block'].copy()
        self.centered = self.template - self.template.mean(0)
        self.vertices = np.array([[tuple(v) for v in s.get_vertices()] for s in env.block.shapes])
        self.goal = pymunk_to_shapely(env._get_goal_pose_body(env.goal_pose), env.block.shapes)
        self.threshold = env.success_threshold

    def rewards(self, observations):
        original_shape = observations.shape[:-1]
        q = np.asarray(observations)[..., :18].reshape(-1, 9, 2)
        qc = q - q.mean(1, keepdims=True)
        dot = (self.centered * qc).sum((1, 2))
        cross = (self.centered[:, 0] * qc[:, :, 1] - self.centered[:, 1] * qc[:, :, 0]).sum(1)
        theta = np.arctan2(cross, dot)
        c, s = np.cos(theta), np.sin(theta)
        rot = np.stack([c, -s, s, c], -1).reshape(-1, 2, 2)
        pos = q.mean(1) - np.einsum('nij,j->ni', rot, self.template.mean(0))
        points = np.einsum('nij,skj->nski', rot, self.vertices) + pos[:, None, None, :]
        polys = shapely.union_all(shapely.polygons(points), axis=1)
        area = shapely.area(shapely.intersection(polys, self.goal))
        return np.clip(area / self.goal.area / self.threshold, 0, 1).reshape(original_shape)


def make_env(seed, condition='recovery', state=None):
    env = PushTKeypointsEnv(legacy=True)
    env.seed(int(seed)); env.reset()
    rng = np.random.default_rng(seed)
    if state is not None:
        env._set_state(np.asarray(state))
    else:
        # All shifted conditions share exactly the recovery start for this seed.
        if condition == 'easy':
            theta = float(env.goal_pose[2] + rng.uniform(-0.10, 0.10))
            pos = env.goal_pose[:2] + [rng.uniform(-12, 12), rng.uniform(25, 45)]
        else:
            theta = float(env.goal_pose[2] + rng.choice([-1, 1]) * rng.uniform(0.8, 1.5))
            pos = env.goal_pose[:2] + rng.uniform(-65, 65, 2)
        env.block.angle = theta
        env.block.position = pos.tolist()
        # Place the pusher outside the block, on an easy or deliberately awkward side.
        delta = pos - env.goal_pose[:2]
        direction = delta / max(np.linalg.norm(delta), 1)
        side = 1 if condition == 'easy' else -1
        agent = np.clip(pos + side * 155 * direction, 25, 487)
        env.agent.position = agent.tolist()
        for body in (env.agent, env.block):
            body.velocity = (0, 0); body.angular_velocity = 0
            body.force = (0, 0); body.torque = 0
            env.space.reindex_shapes_for_body(body)
    if condition == 'friction_shift':
        for shape in (*env.agent.shapes, *env.block.shapes):
            shape.friction = 0.9
    elif condition == 'damping_shift':
        env.space.damping = 0.8
    return env


def observe(env):
    return env._get_obs()[:20].astype(np.float32)


def rollout(env, actions, absorbing=True):
    observations, rewards = [], []
    done = False
    for action in actions:
        if not done:
            full, reward, terminal, _ = env.step(action)
            obs = full[:20].astype(np.float32)
            done = bool(terminal) and absorbing
        observations.append(obs.copy()); rewards.append(float(reward))
    return np.asarray(observations), np.asarray(rewards, dtype=np.float32)


def dataset(cache, original, cfg, root):
    paths = list((cache / 'dataset').rglob('*.zarr'))
    assert len(paths) == 1
    z = zarr.open(str(paths[0]), mode='r')
    kps = z['data/keypoint'][:].reshape(-1, 18).astype(np.float32)
    state, actions = z['data/state'][:], z['data/action'][:].astype(np.float32)
    obs = np.concatenate([kps, state[:, :2]], -1)
    ends = z['meta/episode_ends'][:]
    starts = np.concatenate([[0], ends[:-1]])
    dc = original['task']['dataset']
    rng = np.random.default_rng(dc['seed'])
    excluded = rng.choice(len(ends), min(max(1, round(len(ends) * dc['val_ratio'])), len(ends)-1), replace=False)
    eligible = np.setdiff1d(np.arange(len(ends)), excluded)
    rng = np.random.default_rng(dc['seed'])
    eligible = np.sort(eligible[rng.choice(len(eligible), dc['max_train_episodes'], replace=False)])
    val_ids = np.random.default_rng(2026).choice(eligible, 18, replace=False)
    train_ids = np.setdiff1d(eligible, val_ids)
    assert not set(train_ids) & set(val_ids)
    geo_env = make_env(9191)
    geo = Geometry(geo_env); geo_env.close()
    # Candidate library contains training demonstrations only, with no padded futures.
    locations = np.concatenate([np.arange(starts[ep]+1, ends[ep]-HMAX, 4) for ep in train_ids])
    library = dict(obs=obs[locations], actions=np.stack([actions[t:t+HMAX] for t in locations]))
    out = {}
    for split, ids, count, random_seed in [('train', train_ids, cfg['train_windows'], 1041),
                                         ('validation', val_ids, cfg['validation_windows'], 1042)]:
        rng = np.random.default_rng(random_seed)
        histories, candidates, futures, scores = [], [], [], []
        source_ids = [int(ep) for ep in ids if ends[ep]-starts[ep] > HMAX+1]
        assert source_ids, f'No complete {HMAX}-step windows in {split}'
        emit('eligible_windows', split=split, source_episode_ids=source_ids,
             excluded_short_episode_ids=[int(ep) for ep in ids if int(ep) not in source_ids])
        for i in range(count):
            ep = int(rng.choice(source_ids))
            t = int(rng.integers(starts[ep]+1, ends[ep]-HMAX))
            act = actions[t:t+HMAX].copy()
            if i % 2 == 0:
                history = obs[t-1:t+1].copy()
                future = obs[t+1:t+HMAX+1].copy()
                reward = geo.rewards(future).astype(np.float32)
                # Observed terminal success is absorbing, never cross episode boundaries.
                terminal = np.flatnonzero(reward >= 1)
                if len(terminal):
                    future[terminal[0]:] = future[terminal[0]]
                    reward[terminal[0]:] = 1
            else:
                # Nominal dynamics only. Episode IDs remain disjoint across splits.
                env = make_env(300000+i, state=state[t, :5])
                current = observe(env)
                history = np.stack([current, current])
                knots = rng.normal(0, 25, (9, 2))
                perturbation = np.stack([np.interp(np.arange(HMAX), np.linspace(0,HMAX-1,9), knots[:,d]) for d in range(2)], -1)
                act = np.clip(act + perturbation, 0, 512).astype(np.float32)
                future, reward = rollout(env, act)
                env.close()
            histories.append(history); candidates.append(act)
            futures.append(future); scores.append(reward)
            if (i+1) % 100 == 0 or i+1 == count:
                emit('data_progress', split=split, windows=i+1, total=count)
        out[split] = tuple(np.asarray(v, dtype=np.float32) for v in (histories,candidates,futures,scores))
    split_record = dict(train_episode_ids=train_ids.tolist(), validation_episode_ids=sorted(val_ids.tolist()),
                        train_windows=cfg['train_windows'], validation_windows=cfg['validation_windows'],
                        max_horizon=HMAX, synthetic_fraction=0.5, extra_simulated_controls=sum(cfg[k]//2*HMAX for k in ('train_windows','validation_windows')),
                        library_windows=len(locations), dataset_seed=1041)
    save(root/'data_split.json', split_record)
    np.savez_compressed(root/'training_data.npz', **{f'{s}_{k}':v for s, data in out.items() for k,v in zip(('history','actions','future','rewards'),data)})
    emit('data_final', **split_record)
    return out, library


def features(history, actions, horizon):
    # Inputs after the requested horizon are explicitly removed.
    b = len(actions)
    h = torch.as_tensor(history, dtype=torch.float32, device='cuda').reshape(b, 40)/256-1
    a = torch.as_tensor(actions, dtype=torch.float32, device='cuda')/256-1
    horizons = torch.as_tensor(horizon, device='cuda').expand(b)
    mask = torch.arange(HMAX, device='cuda')[None] < horizons[:, None]
    return torch.cat([h, (a*mask[...,None]).reshape(b,-1), mask.float()], -1), mask


class Predictor(nn.Module):
    def __init__(self, width, output):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(INPUT,width), nn.SiLU(), nn.Linear(width,width), nn.SiLU(), nn.Linear(width,output))

    def forward(self, x):
        return self.net(x)


def parameter_count(width, output):
    return (INPUT+1)*width + (width+1)*width + (width+1)*output


def scorer_width(width):
    target = parameter_count(width, HMAX*20)
    approximate = int((- (INPUT+3) + math.sqrt((INPUT+3)**2+4*(target-1)))/2)
    return min(range(max(1,approximate-2),approximate+3), key=lambda w:abs(parameter_count(w,1)-target))


def train_models(data, cfg, root):
    models, metrics = {}, []
    histories, acts, future, rewards = [torch.as_tensor(a, device='cuda') for a in data['train']]
    vh, va, vf, vr = [torch.as_tensor(a, device='cuda') for a in data['validation']]
    targets = (future - histories[:, -1, None, :])/100
    vt = (vf-vh[:,-1,None,:])/100
    horizon_values = torch.tensor(cfg['horizons'], device='cuda')
    for seed in cfg['training_seeds']:
        for width in cfg['widths']:
            for kind in ('world_model', 'direct_scorer'):
                torch.manual_seed(seed)
                actual_width = width if kind == 'world_model' else scorer_width(width)
                model = Predictor(actual_width,HMAX*20 if kind=='world_model' else 1).cuda()
                optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=1e-5)
                best, weights, best_epoch = float('inf'), None, 0
                synchronize(); started = time.perf_counter()
                for epoch in range(cfg['epochs']):
                    model.train(); losses=[]
                    for ix in torch.randperm(len(histories),device='cuda').split(cfg['batch_size']):
                        hs = horizon_values[torch.randint(len(horizon_values),(len(ix),),device='cuda')]
                        x, mask = features(histories[ix],acts[ix],hs)
                        prediction = model(x)
                        if kind == 'world_model':
                            residual = prediction.reshape(-1,HMAX,20)-targets[ix]
                            loss = (residual.square()*mask[...,None]).sum()/(mask.sum()*20)
                        else:
                            target = rewards[ix].masked_fill(~mask,-1).max(1).values
                            loss = (prediction[:,0]-target).square().mean()
                        optimizer.zero_grad(set_to_none=True); loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
                        losses.append(float(loss.detach()))
                    model.eval()
                    with torch.inference_mode():
                        val=[]
                        for horizon in cfg['horizons']:
                            batch_val=[]
                            for ix in torch.arange(len(vh), device='cuda').split(cfg['batch_size']):
                                x, mask = features(vh[ix],va[ix],horizon)
                                pred = model(x)
                                if kind=='world_model':
                                    loss = (pred.reshape(-1,HMAX,20)[:,:horizon]-vt[ix,:horizon]).square().mean()
                                else:
                                    loss = (pred[:,0]-vr[ix,:horizon].max(1).values).square().mean()
                                batch_val.append((float(loss),len(ix)))
                            val.append(sum(v*n for v,n in batch_val)/len(vh))
                    validation = float(np.mean(val))
                    if not np.isfinite(validation):
                        raise RuntimeError('Non-finite validation loss')
                    if validation < best:
                        best, best_epoch = validation, epoch+1
                        weights = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    if epoch==0 or (epoch+1)%5==0 or epoch+1==cfg['epochs']:
                        emit('training',kind=kind,width=width,seed=seed,epoch=epoch+1,
                             train_mse=float(np.mean(losses)),validation_mse=validation,horizon_mse=val)
                model.load_state_dict(weights); model.eval().requires_grad_(False)
                synchronize()
                key = f'{kind}_w{width}_s{seed}'
                checkpoint = root/'models'/f'{key}.pt'; checkpoint.parent.mkdir(exist_ok=True)
                torch.save(dict(state_dict=weights,width=actual_width,kind=kind,seed=seed),checkpoint)
                metric = dict(key=key,kind=kind,width=width,actual_width=actual_width,seed=seed,
                              parameters=sum(p.numel() for p in model.parameters()),best_epoch=best_epoch,
                              validation_mse=best,training_seconds=time.perf_counter()-started,sha256=digest(checkpoint))
                emit('model_final',**metric); metrics.append(metric)
                models[kind,width,seed] = model
    save(root/'training_metrics.json',metrics)
    return models


@torch.inference_mode()
def proposals(policy, history, library, count, seed):
    torch.manual_seed(seed)
    h = torch.as_tensor(np.asarray(history),device='cuda').float()[None]
    first = policy.predict_action({'obs':h})['action'].cpu().numpy()
    if count == 1:
        return first
    torch.manual_seed(seed+1)
    rest = policy.predict_action({'obs':h.repeat(count-1,1,1)})['action'].cpu().numpy()
    chunks = np.concatenate([first,rest],0)
    distance = np.mean((library['obs']-history[-1])**2,axis=1)
    near = np.argsort(distance)[:min(64,len(distance))]
    actions=[]
    for chunk in chunks:
        # Retrieve a complete training action sequence with a similar prefix.
        # This is shared across selectors and never queries predicted or true futures.
        cost = np.mean((library['actions'][near,:8]-chunk)**2,axis=(1,2))
        ref = library['actions'][near[int(np.argmin(cost))]].copy()
        offset = chunk[-1]-ref[7]
        ref[8:] += offset[None]*np.linspace(1,0,HMAX-8)[:,None]
        ref[:8] = chunk
        actions.append(np.clip(ref,0,512))
    return np.asarray(actions,dtype=np.float32)


def oracle_scores(seed, condition, past_actions, actions, horizon):
    # Replay preserves solver contact history; approximate state copies are avoided.
    scores=[]
    for candidate in actions:
        env=make_env(seed,condition)
        for action in past_actions:
            env.step(action)
        _,reward=rollout(env,candidate[:horizon])
        scores.append(float(reward.max())); env.close()
    return np.asarray(scores)


@torch.inference_mode()
def predicted_scores(model,kind,history,actions,horizon,geo):
    x,_=features(np.repeat(np.asarray(history)[None],len(actions),axis=0),actions,horizon)
    pred=model(x).cpu().numpy()
    if kind=='world_model':
        futures=pred.reshape(-1,HMAX,20)*100+np.asarray(history[-1])[None,None,:]
        return geo.rewards(futures[:,:horizon]).max(1),futures
    return np.clip(pred[:,0],0,1),None


def checks(policy, library, cfg, root):
    env=make_env(90123,'recovery'); obs=observe(env); geo=Geometry(env)
    acts=proposals(policy,[obs,obs],library, cfg['candidates'],981)
    single=proposals(policy,[obs,obs],library,1,981)
    assert np.allclose(acts[0,:8],single[0],atol=1e-5)
    full,reward,_,_=env.step(acts[0,0])
    geometry_error=abs(float(geo.rewards(full[None,:20])[0])-reward)
    assert geometry_error < 1e-5, geometry_error
    x,_=features(np.stack([[obs,obs]]),acts[:1],8)
    changed=acts[:1].copy(); changed[:,8:]=0
    xx,_=features(np.stack([[obs,obs]]),changed,8)
    assert torch.equal(x,xx), 'Future-action leakage'
    oracle=oracle_scores(90123,'recovery',[],acts[:1],8)[0]
    e2=make_env(90123,'recovery'); _,rs=rollout(e2,acts[0,:8]); e2.close()
    assert abs(oracle-float(rs.max()))<1e-7
    # Contact-rich reproducible probe, tested across shifts (not used to select test cases).
    physics={}
    for condition in ('recovery','friction_shift','damping_shift'):
        e=make_env(9876,condition,state=np.array([256,420,256,280,0.0]))
        path=np.concatenate([np.linspace([256,420],[256,200],32),np.linspace([256,200],[350,220],32)])
        states,_=rollout(e,path,absorbing=False); physics[condition]=states; e.close()
    differences={c:float(np.max(np.abs(v-physics['recovery']))) for c,v in physics.items() if c!='recovery'}
    for c,v in differences.items():
        assert v>0.01, f'{c} has no measurable effect: {v}'
    facts=dict(cuda=True,geometry_reward_error=geometry_error,candidate_zero_preserved=True,
               masked_future_actions=True,oracle_replay_exact=True,physics_max_coordinate_change=differences)
    save(root/'checks.json',facts); emit('checks_passed',**facts); env.close()


def evaluate(policy,library,models,cfg,root):
    rows=[]; decisions_path=root/'decisions.jsonl'
    arms=[dict(kind='policy',width=0,seed=0,horizon=8)]
    for seed in cfg['training_seeds']:
        for width in cfg['evaluation_widths']:
            for horizon in cfg['horizons']:
                for kind in ('world_model','direct_scorer'):
                    arms.append(dict(kind=kind,width=width,seed=seed,horizon=horizon))
    arms += [dict(kind='oracle',width=0,seed=0,horizon=h) for h in cfg['horizons']]
    jobs=[(cfg['evaluation_seed_start']+i,c,a) for i in range(cfg['evaluation_seeds']) for c in cfg['conditions'] for a in arms if a['kind']!='oracle' or i<cfg['oracle_seeds']]
    # Interleave arm order to reduce ordering/thermal bias.
    np.random.default_rng(141).shuffle(jobs)
    with decisions_path.open('w',encoding='utf-8') as fd:
        for number,(env_seed,condition,arm) in enumerate(jobs):
            env=make_env(env_seed,condition); obs=observe(env); geo=Geometry(env)
            initial_reward=float(geo.rewards(obs[None])[0])
            assert initial_reward < 1, 'Trivial already-successful reset'
            history=[obs.copy(),obs.copy()]; past=[]; rewards=[]; latencies=[]
            chosen_errors=[]; prediction_errors=[]; started=time.perf_counter(); success=False; decision=0
            model=models.get((arm['kind'],arm['width'],arm['seed']))
            # Initial history sufficient for the policy; no hidden dynamics parameter is exposed.
            while len(past)<cfg['max_steps'] and not success:
                action_seed=int(env_seed*1000+decision*2)
                synchronize(); tic=time.perf_counter()
                acts=proposals(policy,history,library,1 if arm['kind']=='policy' else cfg['candidates'],action_seed)
                synchronize(); proposal_seconds=time.perf_counter()-tic
                score=None; futures=None; chosen=0
                if model is not None:
                    score,futures=predicted_scores(model,arm['kind'],history,acts,arm['horizon'],geo)
                    chosen=int(np.argmax(score))
                elif arm['kind']=='oracle':
                    score=oracle_scores(env_seed,condition,past,acts,arm['horizon'])
                    chosen=int(np.argmax(score))
                synchronize(); latency=time.perf_counter()-tic; latencies.append(latency)
                # Counterfactual ranking is a diagnostic only, outside deployment latency.
                diagnostic=None
                if decision==0 and model is not None and env_seed-cfg['evaluation_seed_start']<cfg['oracle_seeds']:
                    truth=oracle_scores(env_seed,condition,past,acts,arm['horizon'])
                    diagnostic=dict(oracle_scores=truth.tolist(),ranking_regret=float(truth.max()-truth[chosen]),
                                    candidate_score_mae=float(np.mean(np.abs(truth-score))))
                    chosen_errors.append(diagnostic['ranking_regret'])
                actual=[]
                for action in acts[chosen,:cfg['execution_horizon']]:
                    full,reward,done,_=env.step(action)
                    new=full[:20].astype(np.float32)
                    history=[history[-1],new]; actual.append(new)
                    rewards.append(float(reward)); past.append(action.copy()); success=bool(done)
                    if success or len(past)>=cfg['max_steps']:
                        break
                err=None
                if futures is not None:
                    err=float(np.sqrt(np.mean((futures[chosen,:len(actual),:18]-np.asarray(actual)[:,:18])**2)))
                    prediction_errors.append(err)
                record=dict(**arm,env_seed=env_seed,condition=condition,decision=decision,chosen=chosen,
                            prediction_scores=None if score is None else score.tolist(),latency_seconds=latency,
                            proposal_seconds=proposal_seconds,selected_keypoint_rmse=err,diagnostic=diagnostic)
                fd.write(json.dumps(record,allow_nan=False)+'\n'); fd.flush(); decision+=1
            row=dict(**arm,env_seed=env_seed,condition=condition,success=int(success),steps=len(past),
                     initial_reward=initial_reward,max_reward=max(rewards),decisions=decision,
                     mean_latency_seconds=float(np.mean(latencies)),p95_latency_seconds=float(np.quantile(latencies,.95)),
                     total_inference_seconds=float(sum(latencies)),episode_seconds=time.perf_counter()-started,
                     first_decision_ranking_regret=None if not chosen_errors else float(np.mean(chosen_errors)),
                     selected_keypoint_rmse=None if not prediction_errors else float(np.mean(prediction_errors)))
            rows.append(row); save(root/'episodes.json',rows)
            emit('episode',completed=number+1,total=len(jobs),**row)
            env.close()
    with (root/'episodes.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return rows


def summarize(rows,cfg,root):
    groups={}
    for row in rows:
        key=(row['kind'],row['width'],row['horizon'],row['condition'])
        groups.setdefault(key,[]).append(row)
    baseline={(r['env_seed'],r['condition']):r for r in rows if r['kind']=='policy'}
    result=[]; rng=np.random.default_rng(999)
    for (kind,width,horizon,condition),rr in groups.items():
        seeds=sorted(set(r['seed'] for r in rr)); resets=sorted(set(r['env_seed'] for r in rr))
        success=np.array([[next(r['success'] for r in rr if r['seed']==s and r['env_seed']==e) for e in resets] for s in seeds])
        delta=success-np.array([baseline[e,condition]['success'] for e in resets])[None]
        boot=[]
        for _ in range(cfg['bootstrap_samples']):
            si=rng.integers(len(seeds),size=len(seeds)); ei=rng.integers(len(resets),size=len(resets))
            boot.append(float(delta[si][:,ei].mean()))
        result.append(dict(kind=kind,width=width,horizon=horizon,condition=condition,
                           episodes=len(rr),auxiliary_seeds=0 if kind in ('policy','oracle') else len(seeds),
                           independent_resets=len(resets),success_rate=float(success.mean()),
                           paired_success_gain=float(delta.mean()),paired_gain_95ci=np.quantile(boot,[.025,.975]).tolist(),
                           mean_latency_seconds=float(np.mean([r['mean_latency_seconds'] for r in rr])),
                           mean_steps=float(np.mean([r['steps'] for r in rr]))))
    summary=dict(stage=cfg['stage'],episodes=len(rows),groups=result,
                 limitation='One released policy training seed; auxiliary/reset uncertainty only. Feasibility is not confirmatory.' if cfg['stage']=='feasibility' else 'One released policy training seed; auxiliary/reset uncertainty only. Restricted retrieval-extended proposals.',
                 peak_gpu_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
    save(root/'summary.json',summary); emit('FINAL_SUMMARY',**summary)
    report=['# GPU Push-T horizon study', '', f"Stage: {cfg['stage']}. Completed {len(rows)} evaluation episodes.", '',
            'This is a separate extension; published CPU results remain unchanged.', '',
            '| Method | Width | Lookahead | Condition | Success | Mean decision seconds |',
            '|---|---:|---:|---|---:|---:|']
    for r in sorted(result,key=lambda x:(x['condition'],x['kind'],x['width'],x['horizon'])):
        report.append(f"| {r['kind']} | {r['width']} | {r['horizon']} | {r['condition']} | {r['success_rate']:.3f} | {r['mean_latency_seconds']:.4f} |")
    report+=['','Intervals and paired differences are in [summary.json](summary.json).', '',
             'The frozen policy did not train on additional synthetic interactions; the matched direct scorer is the representation control. Longer proposals use training-only action retrieval after the first eight diffusion actions. Oracle performance diagnoses this proposal/scoring restriction.', '',
             'Physics shifts are zero-shot; this experiment does not test online adaptation. Simulator time pauses during inference. A loss does not establish that world models are unnecessary in general.', '',summary['limitation']]
    (root/'report.md').write_text('\n'.join(report),encoding='utf-8')


def main():
    sys.path.insert(0,str(REPO))
    from reproduction import load_config
    cfg=load_config('gpu_extension/config.json', 'Run the CUDA capacity/horizon/physics study')
    cfg_hash=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()[:10]
    root=Path(cfg['artifact_root'])/f"{cfg['stage']}-{cfg_hash}-{time.strftime('%Y%m%d-%H%M%S')}"
    root.mkdir(parents=True,exist_ok=False)
    save(root/'config.json',cfg)
    emit('configuration',artifact_directory=str(root),**cfg)
    assert torch.cuda.is_available(), 'A CUDA GPU is required; refusing silent CPU fallback'
    torch.set_num_threads(cfg['threads']); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.manual_seed(71); np.random.seed(71)
    emit('hardware',device=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,
         gpu_memory_mib=torch.cuda.get_device_properties(0).total_memory/2**20)
    source_files=[REPO/'gpu_extension/study.py',REPO/'gpu_extension/config.json',REPO/'gpu_extension/uv.lock',REPO/'assets.json']
    if (REPO/'gpu_extension/batched.py').exists():source_files.append(REPO/'gpu_extension/batched.py')
    provenance={str(p.relative_to(REPO)):digest(p) for p in source_files}
    save(root/'source_hashes.json',provenance); emit('source_provenance',sha256=provenance)
    cache=inputs(cfg); policy,original=load_policy(cache)
    data,library=dataset(cache,original,cfg,root)
    checks(policy,library,cfg,root)
    models=train_models(data,cfg,root)
    if cfg['stage']=='full-study':
        sys.path.insert(0,str(REPO))
        from gpu_extension.batched import run_full
        run_full(policy,library,models,cfg,root)
    else:
        rows=evaluate(policy,library,models,cfg,root)
        summarize(rows,cfg,root)


if __name__=='__main__':
    main()
