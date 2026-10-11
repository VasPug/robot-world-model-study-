"""Controlled CPU pilot: released Diffusion Policy plus learned candidate selectors."""
from __future__ import annotations
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import urllib.request
import zipfile

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/pusht-matplotlib')
sys.path.insert(0, str(Path(__file__).parent / 'vendor'))
import dill
import numpy as np
import pymunk
import torch
from torch import nn
import zarr
from omegaconf import OmegaConf
from diffusers import DDPMScheduler
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.policy.diffusion_transformer_lowdim_policy import DiffusionTransformerLowdimPolicy
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from diffusion_policy.env.pusht.pusht_env import pymunk_to_shapely


def emit(event, **kwargs):
    print(json.dumps({'event': event, **kwargs}, allow_nan=False), flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def fetch_inputs(cfg):
    cache = Path(cfg['input_cache'])
    cache.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((Path(__file__).resolve().parent/'assets.json').read_text())
    for name, spec in manifest.items():
        target = cache / name
        if not target.exists():
            tmp = target.with_suffix('.part')
            urllib.request.urlretrieve(spec['url'], tmp)
            assert sha256(tmp) == spec['sha256'], f'Checksum failed: {name}'
            tmp.rename(target)
        assert sha256(target) == spec['sha256'], f'Checksum failed: {name}'
    data = cache / 'dataset'
    if not (data / '.complete').exists():
        data.mkdir(exist_ok=True)
        with zipfile.ZipFile(cache / 'pusht.zip') as z:
            for entry in z.infolist():
                resolved = (data / entry.filename).resolve()
                assert resolved.is_relative_to(data.resolve())
            z.extractall(data)
        (data / '.complete').touch()
    emit('input_provenance', assets=manifest)
    return cache


def load_policy(cache):
    payload = torch.load(cache / 'transformer.ckpt', map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']
    pc = OmegaConf.to_container(cfg.policy, resolve=True)
    pc.pop('_target_')
    mc = pc.pop('model'); mc.pop('_target_')
    sc = pc.pop('noise_scheduler'); sc.pop('_target_')
    policy = DiffusionTransformerLowdimPolicy(model=TransformerForDiffusion(**mc),
                                              noise_scheduler=DDPMScheduler(**sc), **pc)
    policy.load_state_dict(payload['state_dicts']['ema_model'], strict=True)
    policy.eval().requires_grad_(False)
    return policy, OmegaConf.to_container(cfg, resolve=True)


class Geometry:
    """Known task reward, applied to predicted block keypoints (not simulator futures)."""
    def __init__(self, env):
        self.template = env.kp_manager.local_keypoint_map['block'].copy()
        self.centered = self.template - self.template.mean(0)
        self.shapes = env.block.shapes
        self.goal = pymunk_to_shapely(env._get_goal_pose_body(env.goal_pose), self.shapes)
        self.threshold = env.success_threshold

    def pose(self, kps):
        q = np.asarray(kps).reshape(-1, 2)
        qc = q - q.mean(0)
        dot = (self.centered * qc).sum()
        cross = (self.centered[:, 0] * qc[:, 1] - self.centered[:, 1] * qc[:, 0]).sum()
        theta = np.arctan2(cross, dot)
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s], [s, c]])
        pos = q.mean(0) - self.template.mean(0) @ rot.T
        return pos, theta

    def reward(self, kps):
        pos, theta = self.pose(kps)
        body = pymunk.Body(1, 1)
        body.position = pos.tolist()
        body.angle = float(theta)
        block = pymunk_to_shapely(body, self.shapes)
        return float(np.clip(self.goal.intersection(block).area / self.goal.area / self.threshold, 0, 1))

    def scores(self, trajectories):
        return np.array([max(self.reward(kp) for kp in traj) for traj in trajectories])


def make_env(seed, condition='standard', shift=40):
    env = PushTKeypointsEnv(legacy=True)
    env.seed(seed)
    env.reset()
    if condition == 'shifted_start':
        env.block.position += (shift, 0)
        env.space.reindex_shapes_for_body(env.block)
    obs = env._get_obs()[:20].astype(np.float32)
    return env, obs


def build_examples(cache, original_cfg, geo):
    paths = list((cache / 'dataset').rglob('*.zarr'))
    assert len(paths) == 1, paths
    z = zarr.open(str(paths[0]), mode='r')
    kps = z['data/keypoint'][:].reshape(-1, 18).astype(np.float32)
    state = z['data/state'][:]
    action = z['data/action'][:].astype(np.float32)
    ends = z['meta/episode_ends'][:]
    obs = np.concatenate([kps, state[:, :2]], -1).astype(np.float32)
    dc = original_cfg['task']['dataset']
    n = len(ends)
    # Exactly recover the official checkpoint's eligible 90-episode train split.
    rng = np.random.default_rng(dc['seed'])
    val_ids = rng.choice(n, min(max(1, round(n * dc['val_ratio'])), n-1), replace=False)
    eligible = np.setdiff1d(np.arange(n), val_ids)
    rng = np.random.default_rng(dc['seed'])
    eligible = np.sort(eligible[rng.choice(len(eligible), dc['max_train_episodes'], replace=False)])
    # Auxiliary fitting and validation both stay inside that original training set.
    aux_val = np.random.default_rng(2026).choice(eligible, 18, replace=False)
    train_ids = np.setdiff1d(eligible, aux_val)
    rewards = np.array([geo.reward(k) for k in kps], dtype=np.float32)
    x, yw, ys, labels = [], [], [], []
    for ep in eligible:
        start = 0 if ep == 0 else int(ends[ep-1])
        end = int(ends[ep])
        # Do not invent future labels by crossing episodes or padding termination.
        for t in range(start, end - 8):
            history = obs[[max(start, t-1), t]]
            future = obs[t+1:t+9]
            x.append(np.concatenate([(history / 256 - 1).ravel(), (action[t:t+8] / 256 - 1).ravel()]))
            yw.append(((future - obs[t]) / 50).ravel())
            ys.append([float(np.max(rewards[t+1:t+9]))])
            labels.append(ep)
    x, yw, ys, labels = map(np.asarray, (x, yw, ys, labels))
    tr = np.isin(labels, train_ids)
    split = {'original_policy_train_episodes': eligible.tolist(),
             'auxiliary_train_episodes': train_ids.tolist(), 'auxiliary_validation_episodes': sorted(aux_val.tolist()),
             'train_windows': int(tr.sum()), 'validation_windows': int((~tr).sum())}
    return x.astype(np.float32), yw.astype(np.float32), ys.astype(np.float32), tr, split


class Predictor(nn.Module):
    def __init__(self, output_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(56, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, output_dim))
    def forward(self, x):
        return self.net(x)


def train_auxiliary(cache, pcfg, cfg, geo, root):
    x, yw, ys, tr, split = build_examples(cache, pcfg, geo)
    emit('dataset_split', **split)
    (root / 'data_split.json').write_text(json.dumps(split, indent=2))
    metrics = {}
    for variant, targets in [('world_model_a', yw[:, :20]), ('world_model', yw), ('direct_scorer', ys)]:
        inputs = x.copy()
        if variant == 'world_model_a':
            inputs[:, 42:] = 0

        torch.manual_seed(cfg['seed'])
        model = Predictor(targets.shape[1], cfg['auxiliary_hidden'])
        xt, yt = torch.from_numpy(inputs[tr]), torch.from_numpy(targets[tr])
        xv, yv = torch.from_numpy(inputs[~tr]), torch.from_numpy(targets[~tr])
        opt = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
        best, weights, best_epoch = float('inf'), None, None
        tick, cpu = time.perf_counter(), time.process_time()
        for epoch in range(cfg['epochs']):
            model.train()
            indices = torch.randperm(len(xt))
            loss_total = 0.0
            for batch in indices.split(cfg['batch_size']):
                opt.zero_grad(set_to_none=True)
                loss = nn.functional.mse_loss(model(xt[batch]), yt[batch])
                loss.backward(); opt.step()
                loss_total += loss.item() * len(batch)
            model.eval()
            with torch.inference_mode():
                val_loss = nn.functional.mse_loss(model(xv), yv).item()
            if val_loss < best:
                best, weights, best_epoch = val_loss, copy.deepcopy(model.state_dict()), epoch + 1
            if epoch == 0 or (epoch + 1) % 10 == 0:
                emit('training', variant=variant, epoch=epoch+1, train_mse=loss_total/len(xt), validation_mse=val_loss)
        model.load_state_dict(weights); model.eval()
        with torch.inference_mode():
            pred = model(xv).numpy()
        met = {'parameters': sum(p.numel() for p in model.parameters()), 'best_epoch': best_epoch,
               'validation_mse': best, 'training_wall_seconds': time.perf_counter()-tick,
               'training_cpu_seconds': time.process_time()-cpu}
        if variant in ('world_model', 'world_model_a'):
            horizon = 1 if variant == 'world_model_a' else 8
            base = (x[~tr, 20:40] + 1) * 256
            predicted = pred.reshape(-1, horizon, 20)*50 + base[:, None]
            truth = targets[~tr].reshape(-1, horizon, 20)*50 + base[:, None]
            met['validation_keypoint_rmse_pixels'] = float(np.sqrt(np.mean((predicted-truth)**2)))
            if variant == 'world_model':
                met['validation_score_mae'] = float(np.mean(np.abs(geo.scores(predicted[:, :, :18]) - ys[~tr, 0])))
        else:
            met['validation_score_mae'] = float(np.mean(np.abs(np.clip(pred[:, 0], 0, 1)-ys[~tr, 0])))
        torch.save({'state_dict': weights, 'output_dim': targets.shape[1], 'hidden': cfg['auxiliary_hidden'], 'metrics': met}, root / f'{variant}.pt')
        metrics[variant] = met
        emit('auxiliary_final', variant=variant, **met)
    (root / 'training_metrics.json').write_text(json.dumps(metrics, indent=2))


def load_aux(root, variant):
    p = torch.load(root / f'{variant}.pt', map_location='cpu', weights_only=True)
    model = Predictor(p['output_dim'], p['hidden'])
    model.load_state_dict(p['state_dict'])
    model.eval().requires_grad_(False)
    return model


@torch.inference_mode()
def candidates(policy, history, n, seed):
    # Candidate zero is identical for K=1 and K=4 at an identical history.
    torch.manual_seed(seed)
    h = torch.as_tensor(np.asarray(history), dtype=torch.float32).unsqueeze(0)
    first = policy.predict_action({'obs': h})['action'].numpy()
    if n == 1:
        return first
    torch.manual_seed(seed+1)
    rest = policy.predict_action({'obs': h.repeat(n-1, 1, 1)})['action'].numpy()
    return np.concatenate([first, rest], 0)


def features(history, actions):
    return torch.from_numpy(np.concatenate([
        np.tile((np.asarray(history, dtype=np.float32)/256-1).ravel(), (len(actions), 1)),
        (actions/256-1).reshape(len(actions), -1)], -1).astype(np.float32))


def cluster_interval(rows, key):
    groups = sorted(set(r['env_seed'] for r in rows))
    v = np.array([np.mean([r[key] for r in rows if r['env_seed']==g]) for g in groups])
    rng = np.random.default_rng(9191)
    boot = v[rng.integers(0, len(v), (5000, len(v)))].mean(1)
    return np.quantile(boot, [.025,.975]).tolist()


def evaluate(policy, cfg, root):
    variant = cfg['variant']
    aux = None if variant == 'baseline' else load_aux(Path(cfg['auxiliary_dir']), variant)
    rows, decisions = [], []
    dest = root / variant
    dest.mkdir(exist_ok=True)
    # Rotate ordering per variant to reduce systematic warm-machine timing effects.
    jobs = [(cfg['test_start_seed']+i, c) for i in range(cfg['test_seeds']) for c in cfg['conditions']]
    for env_seed, condition in jobs:
        env, obs = make_env(env_seed, condition, cfg['shift_pixels'])
        geo = Geometry(env)
        history = [obs.copy(), obs.copy()]
        rewards, latencies, cpu_times = [], [], []
        step, decision, success = 0, 0, False
        start = time.perf_counter()
        while step < cfg['max_steps'] and not success:
            seed = cfg['seed'] * 10000000 + (env_seed-cfg['test_start_seed'])*10000 + decision*10
            tic, cpu = time.perf_counter(), time.process_time()
            actions = candidates(policy, history, 1 if variant=='baseline' else cfg['candidates'], seed)
            proposal_seconds = time.perf_counter()-tic
            select_tic = time.perf_counter()
            pred_future, predicted_scores = None, None
            chosen = 0
            if aux is not None:
                with torch.inference_mode():
                    output = aux(features(history, actions)).numpy() if variant != 'world_model_a' else None
                if variant == 'world_model':
                    pred_future = (output.reshape(-1, 8, 20)*50 + history[-1][None,None,:])[:, :, :18]
                    predicted_scores = geo.scores(pred_future)
                elif variant == 'world_model_a':
                    with torch.inference_mode():
                        h = np.tile(np.asarray(history)[None], (len(actions),1,1))
                        futures = []
                        for t in range(8):
                            xx = np.zeros((len(actions),56), dtype=np.float32)
                            xx[:,:40] = (h/256-1).reshape(len(actions),40)
                            xx[:,40:42] = actions[:,t]/256-1
                            nxt = h[:,-1] + aux(torch.from_numpy(xx)).numpy()*50
                            futures.append(nxt[:,:18])
                            h = np.stack([h[:,-1],nxt],1)
                    pred_future = np.stack(futures,1)
                    predicted_scores = geo.scores(pred_future)
                else:
                    predicted_scores = np.clip(output[:,0], 0, 1)
                chosen = int(np.argmax(predicted_scores))
            selection_seconds = time.perf_counter()-select_tic
            latency = time.perf_counter()-tic
            cpu_seconds = time.process_time()-cpu
            latencies.append(latency); cpu_times.append(cpu_seconds)
            observed_future, chunk_rewards = [], []
            for action in actions[chosen]:
                if step >= cfg['max_steps'] or success:
                    break
                full, reward, done, _ = env.step(action)
                obs = full[:20].astype(np.float32)
                history = [history[-1], obs]
                observed_future.append(obs[:18]); chunk_rewards.append(float(reward))
                rewards.append(float(reward)); step += 1; success = bool(done)
            d = {'env_seed':env_seed,'condition':condition,'decision':decision,'chosen':chosen,
                 'candidate_count':len(actions),'latency_seconds':latency,'cpu_seconds':cpu_seconds,
                 'proposal_seconds':proposal_seconds,'selection_seconds':selection_seconds,
                 'observed_chunk_max_reward':max(chunk_rewards)}
            if predicted_scores is not None:
                d['predicted_score'] = float(predicted_scores[chosen])
                d['score_absolute_error'] = abs(d['predicted_score']-d['observed_chunk_max_reward'])
            if pred_future is not None:
                count = len(observed_future)
                d['keypoint_rmse_pixels'] = float(np.sqrt(np.mean((pred_future[chosen,:count]-np.asarray(observed_future))**2)))
            decisions.append(d); decision += 1
        row = {'variant':variant,'env_seed':env_seed,'condition':condition,'success':int(success),
               'max_reward':max(rewards),'steps':step,'simulated_seconds':step/10,
               'episode_wall_seconds':time.perf_counter()-start,'decisions':decision,
               'mean_decision_latency_seconds':float(np.mean(latencies)),
               'p95_decision_latency_seconds':float(np.quantile(latencies,.95)),
               'decision_cpu_seconds':sum(cpu_times)}
        rows.append(row); emit('episode', **row)
        (dest / 'episodes.json').write_text(json.dumps(rows,indent=2))
        with open(dest / 'decisions.jsonl','w') as f:
            for d in decisions: f.write(json.dumps(d)+'\n')
        env.close()
    all_lat = np.array([d['latency_seconds'] for d in decisions])
    summary = {'variant':variant,'episodes':len(rows),'independent_reset_seeds':cfg['test_seeds'],
               'policy_training_seeds':1,'auxiliary_training_seeds':0 if variant=='baseline' else 1,
               'success_rate':float(np.mean([r['success'] for r in rows])),
               'success_rate_cluster_bootstrap_95ci':cluster_interval(rows,'success'),
               'mean_max_reward':float(np.mean([r['max_reward'] for r in rows])),
               'mean_max_reward_cluster_bootstrap_95ci':cluster_interval(rows,'max_reward'),
               'mean_decision_latency_seconds':float(all_lat.mean()),
               'p95_decision_latency_seconds':float(np.quantile(all_lat,.95)),
               'total_decision_cpu_seconds':sum(d['cpu_seconds'] for d in decisions),
               'mean_simulated_seconds':float(np.mean([r['simulated_seconds'] for r in rows])),
               'mean_episode_wall_seconds':float(np.mean([r['episode_wall_seconds'] for r in rows])),
               'peak_process_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
               'strata':{}}
    for c in cfg['conditions']:
        rr = [r for r in rows if r['condition']==c]
        summary['strata'][c] = {'success_rate':float(np.mean([r['success'] for r in rr])),
                              'mean_max_reward':float(np.mean([r['max_reward'] for r in rr]))}
    if aux is not None:
        summary['mean_selected_score_absolute_error'] = float(np.mean([d['score_absolute_error'] for d in decisions]))
    if variant in ('world_model','world_model_a'):
        summary['mean_selected_keypoint_rmse_pixels'] = float(np.mean([d['keypoint_rmse_pixels'] for d in decisions]))
    (dest / 'summary.json').write_text(json.dumps(summary,indent=2))
    (dest / 'config.json').write_text(json.dumps(cfg,indent=2))
    with open(dest / 'episodes.csv','w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    emit('FINAL_SUMMARY', **summary)


def main():
    from reproduction import load_config, require_empty
    cfg=load_config('config.json', 'Run a CPU Push-T pilot variant', auxiliary=True)
    require_empty(Path(cfg['artifact_root'])/cfg['variant'])
    if cfg['variant']=='baseline':
        for name in ('world_model_a.pt','world_model.pt','direct_scorer.pt','training_metrics.json'):
            if (Path(cfg['artifact_root'])/name).exists():
                raise FileExistsError('Auxiliary training outputs already exist. Choose a new --output-dir.')
    torch.set_num_threads(cfg['threads']);torch.set_num_interop_threads(1)
    np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    root=Path(cfg['artifact_root']);root.mkdir(parents=True,exist_ok=True)
    emit('configuration', **cfg)
    emit('hardware', device='cpu', logical_cpus=os.cpu_count(), torch=torch.__version__, threads=torch.get_num_threads())
    cache=fetch_inputs(cfg)
    policy, pcfg=load_policy(cache)
    assert policy.num_inference_steps==cfg['num_inference_steps']==100
    assert policy.n_action_steps==cfg['execution_horizon']==8
    emit('policy', parameters=sum(p.numel() for p in policy.parameters()), training_seed=pcfg['training']['seed'],
         weights='released EMA checkpoint', inference_steps=policy.num_inference_steps)
    # Warmup excluded from timing; this is an infrastructure check, not an evaluation trial.
    env,obs=make_env(90210)
    geo=Geometry(env)
    probe=candidates(policy,[obs,obs],1,999)
    assert probe.shape==(1,8,2) and np.isfinite(probe).all()
    _,reward,_,_=env.step(probe[0,0])
    reconstructed=geo.reward(env._get_obs()[:18])
    assert abs(reward-reconstructed)<1e-6,(reward,reconstructed)
    emit('sanity', action_shape=list(probe.shape), geometry_reward_error=abs(reward-reconstructed))
    if cfg['variant']=='baseline':
        train_auxiliary(cache,pcfg,cfg,geo,root)
    else:
        auxiliary=Path(cfg['auxiliary_dir'])
        actual_hash=sha256(auxiliary/f"{cfg['variant']}.pt")
        expected=cfg.get('expected_auxiliary_sha256')
        if expected is not None and actual_hash != expected:
            raise ValueError('Released auxiliary checkpoint checksum mismatch')
        emit('auxiliary_artifacts', sha256=actual_hash, reference_weights=expected is not None,
             training_metrics=json.loads((auxiliary/'training_metrics.json').read_text())[cfg['variant']])
    env.close()
    evaluate(policy,cfg,root)


if __name__=='__main__':
    main()
