"""Small fully observed CPU navigation study; original experimental simulator, not a paper benchmark."""
from __future__ import annotations
import copy
import csv
import json
from collections import deque
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn

SIZE=11
H=6
DELTAS=np.array([[0,-1],[1,0],[0,1],[-1,0],[0,0]],dtype=int)
METHODS=['baseline','world_model_a','world_model_b','direct_scorer']


def log(event,**kw):
    print(json.dumps({'event':event,**kw},allow_nan=False),flush=True)


class Env:
    def __init__(self,scenario,seed):
        rng=np.random.default_rng(seed)
        self.scenario=scenario
        self.wall=np.zeros((SIZE,SIZE),dtype=bool)
        self.wall[[0,-1],:]=True;self.wall[:,[0,-1]]=True
        self.door=None;self.key=None
        if scenario in ('detour','key_door','rotated_controls'):
            if scenario=='key_door':
                self.wall[1:-1,5]=True
                self.door=(5,int(rng.integers(2,9)))
                self.wall[self.door[1],self.door[0]]=False
                self.key=(int(rng.integers(1,5)),int(rng.integers(1,10)))
            else:
                # Two staggered gaps force long detours and context-dependent turns.
                for x in (4,7):
                    self.wall[1:-1,x]=True
                self.wall[int(rng.integers(1,4)),4]=False
                self.wall[int(rng.integers(7,10)),7]=False
        self.pos=np.array([int(rng.integers(1,4)),int(rng.integers(1,10))])
        self.goal=np.array([int(rng.integers(8,10)),int(rng.integers(1,10))])
        self.has_key=int(self.key is None)
        if self.key==tuple(self.pos): self.has_key=1
        self.last_action=4;self.last_delta=np.zeros(2,dtype=int)
        self.rotation=1 if scenario=='rotated_controls' else 0
        self.steps=0

    def blocked(self,pos,has_key):
        x,y=map(int,pos)
        return not(0<=x<SIZE and 0<=y<SIZE) or self.wall[y,x] or (self.door==(x,y) and not has_key)

    def transition(self,pos,has_key,action):
        delta=DELTAS[(int(action)+self.rotation)%4]
        nxt=np.asarray(pos)+delta
        if self.blocked(nxt,has_key): nxt=np.asarray(pos).copy()
        key=int(has_key or self.key==tuple(nxt))
        return nxt,key

    def step(self,action):
        old=self.pos.copy()
        self.pos,self.has_key=self.transition(self.pos,self.has_key,action)
        self.last_action=int(action);self.last_delta=self.pos-old;self.steps+=1
        return bool(np.array_equal(self.pos,self.goal))

    def observation(self,pos=None,has_key=None,last_action=None,last_delta=None):
        pos=self.pos if pos is None else np.asarray(pos)
        key=self.has_key if has_key is None else int(has_key)
        act=self.last_action if last_action is None else int(last_action)
        delta=self.last_delta if last_delta is None else np.asarray(last_delta)
        target=np.asarray(self.key) if not key else self.goal
        local=[]
        for dy in (-1,0,1):
            for dx in (-1,0,1):local.append(float(self.blocked(pos+[dx,dy],key)))
        keypos=np.asarray(self.key) if self.key is not None else self.goal
        door=np.asarray(self.door) if self.door is not None else self.goal
        return np.concatenate([self.wall.astype(float).ravel(),pos/10,self.goal/10,keypos/10,door/10,
            [key],(target-pos)/10,local,np.eye(5)[act],delta]).astype(np.float32)

    def score(self,pos,has_key):
        target=np.asarray(self.key) if not has_key else self.goal
        # Fixed transparent short-horizon objective. It does not use BFS/oracle distances.
        stage=20*has_key if self.key is not None else 0
        return (stage-np.abs(np.asarray(pos)-target).sum())/40

    def expert(self,pos=None,has_key=None):
        pos=tuple(self.pos if pos is None else pos)
        key=self.has_key if has_key is None else has_key
        start=(pos[0],pos[1],int(key));q=deque([start]);parents={start:None}
        end=None
        while q:
            x,y,k=q.popleft()
            if (x,y)==tuple(self.goal):end=(x,y,k);break
            for a in range(4):
                p,kk=self.transition((x,y),k,a);n=(int(p[0]),int(p[1]),int(kk))
                if n not in parents:parents[n]=((x,y,k),a);q.append(n)
        assert end is not None
        actions=[]
        while parents[end] is not None:
            end,a=parents[end];actions.append(a)
        return actions[::-1]


OBS=len(Env('open',0).observation())


class Net(nn.Module):
    def __init__(self,inp,out,hidden):
        super().__init__();self.layers=nn.Sequential(nn.Linear(inp,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,out))
    def forward(self,x):return self.layers(x)


def collect(scenario,seeds,rng,exploration=.25):
    records=[]
    for seed in seeds:
        env=Env(scenario,int(seed));trajectory=[]
        for t in range(70):
            plan=env.expert()
            if not plan:break
            label=np.array((plan+[plan[-1]]*H)[:H],dtype=np.int64)
            action=int(rng.integers(4)) if rng.random()<exploration else int(plan[0])
            trajectory.append((env.observation(),env.pos.copy(),action,label))
            done=env.step(action)
            if done:break
        trajectory.append((env.observation(),env.pos.copy(),0,np.zeros(H,dtype=np.int64)))
        for t in range(len(trajectory)-1):
            obs,pos,_,label=trajectory[t]
            actions=np.array([trajectory[min(t+j,len(trajectory)-2)][2] for j in range(H)])
            futures=np.array([trajectory[min(t+j+1,len(trajectory)-1)][1] for j in range(H)])
            delta=futures[0]-pos
            cls=int(np.flatnonzero(np.all(DELTAS==delta,axis=1))[0])
            key=int(obs[129]);scores=[]
            for p in futures:
                key=int(key or env.key==tuple(p));scores.append(env.score(p,key))
            records.append((obs,actions,label,cls,((futures-pos)/H).astype(np.float32),max(scores)))
    return records


def arrays(records):
    obs=np.stack([r[0] for r in records]);actions=np.stack([r[1] for r in records])
    aa=np.eye(4,dtype=np.float32)[actions].reshape(-1,H*4)
    return {'baseline':(obs,np.stack([r[2] for r in records])),
            'world_model_a':(np.concatenate([obs,aa[:,:4]],1),np.array([r[3] for r in records])),
            'world_model_b':(np.concatenate([obs,aa],1),np.stack([r[4] for r in records]).reshape(-1,H*2)),
            'direct_scorer':(np.concatenate([obs,aa],1),np.array([[r[5]] for r in records],dtype=np.float32))}


def fit_models(records,cfg,seed,initial=None,epochs=None,replay=None):
    torch.manual_seed(seed)
    models={};metrics={}
    data=arrays(records)
    for name,(x,y) in data.items():
        dims={'baseline':H*4,'world_model_a':5,'world_model_b':H*2,'direct_scorer':1}
        model=Net(x.shape[1],dims[name],cfg['hidden']) if initial is None else copy.deepcopy(initial[name])
        model.train().requires_grad_(True)
        x=torch.from_numpy(x);y=torch.from_numpy(y)
        opt=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate']*(1 if initial is None else .5))
        tic,cpu=time.perf_counter(),time.process_time()
        for epoch in range(epochs or cfg['epochs']):
            ix=torch.randperm(len(x));total=0
            for ids in ix.split(cfg['batch_size']):
                out=model(x[ids])
                if name=='baseline':loss=nn.functional.cross_entropy(out.reshape(-1,4),y[ids].reshape(-1))
                elif name=='world_model_a':loss=nn.functional.cross_entropy(out,y[ids])
                else:loss=nn.functional.mse_loss(out,y[ids])
                opt.zero_grad(set_to_none=True);loss.backward();opt.step();total+=float(loss)*len(ids)
        model.eval().requires_grad_(False)
        models[name]=model
        metrics[name]={'loss':total/len(x),'examples':len(x),'parameters':sum(p.numel() for p in model.parameters()),
                       'training_wall_seconds':time.perf_counter()-tic,'training_cpu_seconds':time.process_time()-cpu}
        log('training',seed=seed,method=name,adaptation=initial is not None,**metrics[name])
    return models,metrics


@torch.inference_mode()
def propose(models,env,rng,cfg,method):
    obs=env.observation()
    logits=models['baseline'](torch.from_numpy(obs[None])).reshape(H,4)
    probs=torch.softmax(logits/.8,dim=-1).numpy()
    n=1 if method=='baseline' else cfg['candidates']
    actions=np.stack([np.argmax(probs,1)]+[np.array([rng.choice(4,p=p) for p in probs]) for _ in range(n-1)])
    # Candidates include the baseline's proposal. All selectors get this same pool.
    return obs,actions,probs


@torch.inference_mode()
def choose(models,env,obs,actions,method):
    if method=='baseline':return int(actions[0,0]),0,None
    n=len(actions)
    if method=='world_model_a':
        positions=np.tile(env.pos,(n,1));keys=np.full(n,env.has_key);prev=np.tile(env.last_delta,(n,1));last=np.full(n,env.last_action)
        values=np.full(n,-np.inf);futures=[]
        for t in range(H):
            xx=np.stack([env.observation(p,k,a,d) for p,k,a,d in zip(positions,keys,last,prev)])
            xx=np.concatenate([xx,np.eye(4,dtype=np.float32)[actions[:,t]]],1)
            cls=models[method](torch.from_numpy(xx)).argmax(-1).numpy()
            delta=DELTAS[cls]
            positions=np.clip(positions+delta,1,SIZE-2)
            keys=np.array([int(k or env.key==tuple(p)) for k,p in zip(keys,positions)])
            values=np.maximum(values,[env.score(p,k) for p,k in zip(positions,keys)])
            futures.append(positions.copy());last=actions[:,t];prev=delta
        predicted=np.stack(futures,1)
    else:
        xx=np.concatenate([np.tile(obs,(n,1)),np.eye(4,dtype=np.float32)[actions].reshape(n,-1)],1)
        out=models[method](torch.from_numpy(xx)).numpy()
        if method=='direct_scorer':values=out[:,0];predicted=None
        else:
            predicted=np.clip(np.rint(out.reshape(n,H,2)*H+env.pos),1,SIZE-2).astype(int)
            keys=np.full(n,env.has_key);values=np.full(n,-np.inf)
            for t in range(H):
                keys=np.array([int(k or env.key==tuple(p)) for k,p in zip(keys,predicted[:,t])])
                values=np.maximum(values,[env.score(p,k) for p,k in zip(predicted[:,t],keys)])
    best=int(np.argmax(values))
    return int(actions[best,0]),best,(None if predicted is None else predicted[best,0].copy())


def initial_features(env,models):
    with torch.inference_mode():p=torch.softmax(models['baseline'](torch.from_numpy(env.observation()[None])).reshape(H,4)[0],-1).numpy()
    return [float(env.wall.mean()),float(np.abs(env.pos-env.goal).sum()/20),float(env.key is not None),
            float(-(p*np.log(p+1e-12)).sum()),float(env.pos[1]/10)]


def episode(models,scenario,reset_seed,method,seed,cfg):
    env=Env(scenario,reset_seed);optimal=len(env.expert());features=initial_features(env,models)
    rng=np.random.default_rng(seed*100000+reset_seed)
    times=[];cpus=[];errors=[];bumps=0;success=False
    tic=time.perf_counter()
    for step in range(cfg['max_steps']):
        t,c=time.perf_counter(),time.process_time()
        obs,actions,probs=propose(models,env,rng,cfg,method)
        # Baseline generates only its own action; selectors pay candidate-generation cost.
        action,chosen,pred=choose(models,env,obs,actions,method)
        times.append(time.perf_counter()-t);cpus.append(time.process_time()-c)
        old=env.pos.copy();success=env.step(action);bumps+=int(np.array_equal(old,env.pos))
        if pred is not None:errors.append(float(np.linalg.norm(pred-env.pos)))
        if success:break
    return {'training_seed':seed,'scenario':scenario,'reset_seed':reset_seed,'method':method,
        'success':int(success),'steps':env.steps,'optimal_steps':optimal,'success_weighted_path_efficiency':float(success*optimal/max(optimal,env.steps)),
        'blocked_moves':bumps,'inference_seconds':sum(times),'inference_cpu_seconds':sum(cpus),
        'mean_latency_seconds':float(np.mean(times)),'p95_latency_seconds':float(np.quantile(times,.95)),
        'wall_seconds':time.perf_counter()-tic,'prediction_mae_cells':float(np.mean(errors)) if errors else None,
        'features':features}


def gate_features(rows):
    return np.array([r['features']+[r['shots']/5] for r in rows],dtype=float)


def fit_ridge(x,y):
    mean=x.mean(0);scale=np.maximum(x.std(0),.1)
    xx=np.c_[np.ones(len(x)),(x-mean)/scale]
    reg=np.eye(xx.shape[1])*5;reg[0,0]=1e-6
    return np.linalg.solve(xx.T@xx+reg,xx.T@y),mean,scale


def gate(rows,cfg):
    # Episode-level choice, made before any model rollout. Fit only on validation resets.
    gate_rows=[]
    for seed in cfg['training_seeds']:
        for lam in cfg['lambdas_per_second']:
            validation=[r for r in rows if r['split']=='validation' and r['training_seed']==seed]
            tests=[r for r in rows if r['split']=='test' and r['training_seed']==seed]
            params={}
            for method in METHODS:
                rr=[r for r in validation if r['method']==method]
                xx=gate_features(rr);yy=np.array([r['success']-lam*r['inference_seconds'] for r in rr])
                params[method]=fit_ridge(xx,yy)
            base=[r for r in tests if r['method']=='baseline']
            lookup={(r['scenario'],r['shots'],r['reset_seed'],r['method']):r for r in tests}
            for r in base:
                x=gate_features([r])[0]
                predictions={}
                for m,(w,mean,scale) in params.items():predictions[m]=float(np.r_[1,(x-mean)/scale]@w)
                chosen=max(predictions,key=predictions.get)
                selected=lookup[(r['scenario'],r['shots'],r['reset_seed'],chosen)]
                gate_rows.append({'training_seed':seed,'scenario':r['scenario'],'shots':r['shots'],'reset_seed':r['reset_seed'],
                                  'lambda_per_second':lam,'chosen_method':chosen,'success':selected['success'],
                                  'inference_seconds':selected['inference_seconds'],
                                  'utility':selected['success']-lam*selected['inference_seconds']})
    return gate_rows


def summarize(rows,cfg):
    summaries=[]
    for scenario in cfg['scenarios']:
        for shots in cfg['adaptation_episodes']:
            for method in METHODS:
                rr=[r for r in rows if r['split']=='test' and r['scenario']==scenario and r['shots']==shots and r['method']==method]
                out={'scenario':scenario,'shots':shots,'method':method,'n':len(rr)}
                for key in ['success','steps','success_weighted_path_efficiency','inference_seconds','mean_latency_seconds','p95_latency_seconds','blocked_moves']:
                    out[key]=float(np.mean([r[key] for r in rr]))
                out['success_by_training_seed']=[float(np.mean([r['success'] for r in rr if r['training_seed']==s])) for s in cfg['training_seeds']]
                summaries.append(out);log('RESULT',**out)
    return summaries


def main():
    from reproduction import load_config, require_empty
    cfg=load_config('navigation_config.json', 'Run the CPU navigation study')
    require_empty(cfg['artifact_root'])
    assert cfg['horizon']==H
    torch.set_num_threads(cfg['threads']);torch.set_num_interop_threads(1)
    root=Path(cfg['artifact_root']);root.mkdir(parents=True,exist_ok=True)
    (root/'config.json').write_text(json.dumps(cfg,indent=2))
    log('configuration',**cfg,observation_dimension=OBS)
    rows=[];training=[]
    for seed in cfg['training_seeds']:
        rng=np.random.default_rng(seed)
        train=[]
        for scenario in ('open','detour','key_door'):
            train+=collect(scenario,np.arange(seed*1000,seed*1000+cfg['train_layouts_per_scenario']),rng)
        models,met=fit_models(train,cfg,seed);training.append({'seed':seed,'shots':0,'metrics':met})
        replay_idx=rng.choice(len(train),min(500,len(train)),replace=False)
        replay=[train[i] for i in replay_idx]
        for scenario in cfg['scenarios']:
            # Adaptation demonstrations are disjoint from train, validation, and test resets.
            for shots in cfg['adaptation_episodes']:
                if shots:
                    adaptation=collect(scenario,np.arange(300000+seed*10,300000+seed*10+shots),rng,exploration=0)
                    # Identical experience and replay budget for every arm; duplicate to give new data weight.
                    records=adaptation*max(1,500//max(1,len(adaptation)))+replay
                    adapted,mm=fit_models(records,cfg,seed,initial=models,epochs=cfg['adaptation_epochs'])
                    training.append({'seed':seed,'scenario':scenario,'shots':shots,'adaptation_windows':len(adaptation),'metrics':mm})
                else:adapted=models
                for split,start,count in [('validation',400000,cfg['validation_layouts']),('test',500000,cfg['test_layouts'])]:
                    for j in range(count):
                        for method in METHODS[(j+seed)%4:]+METHODS[:(j+seed)%4]:
                            row=episode(adapted,scenario,start+j,method,seed,cfg)
                            row.update({'shots':shots,'split':split});rows.append(row)
                    log('evaluation_block',seed=seed,scenario=scenario,shots=shots,split=split,episodes=count*len(METHODS))
                with open(root/'episodes.jsonl','w') as f:
                    for r in rows:f.write(json.dumps(r)+'\n')
    sums=summarize(rows,cfg)
    (root/'summary.json').write_text(json.dumps(sums,indent=2))
    (root/'training_metrics.json').write_text(json.dumps(training,indent=2))
    gates=gate(rows,cfg)
    (root/'gate_results.json').write_text(json.dumps(gates,indent=2))
    for lam in cfg['lambdas_per_second']:
        rr=[r for r in gates if r['lambda_per_second']==lam]
        log('GATE_RESULT',lambda_per_second=lam,success=float(np.mean([r['success'] for r in rr])),
            utility=float(np.mean([r['utility'] for r in rr])),inference_seconds=float(np.mean([r['inference_seconds'] for r in rr])),
            choices={m:sum(r['chosen_method']==m for r in rr) for m in METHODS})
    keys=[k for k in rows[0] if k!='features']
    with open(root/'episodes.csv','w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in rows)
    log('FINAL_SUMMARY',total_evaluation_episodes=len(rows),test_episodes=sum(r['split']=='test' for r in rows),
        training_seeds=cfg['training_seeds'],gate='validation-fitted episode-level ridge selection; no per-step gating')


if __name__=='__main__':main()
