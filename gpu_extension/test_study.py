"""Lightweight correctness checks; no training or experiment evaluation."""
import unittest
import numpy as np
import torch
from unittest.mock import patch
from diffusers import DDPMScheduler
from gpu_extension.batched import coefficients, ddpm_step, sample_batch, evaluate_batch
from gpu_extension.study import load_policy, Predictor
from pathlib import Path
from io import StringIO
from gpu_extension.study import Geometry, make_env, observe, rollout, oracle_scores, features, parameter_count, scorer_width


class CorrectnessTests(unittest.TestCase):
    def test_geometry_matches_simulator(self):
        for seed in (123,456):
            env=make_env(seed,'recovery'); geo=Geometry(env)
            actions=np.linspace([250,420],[250,240],8)
            obs,reward=rollout(env,actions)
            np.testing.assert_allclose(geo.rewards(obs),reward,atol=1e-6)
            env.close()

    def test_physics_interventions_are_effective(self):
        states={}
        for condition in ('recovery','friction_shift','damping_shift'):
            env=make_env(9876,condition,state=np.array([256,420,256,280,0.0]))
            path=np.concatenate([np.linspace([256,420],[256,200],32),np.linspace([256,200],[350,220],32)])
            states[condition],_=rollout(env,path,absorbing=False); env.close()
        for condition in ('friction_shift','damping_shift'):
            self.assertGreater(float(np.max(np.abs(states[condition]-states['recovery']))),0.01)

    def test_replay_matches_contact_history(self):
        for condition in ('recovery','friction_shift','damping_shift'):
            env=make_env(432,condition)
            past=np.linspace([300,450],[256,300],8)
            future=np.linspace([256,300],[256,200],8)
            for action in past: env.step(action)
            _,truth=rollout(env,future); env.close()
            pred=oracle_scores(432,condition,past,future[None],8)
            self.assertAlmostEqual(float(pred[0]),float(truth.max()),places=7)

    def test_shift_pairing_and_nontrivial_resets(self):
        for seed in range(720000,720110):
            nominal=make_env(seed,'recovery'); a=observe(nominal)
            geo=Geometry(nominal)
            self.assertLess(float(geo.rewards(a[None])[0]),1)
            nominal.close()
            for condition in ('friction_shift','damping_shift'):
                env=make_env(seed,condition)
                np.testing.assert_array_equal(a,observe(env)); env.close()
            easy=make_env(seed,'easy')
            self.assertLess(float(Geometry(easy).rewards(observe(easy)[None])[0]),1)
            easy.close()

    def test_parameter_matching(self):
        for width in (128,512,2560):
            wm=parameter_count(width,64*20); direct=parameter_count(scorer_width(width),1)
            self.assertLess(abs(wm-direct)/wm,0.005)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_no_future_action_leakage(self):
        h=np.zeros((2,2,20),dtype=np.float32)
        a=np.zeros((2,64,2),dtype=np.float32)
        x,_=features(h,a,8)
        a[:,8:]=500
        y,_=features(h,a,8)
        self.assertTrue(torch.equal(x,y))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_batched_ddpm_matches_pinned_scheduler(self):
        scheduler=DDPMScheduler(num_train_timesteps=100,beta_schedule='squaredcos_cap_v2',
                                variance_type='fixed_small',prediction_type='epsilon',clip_sample=True)
        scheduler.set_timesteps(100)
        coefs=coefficients(scheduler)
        generator=torch.Generator(device='cuda').manual_seed(113)
        for index in (0,49,98,99):
            timestep=scheduler.timesteps[index]
            sample=torch.randn((2,10,2),device='cuda',generator=generator)
            epsilon=torch.randn((2,10,2),device='cuda',generator=generator)
            noise=torch.randn((2,10,2),device='cuda',generator=generator)
            with patch('diffusers.schedulers.scheduling_ddpm.randn_tensor',return_value=noise):
                expected=scheduler.step(epsilon,timestep,sample).prev_sample
            actual=ddpm_step(sample,epsilon,noise,coefs[index],1.0)
            torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_policy_candidate_zero_independent_of_batch(self):
        import os
        cache=Path(os.environ.get('PUSHT_CACHE_DIR', str(Path(__file__).resolve().parents[1]/'.cache/pusht')))
        if not (cache/'transformer.ckpt').exists():self.skipTest('Cached checkpoint unavailable')
        policy,_=load_policy(cache)
        env=make_env(9817);obs=observe(env);env.close()
        histories=np.repeat(np.array([[obs,obs]]),2,axis=0)
        single=sample_batch(policy,histories[:1],[111],1)
        multiple=sample_batch(policy,histories,[111,112],2)
        np.testing.assert_allclose(single[0,0],multiple[0,0],atol=.05,rtol=0)
        reversed_batch=sample_batch(policy,histories[::-1],[112,111],2)
        np.testing.assert_allclose(multiple[0],reversed_batch[1],atol=.05,rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_batched_episode_bookkeeping(self):
        # Deterministic synthetic policy fixtures; this is not a performance run.
        def fixed_chunks(policy,histories,seeds,count):
            return np.full((len(histories),count,8,2),250,dtype=np.float32)
        library={'obs':np.zeros((2,20),dtype=np.float32),
                 'actions':np.full((2,64,2),250,dtype=np.float32)}
        cfg={'max_steps':8,'candidates':2,'execution_horizon':8}
        jobs=[(810000,'recovery','validation'),(810001,'friction_shift','validation')]
        latency={'mean_seconds':.1,'p95_seconds':.2}
        for kind in ('policy','world_model','direct_scorer'):
            model=None if kind=='policy' else Predictor(16,1280 if kind=='world_model' else 1).cuda().eval()
            arm={'kind':kind,'width':16,'seed':11,'horizon':32}
            episodes,decisions=StringIO(),StringIO()
            with patch('gpu_extension.batched.sample_batch',side_effect=fixed_chunks):
                rows=evaluate_batch(None,library,model,arm,jobs,cfg,latency,episodes,decisions)
            self.assertEqual(len(rows),2)
            self.assertTrue(all(r['steps']==8 and r['decisions']==1 for r in rows))
            self.assertTrue(all(r['split']=='validation' for r in rows))
            self.assertEqual(len(episodes.getvalue().splitlines()),2)
            self.assertEqual(len(decisions.getvalue().splitlines()),2)


if __name__=='__main__':
    unittest.main()
