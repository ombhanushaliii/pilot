import unittest
import numpy as np
import torch
from cf_pilot.deep_rl import Replay, QNetwork, action_masks, infer, learn, observations, rewards, td_targets
from cf_pilot.rl_data import binary_metrics


class DeepRLTests(unittest.TestCase):
    def test_learning_discovers_hidden_evidence_on_separable_fixture(self):
        x=np.zeros((2,48),dtype=np.float32);x[:,5]=[-1,1]
        data={'x':x,'y':np.array([0,1],dtype=np.int8)}
        candidate={'name':'fixture','width':32,'dueling':True,'double':True,'false_positive_penalty':.5}
        settings={'updates':600,'validation_interval':100,'environments':8,'batch_size':32,
                  'replay_per_class':500,'warmup_updates':25,'learning_rate':.003,'target_tau':.05,'inspection_cost':.005}
        saved=learn(data,data,candidate,settings,42,binary_metrics,lambda *a,**k:None)
        network=QNetwork(32,True);network.load_state_dict(saved['weights'])
        pred,inspections,_,paths=infer(network,x,traces=True)
        np.testing.assert_array_equal(pred,[0,1])
        self.assertTrue((inspections>=1).all())
        self.assertTrue(all(0 in path for path in paths))

    def test_hidden_values_do_not_enter_observation(self):
        a = np.zeros((1, 48), dtype=np.float32)
        b = a.copy(); b[:, 5:24] = 7; b[:, 29:48] = 1
        np.testing.assert_array_equal(observations(a, [0]), observations(b, [0]))
        self.assertFalse(np.array_equal(observations(a, [1]), observations(b, [1])))

    def test_action_masks_and_finite_termination(self):
        valid = action_masks([0, 1, 7])
        self.assertEqual(valid.tolist(), [[True]*5,[False,True,True,True,True],[False,False,False,True,True]])
        net = QNetwork()
        with torch.no_grad():
            for p in net.parameters(): p.zero_()
        pred, acquired, masks, paths = infer(net, np.zeros((3,48),dtype=np.float32), traces=True)
        np.testing.assert_array_equal(acquired,[3,3,3])
        self.assertEqual(paths[0],[0,1,2,3])

    def test_double_target_selects_online_but_evaluates_target(self):
        online=torch.tensor([[999.,1.,3.,2.,0.],[1.,2.,3.,4.,5.]])
        target=torch.tensor([[999.,7.,4.,8.,9.],[9.,8.,7.,6.,5.]])
        valid=torch.tensor([[False,True,True,True,True],[False,False,False,True,True]])
        r=torch.tensor([-.005,1.]);done=torch.tensor([False,True])
        torch.testing.assert_close(td_targets(online,target,valid,r,done,True),torch.tensor([3.995,1.]))
        torch.testing.assert_close(td_targets(online,target,valid,r,done,False),torch.tensor([8.995,1.]))

    def test_replay_correction_and_circular_capacity(self):
        replay=Replay(3)
        replay.add([0,1,0,1],row=[1,2,3,4],mask=[0]*4,next_mask=[1]*4,action=[0]*4,reward=[-.005]*4,done=[False]*4)
        batch=replay.sample(4,np.random.default_rng(42),.1)
        np.testing.assert_allclose(batch['weight'],[1.8,1.8,.2,.2])
        self.assertTrue(set(batch['row'][:2]) <= {1,3})
        self.assertTrue(set(batch['row'][2:]) <= {2,4})

    def test_terminal_rewards_and_dueling_outputs(self):
        np.testing.assert_allclose(rewards([4,3,4,3,0],[1,1,0,0,1],.25),[1,-1,-.25,0,-.005])
        for dueling in (False,True):
            self.assertEqual(QNetwork(64,dueling)(torch.zeros((2,52))).shape,(2,5))
