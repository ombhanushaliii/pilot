import unittest

from cf_pilot.rl import (ACTIONS, EvidenceEnvironment, LOW, REVIEW, TabularAgent,
                        initial_state, rollout, terminal_reward, valid_actions)


class RLTests(unittest.TestCase):
    def test_hidden_target_and_unrevealed_values_not_in_observation(self):
        a = EvidenceEnvironment((1, 2, 3, 0, 1, 4), 0)
        b = EvidenceEnvironment((1, 2, 3, 4, 4, 0), 1)
        self.assertEqual(a.state, b.state)
        state, reward, done = a.step(0)
        self.assertEqual(state, (1, 2, 3, 0, -1, -1))
        self.assertEqual(reward, -.05)
        self.assertFalse(done)

    def test_repeated_actions_budget_and_terminal_are_masked(self):
        env = EvidenceEnvironment((1, 2, 3, 4, 4, 4), 1, budget=1)
        env.step(0)
        self.assertEqual(valid_actions(env.state, 1), [LOW, REVIEW])
        for action in (0, 1):
            with self.assertRaises(ValueError):
                env.step(action)
        self.assertEqual(env.step(REVIEW)[1:], (20., True))
        with self.assertRaises(ValueError):
            env.step(LOW)

    def test_q_learning_masks_invalid_bootstrap_and_uses_maximum(self):
        agent = TabularAgent("q_learning", alpha=.5, gamma=.9)
        state = initial_state((0, 0, 0, 1, 1, 1))
        nxt = (0, 0, 0, 1, -1, -1)
        agent.q[nxt] = [999., 2., 3., 4., 5.]
        agent.update(state, 0, -1., nxt, False)
        self.assertAlmostEqual(agent.q[state][0], .5 * (-1 + .9 * 5))

    def test_sarsa_uses_chosen_action_not_maximum(self):
        agent = TabularAgent("sarsa", alpha=.5, gamma=.9)
        state = initial_state((0, 0, 0, 1, 1, 1))
        nxt = (0, 0, 0, 1, -1, -1)
        agent.q[nxt] = [999., 2., 3., 4., 5.]
        agent.update(state, 0, -1., nxt, False, 1)
        self.assertAlmostEqual(agent.q[state][0], .5 * (-1 + .9 * 2))
        with self.assertRaises(ValueError):
            agent.update(state, 0, -1, nxt, False, 0)

    def test_terminal_updates_never_bootstrap(self):
        state = initial_state((0, 0, 0, 1, 1, 1))
        for algorithm in ("sarsa", "q_learning"):
            agent = TabularAgent(algorithm, alpha=1.)
            agent.q[state] = [1000.] * 5
            agent.update(state, REVIEW, -2., state, True)
            self.assertEqual(agent.q[state][REVIEW], -2.)

    def test_reward_table_and_inspection_cost(self):
        self.assertEqual([terminal_reward(p, y) for p, y in ((1, 1), (0, 1), (0, 0), (1, 0))], [20., -20., 1., -2.])
        with self.assertRaises(ValueError):
            terminal_reward(2, 0)

    def test_policy_roundtrip_and_no_label_needed_for_rollout(self):
        agent = TabularAgent("sarsa")
        state = initial_state((1, 2, 3, 4, 1, 2))
        agent.q[state] = [3., 0., 0., 1., 0.]
        agent.q[(1, 2, 3, 4, -1, -1)] = [999., 0., 0., 1., 2.]
        restored = TabularAgent.restore(agent.export())
        self.assertEqual(rollout((1, 2, 3, 4, 1, 2), restored), (1, 1, [ACTIONS[0], ACTIONS[REVIEW]]))
        self.assertEqual(agent.export(), restored.export())

    def test_seed_reproducibility_and_unseen_state_stops(self):
        for algorithm in ("sarsa", "q_learning"):
            a, b = [TabularAgent(algorithm, seed=42) for _ in range(2)]
            for i in range(50):
                a.episode((0, 1, 2, 3, 1, 0), i % 2, .5)
                b.episode((0, 1, 2, 3, 1, 0), i % 2, .5)
            self.assertEqual(a.export(), b.export())
        self.assertEqual(rollout((4, 4, 4, 4, 4, 4), TabularAgent("q_learning")), (0, 0, [ACTIONS[LOW]]))
