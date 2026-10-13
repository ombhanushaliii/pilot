"""Tabular evidence-inspection environment, Q-learning and on-policy SARSA."""

import random


STATE_FEATURES = ("attempts", "attempted_problems", "last_submission_fraction",
                  "gap_median_fraction", "retry_fraction", "distinct_languages")
ACTIONS = ("inspect_pacing", "inspect_trajectory", "inspect_language", "low_priority", "review")
LOW, REVIEW = 3, 4
REWARDS = {"true_positive": 20., "false_negative": -20.,
           "true_negative": 1., "false_positive": -2.}


def terminal_reward(decision, target, rewards=REWARDS):
    if decision not in (0, 1) or target not in (0, 1):
        raise ValueError("Binary decision and recorded-outcome target required")
    name = ("true_positive" if decision else "false_negative") if target else (
        "false_positive" if decision else "true_negative")
    return rewards[name]


def initial_state(values):
    if len(values) != len(STATE_FEATURES):
        raise ValueError("Unexpected observation dimension")
    return tuple(values[:3]) + (-1, -1, -1)


def valid_actions(state, budget=3):
    revealed = sum(x != -1 for x in state[3:])
    return ([a for a in range(3) if state[a + 3] == -1] if revealed < budget else []) + [LOW, REVIEW]


class EvidenceEnvironment:
    """Hidden target supplies reward only. A missing value (bin 4) is not unrevealed (-1)."""

    def __init__(self, values, target, *, budget=3, cost=.05, rewards=None):
        if type(budget) is not int or not 0 <= budget <= 3 or cost < 0 or target not in (0, 1):
            raise ValueError("Invalid environment configuration")
        self._values, self._target = tuple(values), target
        self.budget, self.cost, self.rewards = budget, cost, rewards or REWARDS
        self.state, self.done, self.inspections = initial_state(values), False, 0

    def step(self, action):
        if self.done or action not in valid_actions(self.state, self.budget):
            raise ValueError("Terminal, repeated or otherwise invalid action")
        if action < 3:
            updated = list(self.state)
            updated[action + 3] = self._values[action + 3]
            self.state = tuple(updated)
            self.inspections += 1
            return self.state, -self.cost, False
        self.done = True
        return self.state, terminal_reward(int(action == REVIEW), self._target, self.rewards), True


class TabularAgent:
    def __init__(self, algorithm, *, alpha=.05, gamma=1., seed=0, budget=3):
        if algorithm not in {"q_learning", "sarsa"} or not 0 < alpha <= 1 or not 0 <= gamma <= 1:
            raise ValueError("Invalid tabular learner")
        self.algorithm, self.alpha, self.gamma, self.budget = algorithm, alpha, gamma, budget
        self.rng = random.Random(seed)
        self.q = {}

    def values(self, state):
        return self.q.get(state, [0.] * len(ACTIONS))

    def choose(self, state, epsilon=0.):
        allowed = valid_actions(state, self.budget)
        if epsilon and self.rng.random() < epsilon:
            return self.rng.choice(allowed)
        values = self.values(state)
        best = max(values[a] for a in allowed)
        tied = [a for a in allowed if values[a] == best]
        # Training explores ties; inference is deterministic and conservatively stops on an unseen state.
        if epsilon:
            return self.rng.choice(tied)
        return next(a for a in (LOW, REVIEW, 0, 1, 2) if a in tied)

    def update(self, state, action, reward, next_state, done, next_action=None):
        if action not in valid_actions(state, self.budget):
            raise ValueError("Invalid action in learning update")
        target = reward
        if not done:
            allowed = valid_actions(next_state, self.budget)
            if self.algorithm == "sarsa":
                if next_action not in allowed:
                    raise ValueError("SARSA requires the actual valid next policy action")
                continuation = self.values(next_state)[next_action]
            else:
                continuation = max(self.values(next_state)[a] for a in allowed)
            target += self.gamma * continuation
        values = self.q.setdefault(state, [0.] * len(ACTIONS))
        values[action] += self.alpha * (target - values[action])

    def episode(self, values, target, epsilon, *, cost=.05, rewards=None):
        env = EvidenceEnvironment(values, target, budget=self.budget, cost=cost, rewards=rewards)
        state, total = env.state, 0.
        action = self.choose(state, epsilon)
        while True:
            next_state, reward, done = env.step(action)
            total += reward
            next_action = self.choose(next_state, epsilon) if not done and self.algorithm == "sarsa" else None
            self.update(state, action, reward, next_state, done, next_action)
            if done:
                return total
            state = next_state
            action = next_action if self.algorithm == "sarsa" else self.choose(state, epsilon)

    def export(self):
        return {"algorithm": self.algorithm, "alpha": self.alpha, "gamma": self.gamma,
                "budget": self.budget, "q": {",".join(map(str, k)): v[:] for k, v in sorted(self.q.items())}}

    @classmethod
    def restore(cls, model):
        obj = cls(model["algorithm"], alpha=model["alpha"], gamma=model["gamma"], budget=model["budget"])
        obj.q = {tuple(map(int, key.split(","))): values[:] for key, values in model["q"].items()}
        return obj


def rollout(values, agent):
    """Inference accepts no target. Hidden summaries enter the state only after inspection."""
    state, actions = initial_state(values), []
    for _ in range(agent.budget + 1):
        action = agent.choose(state)
        actions.append(ACTIONS[action])
        if action in (LOW, REVIEW):
            return int(action == REVIEW), len(actions) - 1, actions
        updated = list(state)
        updated[action + 3] = values[action + 3]
        state = tuple(updated)
    raise ValueError("Policy did not terminate within its evidence budget")
