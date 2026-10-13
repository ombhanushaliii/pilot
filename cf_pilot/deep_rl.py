"""Continuous-feature DQN / dueling Double DQN for hidden-outcome inspection."""
import copy
import numpy as np
import torch
from torch import nn

from .benchmark import FEATURE_COLUMNS, FEATURE_GROUPS


GROUPS = tuple(FEATURE_GROUPS)
VISIBILITY = np.zeros((8, 2 * len(FEATURE_COLUMNS)), dtype=np.float32)
BITS = np.array([[(mask >> i) & 1 for i in range(3)] for mask in range(8)], dtype=np.float32)
for mask in range(8):
    for group in (0,) + tuple(i + 1 for i in range(3) if (mask >> i) & 1):
        for name in FEATURE_GROUPS[GROUPS[group]]:
            idx = FEATURE_COLUMNS.index(name)
            VISIBILITY[mask, [idx, idx + len(FEATURE_COLUMNS)]] = 1.
OBSERVATION_DIM = 2 * len(FEATURE_COLUMNS) + 4


def observations(x, masks):
    masks = np.asarray(masks, dtype=np.int8)
    if x.shape[1] != 48 or np.any((masks < 0) | (masks > 7)):
        raise ValueError("Invalid continuous evidence state")
    visible = np.clip(x, -8., 8.) * VISIBILITY[masks]
    remaining = (3 - BITS[masks].sum(axis=1))[:, None] / 3
    return np.concatenate((visible, BITS[masks], remaining), axis=1).astype(np.float32)


def action_masks(masks):
    bits = BITS[np.asarray(masks, dtype=np.int8)]
    return np.concatenate((bits == 0, np.ones((len(bits), 2), dtype=bool)), axis=1)


def rewards(actions, targets, penalty, cost=.005):
    actions, targets = np.asarray(actions), np.asarray(targets)
    return np.where(actions < 3, -cost,
        np.where(actions == 4, np.where(targets == 1, 1., -penalty),
                 np.where(targets == 1, -1., 0.))).astype(np.float32)


class QNetwork(nn.Module):
    def __init__(self, width=64, dueling=False):
        super().__init__()
        self.dueling = dueling
        self.shared = nn.Sequential(nn.Linear(OBSERVATION_DIM, width), nn.ReLU(),
                                    nn.Linear(width, width), nn.ReLU())
        self.advantage = nn.Linear(width, 5)
        if dueling:
            self.value = nn.Linear(width, 1)

    def forward(self, x):
        hidden = self.shared(x)
        advantage = self.advantage(hidden)
        if not self.dueling:
            return advantage
        return self.value(hidden) + advantage - advantage.mean(dim=1, keepdim=True)


def td_targets(online_next, target_next, valid_next, r, done, double=True, gamma=1.):
    """Terminal transitions do not bootstrap; invalid inspection actions never win."""
    if double:
        chosen = online_next.masked_fill(~valid_next, -torch.inf).argmax(dim=1)
        continuation = target_next.gather(1, chosen[:, None]).squeeze(1)
    else:
        continuation = target_next.masked_fill(~valid_next, -torch.inf).max(dim=1).values
    return r + gamma * torch.where(done, torch.zeros_like(continuation), continuation)


class Replay:
    """Two compact transition rings; balanced replay uses natural-frequency correction."""
    FIELDS = {"row": np.int32, "mask": np.int8, "next_mask": np.int8,
              "action": np.int8, "reward": np.float32, "done": bool}

    def __init__(self, capacity=50000):
        self.capacity, self.position, self.count = capacity, [0, 0], [0, 0]
        self.data = [{key: np.empty(capacity, dtype=dtype) for key, dtype in self.FIELDS.items()} for _ in range(2)]

    def add(self, targets, **fields):
        for cls in (0, 1):
            selected = np.flatnonzero(np.asarray(targets) == cls)
            if not len(selected):
                continue
            at = (self.position[cls] + np.arange(len(selected))) % self.capacity
            for key in self.FIELDS:
                self.data[cls][key][at] = np.asarray(fields[key])[selected]
            self.position[cls] = int((self.position[cls] + len(selected)) % self.capacity)
            self.count[cls] = min(self.capacity, self.count[cls] + len(selected))

    def sample(self, size, rng, positive_prior):
        if not all(self.count) or size % 2:
            raise ValueError("Replay needs both outcomes and an even batch size")
        sampled = [rng.integers(0, n, size=size // 2) for n in self.count]
        result = {key: np.concatenate([self.data[c][key][sampled[c]] for c in (0, 1)]) for key in self.FIELDS}
        result["weight"] = np.r_[np.full(size // 2, 2 * (1 - positive_prior)),
                                np.full(size // 2, 2 * positive_prior)].astype(np.float32)
        return result


@torch.no_grad()
def infer(network, x, batch_size=4096, traces=False):
    network.eval()
    n = len(x)
    decisions, counts = np.zeros(n, dtype=np.int8), np.zeros(n, dtype=np.int8)
    mask_out = np.zeros(n, dtype=np.int8)
    path_out = [[] for _ in range(n)] if traces else None
    for start in range(0, n, batch_size):
        stop = min(n, start + batch_size)
        indices = np.arange(start, stop)
        masks = np.zeros(len(indices), dtype=np.int8)
        for step in range(4):
            if not len(indices):
                break
            obs = torch.from_numpy(observations(x[indices], masks))
            valid = torch.from_numpy(action_masks(masks))
            actions = network(obs).masked_fill(~valid, -torch.inf).argmax(dim=1).numpy()
            if traces:
                for i, action in zip(indices, actions):
                    path_out[int(i)].append(int(action))
            terminal = actions >= 3
            ended = indices[terminal]
            decisions[ended] = (actions[terminal] == 4).astype(np.int8)
            counts[ended] = step
            mask_out[ended] = masks[terminal]
            alive = ~terminal
            masks = (masks[alive] | (1 << actions[alive])).astype(np.int8)
            indices = indices[alive]
        if len(indices):
            raise ValueError("Deep policy failed to terminate within its action budget")
    return decisions, counts, mask_out, path_out


def clone_state(network):
    return {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}


def learn(train, val, candidate, settings, seed, score, progress=print):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    online = QNetwork(candidate["width"], candidate["dueling"])
    target = copy.deepcopy(online).eval()
    optimizer = torch.optim.AdamW(online.parameters(), lr=settings["learning_rate"], weight_decay=1e-4)
    replay = Replay(settings["replay_per_class"])
    nenv = settings["environments"]
    rows, masks = rng.integers(0, len(train["y"]), size=nenv), np.zeros(nenv, dtype=np.int8)
    positive_prior = float(train["y"].mean())
    history, best, total_reward, finished, interactions = [], None, 0., 0, 0
    for update in range(1, settings["updates"] + 1):
        epsilon = max(.05, 1 - .95 * update / (settings["updates"] * .8))
        obs = torch.from_numpy(observations(train["x"][rows], masks))
        valid = action_masks(masks)
        with torch.no_grad():
            actions = online(obs).masked_fill(~torch.from_numpy(valid), -torch.inf).argmax(dim=1).numpy()
        exploratory = rng.random(nenv) < epsilon
        random_actions = np.where(valid, rng.random(valid.shape), -1).argmax(axis=1)
        actions[exploratory] = random_actions[exploratory]
        terminal = actions >= 3
        next_masks = np.where(terminal, masks, masks | (1 << np.minimum(actions, 2))).astype(np.int8)
        r = rewards(actions, train["y"][rows], candidate["false_positive_penalty"], settings["inspection_cost"])
        replay.add(train["y"][rows], row=rows, mask=masks, next_mask=next_masks, action=actions, reward=r, done=terminal)
        total_reward += float(r.sum())
        interactions += nenv
        finished += int(terminal.sum())
        rows[terminal] = rng.integers(0, len(train["y"]), size=int(terminal.sum()))
        masks = np.where(terminal, 0, next_masks).astype(np.int8)
        if update >= settings["warmup_updates"] and all(replay.count):
            batch = replay.sample(settings["batch_size"], rng, positive_prior)
            current = torch.from_numpy(observations(train["x"][batch["row"]], batch["mask"]))
            following = torch.from_numpy(observations(train["x"][batch["row"]], batch["next_mask"]))
            with torch.no_grad():
                targets = td_targets(online(following), target(following),
                    torch.from_numpy(action_masks(batch["next_mask"])), torch.from_numpy(batch["reward"]),
                    torch.from_numpy(batch["done"]), candidate["double"])
            expected = online(current).gather(1, torch.from_numpy(batch["action"].astype(np.int64))[:, None]).squeeze(1)
            loss = (torch.nn.functional.smooth_l1_loss(expected, targets, reduction="none") * torch.from_numpy(batch["weight"])).mean()
            if not torch.isfinite(loss):
                raise ValueError("Non-finite deep RL loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(online.parameters(), 5.)
            optimizer.step()
            with torch.no_grad():
                for old, new in zip(target.parameters(), online.parameters()):
                    old.lerp_(new, settings["target_tau"])
        if update % settings["validation_interval"] == 0 or update == settings["updates"]:
            pred, acquired, _, _ = infer(online, val["x"])
            measurement = score(val["y"], pred, acquired)
            point = {"updates": update, "interactions": interactions, "finished_episodes": finished,
                     "epsilon": epsilon, "validation": measurement,
                     "training_reward_per_interaction": total_reward / interactions}
            history.append(point)
            if best is None or (measurement["f1"], measurement["precision"]) > (best["validation"]["f1"], best["validation"]["precision"]):
                best = {"weights": clone_state(online), "validation": measurement, "selected_update": update}
            progress(f"{candidate['name']} seed {seed}: {update:,} updates, validation F1 {measurement['f1']:.3f}, P {measurement['precision']:.3f}, R {measurement['recall']:.3f}", flush=True)
    best.update(history=history, interactions=interactions, finished_episodes=finished,
                seed=seed, candidate=candidate)
    return best
