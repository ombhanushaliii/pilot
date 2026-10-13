"""Matched RL ablations with fixed dimensions, history, visibility and cost controls."""
import copy
import numpy as np
import torch
from torch import nn

from .benchmark import FEATURE_COLUMNS, FEATURE_GROUPS
from .context import CONTEXT_COLUMNS
from .deep_rl import Replay, clone_state, td_targets
from .rl_data import binary_metrics

NFEATURES = len(CONTEXT_COLUMNS)
OBSERVATION_DIM = 2 * NFEATURES + 4
BITS = np.array([[(mask >> i) & 1 for i in range(3)] for mask in range(8)], dtype=np.float32)
VISIBILITY = np.zeros((8, 2 * NFEATURES), dtype=np.float32)
for mask in range(8):
    # Current-evidence deviations must be masked with that evidence, even though
    # their historical reference statistics are available at the initial state.
    derived = {"pacing": [CONTEXT_COLUMNS.index("gap_history_log_deviation")],
               "trajectory": [CONTEXT_COLUMNS.index("retry_history_deviation")],
               "language": []}
    hidden_derived = set(derived["pacing"] + derived["trajectory"])
    indices = [i for i in range(24, NFEATURES) if i not in hidden_derived]
    indices += [FEATURE_COLUMNS.index(n) for n in FEATURE_GROUPS["basic"]]
    for i, group in enumerate(("pacing", "trajectory", "language")):
        if mask & (1 << i):
            indices += [FEATURE_COLUMNS.index(n) for n in FEATURE_GROUPS[group]] + derived[group]
    VISIBILITY[mask, indices] = 1
    VISIBILITY[mask, [i + NFEATURES for i in indices]] = 1


def initial_mask(candidate):
    return 7 if candidate["full_visible"] else 0


def observations(x, masks):
    masks = np.asarray(masks, dtype=np.int8)
    if x.shape[1] != 2 * NFEATURES or np.any((masks < 0) | (masks > 7)):
        raise ValueError("Invalid ablation observation")
    values = np.clip(x, -8, 8) * VISIBILITY[masks]
    remaining = (3 - BITS[masks].sum(axis=1))[:, None] / 3
    return np.concatenate([values, BITS[masks], remaining], axis=1).astype(np.float32)


def valid_actions(masks):
    return np.concatenate([BITS[np.asarray(masks)] == 0, np.ones((len(masks), 2), dtype=bool)], axis=1)


class Network(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(OBSERVATION_DIM, width), nn.ReLU(),
                                    nn.Linear(width, width), nn.ReLU())
        self.value = nn.Linear(width, 1)
        self.advantage = nn.Linear(width, 5)

    def forward(self, x):
        hidden = self.shared(x)
        advantages = self.advantage(hidden)
        return self.value(hidden) + advantages - advantages.mean(dim=1, keepdim=True)


@torch.no_grad()
def infer(network, x, candidate, batch_size=4096):
    network.eval()
    prediction = np.zeros(len(x), dtype=np.int8)
    inspected = np.zeros(len(x), dtype=np.int8)
    for first in range(0, len(x), batch_size):
        indices = np.arange(first, min(first + batch_size, len(x)))
        masks = np.full(len(indices), initial_mask(candidate), dtype=np.int8)
        for step in range(4):
            if not len(indices):
                break
            inputs = torch.from_numpy(observations(x[indices], masks))
            valid = torch.from_numpy(valid_actions(masks))
            action = network(inputs).masked_fill(~valid, -torch.inf).argmax(dim=1).numpy()
            terminal = action >= 3
            prediction[indices[terminal]] = action[terminal] == 4
            inspected[indices[terminal]] = step
            masks = (masks[~terminal] | (1 << action[~terminal])).astype(np.int8)
            indices = indices[~terminal]
        if len(indices):
            raise ValueError("Policy did not terminate")
    return prediction, inspected


def measure(data, prediction, count):
    pooled = binary_metrics(data["y"], prediction, count)
    by_contest = {str(cid): binary_metrics(data["y"][data["cid"] == cid],
                  prediction[data["cid"] == cid], count[data["cid"] == cid])
                  for cid in sorted(set(data["cid"]))}
    return {"pooled": pooled, "per_contest": by_contest,
            "macro_f1": float(np.mean([v["f1"] for v in by_contest.values()])),
            "worst_contest_f1": min(v["f1"] for v in by_contest.values())}


def rank(metrics, pooled=False):
    if pooled:
        return metrics["pooled"]["f1"], metrics["pooled"]["precision"]
    return metrics["macro_f1"], metrics["worst_contest_f1"], metrics["pooled"]["f1"]


def learn(train, validation, candidate, settings, seed, progress=print):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    online = Network(settings["width"])
    target = copy.deepcopy(online).eval()
    opt = torch.optim.AdamW(online.parameters(), lr=settings["learning_rate"], weight_decay=1e-4)
    replay = Replay(settings["replay_per_class"])
    n = settings["environments"]
    rows = rng.integers(0, len(train["y"]), size=n)
    masks = np.full(n, initial_mask(candidate), dtype=np.int8)
    prior = float(train["y"].mean())
    best, pooled_best, curve = None, None, []
    interactions, finished, accumulated = 0, 0, 0.
    td_loss = 0.
    for update in range(1, settings["updates"] + 1):
        eps = max(.05, 1 - .95 * update / (settings["updates"] * .8))
        valid = valid_actions(masks)
        with torch.no_grad():
            action = online(torch.from_numpy(observations(train["x"][rows], masks))).masked_fill(
                     ~torch.from_numpy(valid), -torch.inf).argmax(dim=1).numpy()
        random = np.where(valid, rng.random(valid.shape), -1).argmax(axis=1)
        explore = rng.random(n) < eps
        action[explore] = random[explore]
        ended = action >= 3
        following = np.where(ended, masks, masks | (1 << np.minimum(action, 2))).astype(np.int8)
        y = train["y"][rows]
        reward = np.where(action < 3, -candidate["cost"],
                 np.where(action == 4, np.where(y == 1, 1., -settings["false_positive_penalty"]),
                          np.where(y == 1, -1., 0.))).astype(np.float32)
        replay.add(y, row=rows, mask=masks, next_mask=following, action=action, reward=reward, done=ended)
        accumulated += float(reward.sum())
        interactions += n
        finished += int(ended.sum())
        rows[ended] = rng.integers(0, len(train["y"]), size=int(ended.sum()))
        masks = np.where(ended, initial_mask(candidate), following).astype(np.int8)
        if update >= settings["warmup_updates"] and all(replay.count):
            batch = replay.sample(settings["batch_size"], rng, prior)
            current = torch.from_numpy(observations(train["x"][batch["row"]], batch["mask"]))
            after = torch.from_numpy(observations(train["x"][batch["row"]], batch["next_mask"]))
            with torch.no_grad():
                objective = td_targets(online(after), target(after), torch.from_numpy(valid_actions(batch["next_mask"])),
                                       torch.from_numpy(batch["reward"]), torch.from_numpy(batch["done"]), True)
            estimate = online(current).gather(1, torch.from_numpy(batch["action"].astype(np.int64))[:, None]).squeeze(1)
            loss = (nn.functional.smooth_l1_loss(estimate, objective, reduction="none") *
                    torch.from_numpy(batch["weight"])).mean()
            if not torch.isfinite(loss):
                raise ValueError("Non-finite TD loss")
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), 5)
            opt.step()
            td_loss = float(loss.detach())
            with torch.no_grad():
                for older, newer in zip(target.parameters(), online.parameters()):
                    older.lerp_(newer, settings["target_tau"])
        if update % settings["validation_interval"] == 0 or update == settings["updates"]:
            pred, count = infer(online, validation["x"], candidate)
            metrics = measure(validation, pred, count)
            point = {"update": update, "epsilon": eps, "validation": metrics,
                     "training_reward_per_transition": accumulated / interactions, "last_td_loss": td_loss}
            curve.append(point)
            if best is None or rank(metrics) > rank(best["validation"]):
                best = {"weights": clone_state(online), "validation": metrics, "update": update}
            if pooled_best is None or rank(metrics, True) > rank(pooled_best["validation"], True):
                pooled_best = {"weights": clone_state(online), "validation": metrics, "update": update}
            progress(f"{candidate['name']} seed {seed}, update {update:,}: macro F1 {metrics['macro_f1']:.3f}, pooled {metrics['pooled']['f1']:.3f}", flush=True)
    return {"macro": best, "pooled": pooled_best, "history": curve,
            "interactions": interactions, "finished_episodes": finished}
