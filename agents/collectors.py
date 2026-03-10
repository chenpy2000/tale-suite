# Trajectory collectors for VQ-VAE training.
# DiverseCollector: explores without walkthroughs by cycling through strategies (goal-directed, exploration, etc.).
# NoisyWalkthroughCollector: follows game walkthroughs when available, injects random deviations for diversity.

import argparse
import json
import os
import random
from collections import deque
from datetime import datetime, timezone
from numbers import Number

import tales
from tales.agent import register
from tales.token import get_token_counter


def _j(v):
    """JSON-serialize infos (handles nested dicts, sets, etc.)."""
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, dict):
        return {str(k): _j(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_j(x) for x in v]
    if isinstance(v, set):
        return sorted(_j(x) for x in v)
    return str(v)


def _save(path, uid, episodes, env_name, ep_idx, traj):
    """Write episodes + in-progress trajectory to disk for collect_data.py to pick up."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"agent_uid": uid, "updated_at": datetime.now(timezone.utc).isoformat(),
                   "episodes": episodes, "in_progress_episode": {"env_name": env_name, "episode_idx": ep_idx,
                   "num_steps": len(traj), "trajectory": traj}}, f, indent=2, ensure_ascii=True)


class DiverseCollector(tales.Agent):
    """Cycles through strategies (goal, explore, object, nav, random) every few steps.
    No walkthrough needed—works on any game. Keeps episodes that are long enough,
    have decent reward, and aren't too repetitive."""
    def __init__(self, **kwargs):
        self.seed = kwargs.get("seed", 20241001)
        self.rng = random.Random(self.seed)
        self.tok = get_token_counter()
        self.autosave = kwargs.get("autosave", True)
        self.path = kwargs.get("trajectory_path", os.path.join("logs", "trajectories", f"diverse_s{self.seed}.json"))
        self.min_ep = kwargs.get("min_episode_length", 15)
        self.min_r = kwargs.get("min_reward", -10)
        self.max_rep = kwargs.get("max_repetition_rate", 0.4)
        self.env_name = None
        self.ep_idx = 0
        self.episodes = []
        self.traj = []
        self.recent = deque(maxlen=5)
        self.act_cnt = {}
        self.strats = ["goal", "explore", "object", "nav", "random"]
        self.cur = self.rng.choice(self.strats)
        self.steps = 0
        self.switch = self.rng.randint(5, 12)

    @property
    def uid(self):
        return f"Diverse_s{self.seed}"

    @property
    def params(self):
        return {"agent_type": "diverse-collector", "seed": self.seed}

    def reset(self, obs, info, env_name):
        self.env_name = env_name

    def _cands(self, obs, info):
        """Build candidate actions: use admissible_commands if provided, else heuristics from obs."""
        adm = info.get("admissible_commands") if isinstance(info, dict) else None
        if adm:
            return [str(a) for a in adm]
        if adm == []:
            return ["look", "inventory", "wait"]
        a = ["look", "inventory", "examine room"]
        ol = (obs or "").lower()
        if any(w in ol for w in ["door", "exit", "room"]):
            a += ["go north", "go south", "go east", "go west"]
        for o in ["fridge", "oven", "stove", "table", "counter", "cookbook"]:
            if o in ol:
                a += [f"examine {o}", f"open {o}", f"take {o}"]
        return a

    def _pick(self, acts):
        """Pick action by current strategy. Switch strategy every few steps. Avoid repeating same action 3x."""
        self.steps += 1
        if self.steps >= self.switch:
            self.cur = self.rng.choice(self.strats)
            self.steps = 0
            self.switch = self.rng.randint(3, 8)
        if self.cur == "goal":
            kw = ["cook", "fry", "take", "insert", "put", "place", "combine"]
            cand = [a for a in acts if any(k in a.lower() for k in kw)] or acts
        elif self.cur == "explore":
            cnts = [self.act_cnt.get(a, 0) for a in acts]
            cand = [a for a, c in zip(acts, cnts) if c == min(cnts)]
        elif self.cur == "object":
            kw = ["examine", "take", "open", "insert", "drop", "put"]
            cand = [a for a in acts if any(k in a.lower() for k in kw)] or acts
        elif self.cur == "nav":
            kw = ["go", "north", "south", "east", "west", "enter", "exit"]
            cand = [a for a in acts if any(k in a.lower() for k in kw)] or acts
        else:
            cand = [a for a in acts if a not in set(self.recent)] or acts
        a = self.rng.choice(cand)
        if len(self.recent) >= 2 and all(x == a for x in list(self.recent)[-2:]):
            o = [x for x in acts if x != a]
            if o:
                a = self.rng.choice(o)
        return a

    def act(self, obs, reward, done, infos):
        infos = infos or {}
        acts = self._cands(obs, infos) or ["look", "inventory", "wait"]
        a = self._pick(acts)
        self.recent.append(a)
        self.act_cnt[a] = self.act_cnt.get(a, 0) + 1
        r = float(reward) if isinstance(reward, Number) else 0.0
        self.traj.append({"observation": obs, "action": a, "reward": r, "done": done, "infos": _j(infos)})
        if done:
            n = len(self.traj)
            total_r = sum(s["reward"] for s in self.traj)
            acts = [s["action"] for s in self.traj]
            rep = max(acts.count(x) for x in set(acts)) / len(acts) if acts else 0
            # Keep episode only if long enough, rewarding enough, and not too repetitive
            if n >= self.min_ep and total_r >= self.min_r and rep <= self.max_rep:
                self.episodes.append({"env_name": self.env_name, "episode_idx": self.ep_idx, "num_steps": n, "trajectory": self.traj})
                if self.autosave:
                    _save(self.path, self.uid, self.episodes, self.env_name, self.ep_idx + 1, [])
            self.traj = []
            self.ep_idx += 1
            self.recent.clear()
            self.act_cnt.clear()
            self.cur = self.rng.choice(self.strats)
            self.steps = 0
        return a, {"prompt": None, "response": None, "nb_tokens": self.tok(text=obs or "")}

    def episode_truncated(self, obs, info):
        """Save in-progress trajectory when episode ends due to step limit."""
        n = len(self.traj)
        total_r = sum(s["reward"] for s in self.traj)
        acts = [s["action"] for s in self.traj]
        rep = max(acts.count(x) for x in set(acts)) / len(acts) if acts else 0
        if n >= self.min_ep and total_r >= self.min_r and rep <= self.max_rep:
            self.episodes.append({"env_name": self.env_name, "episode_idx": self.ep_idx, "num_steps": n, "trajectory": self.traj})
            if self.autosave:
                _save(self.path, self.uid, self.episodes, self.env_name, self.ep_idx + 1, [])


class NoisyWalkthroughCollector(tales.Agent):
    """Follows extra.walkthrough when the game provides it. With probability noise_rate,
    takes a random admissible action instead; after a deviation, keeps random for 3 steps
    before returning to the walkthrough. Produces near-optimal but varied trajectories."""
    def __init__(self, **kwargs):
        self.seed = kwargs.get("seed", 20241001)
        self.rng = random.Random(self.seed)
        self.tok = get_token_counter()
        self.autosave = kwargs.get("autosave", True)
        self.path = kwargs.get("trajectory_path", os.path.join("logs", "trajectories", f"noisy_s{self.seed}.json"))
        self.noise = kwargs.get("noise_rate", 0.15)
        self.min_ep = kwargs.get("min_episode_length", 10)
        self.env_name = None
        self.ep_idx = 0
        self.episodes = []
        self.walk = []
        self.wi = 0
        self.traj = []
        self.deviated = False
        self.dev_steps = 0

    @property
    def uid(self):
        return f"Noisy_s{self.seed}_n{self.noise}"

    @property
    def params(self):
        return {"agent_type": "noisy-walkthrough", "seed": self.seed, "noise_rate": self.noise}

    def reset(self, obs, info, env_name):
        self.env_name = env_name
        w = info.get("extra.walkthrough") if isinstance(info, dict) else None
        self.walk = [str(a) for a in w] if w and isinstance(w, list) else []
        self.wi = 0
        self.deviated = False
        self.dev_steps = 0

    def _choose(self, infos):
        """Follow walkthrough step-by-step, or inject random action when noise triggers."""
        adm = infos.get("admissible_commands") if isinstance(infos, dict) else []
        if not adm:
            adm = ["look", "inventory", "wait"]
        if self.rng.random() < self.noise and len(adm) > 1 and not self.deviated:
            self.deviated = True
            self.dev_steps = 0
            return self.rng.choice(adm)
        if self.deviated and self.dev_steps < 3:
            self.dev_steps += 1
            return self.rng.choice(adm)
        if self.walk and self.wi < len(self.walk):
            a = self.walk[self.wi]
            self.wi += 1
            return a
        return self.rng.choice(adm) if adm else "look"

    def act(self, obs, reward, done, infos):
        infos = infos or {}
        a = self._choose(infos)
        r = float(reward) if isinstance(reward, Number) else 0.0
        self.traj.append({"observation": obs, "action": a, "reward": r, "done": done, "infos": _j(infos)})
        if done:
            if len(self.traj) >= self.min_ep:
                self.episodes.append({"env_name": self.env_name, "episode_idx": self.ep_idx, "num_steps": len(self.traj), "trajectory": self.traj})
                if self.autosave:
                    _save(self.path, self.uid, self.episodes, self.env_name, self.ep_idx + 1, [])
            self.traj = []
            self.ep_idx += 1
            self.walk = []
            self.wi = 0
            self.deviated = False
        return a, {"prompt": None, "response": None, "nb_tokens": self.tok(text=obs or "")}

    def episode_truncated(self, obs, info):
        """Save in-progress trajectory when episode ends due to step limit."""
        if len(self.traj) >= self.min_ep:
            self.episodes.append({"env_name": self.env_name, "episode_idx": self.ep_idx, "num_steps": len(self.traj), "trajectory": self.traj})
            if self.autosave:
                _save(self.path, self.uid, self.episodes, self.env_name, self.ep_idx + 1, [])


def _diverse_args(p):
    g = p.add_argument_group("diverse-collector")
    g.add_argument("--seed", type=int, default=20241001)
    g.add_argument("--trajectory-path", default=os.path.join("logs", "trajectories", "diverse_collector.json"))
    g.add_argument("--autosave", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--min-episode-length", type=int, default=15)
    g.add_argument("--min-reward", type=float, default=-10)
    g.add_argument("--max-repetition-rate", type=float, default=0.4)
    return p


def _noisy_args(p):
    g = p.add_argument_group("noisy-walkthrough")
    g.add_argument("--seed", type=int, default=20241001)
    g.add_argument("--trajectory-path", default=os.path.join("logs", "trajectories", "noisy_walkthrough.json"))
    g.add_argument("--autosave", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--noise-rate", type=float, default=0.15)
    g.add_argument("--min-episode-length", type=int, default=10)
    return p


register("diverse-collector", "Multi-strategy collector", DiverseCollector, _diverse_args)
register("noisy-walkthrough", "Walkthrough + noise", NoisyWalkthroughCollector, _noisy_args)
