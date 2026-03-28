import argparse
import json
import os
import re
from datetime import datetime, timezone
from numbers import Number

import numpy as np

import tales
from tales.agent import register
from tales.token import get_token_counter


def _jsonable(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, set):
        return sorted(_jsonable(x) for x in v)
    return str(v)


class TrajectoryCollectorAgent(tales.Agent):
    def __init__(self, **kwargs):
        self.seed = kwargs.get("seed", 20241001)
        self.rng = np.random.RandomState(self.seed)
        self.token_counter = get_token_counter()
        self.autosave = kwargs.get("autosave", True)
        self.save_every = kwargs.get("save_every", 25)
        self.trajectory_path = kwargs.get("trajectory_path", os.path.join("logs", "trajectories", f"traj_s{self.seed}.json"))
        self.qtable_path = kwargs.get("qtable_path")
        self.alpha = kwargs.get("alpha", 0.2)
        self.gamma = kwargs.get("gamma", 0.95)
        self.epsilon = kwargs.get("epsilon", 0.3)
        self.epsilon_min = kwargs.get("epsilon_min", 0.05)
        self.epsilon_decay = kwargs.get("epsilon_decay", 0.999)
        self.actions = ["north", "south", "east", "west", "up", "down", "look", "inventory", "take", "drop", "open", "close", "help", "wait"]
        self.env_name = None
        self.ep_idx = 0
        self.episodes = []
        self.traj = []
        self.q = {}
        self.prev_s, self.prev_a, self.prev_score = None, None, None
        self.last_adm = []

    @property
    def uid(self):
        return f"TrajCollector_s{self.seed}_a{self.alpha}_e{self.epsilon}"

    @property
    def params(self):
        return {"agent_type": "trajectory-collector", "seed": self.seed, "trajectory_path": self.trajectory_path,
                "alpha": self.alpha, "gamma": self.gamma, "epsilon": self.epsilon}

    def reset(self, obs, info, env_name):
        self.env_name = env_name

    def _state(self, obs):
        return re.sub(r"\s+", " ", (obs or "").lower().strip())[:512]

    def _candidates(self, obs, info):
        adm = info.get("admissible_commands") if isinstance(info, dict) else None
        if adm:
            self.last_adm = [str(a) for a in adm]
            return self.last_adm
        if adm == []:
            return self.last_adm or ["look", "inventory", "wait"]
        cand = list(self.actions)
        for n in list(dict.fromkeys(re.findall(r"\b[a-zA-Z]{4,}\b", obs or "")))[:5]:
            cand += [f"take {n}", f"open {n}", f"examine {n}"]
        return cand

    def _q(self, s, a):
        return self.q.get(s, {}).get(a, 0.0)

    def _best(self, s, acts):
        vs = [self._q(s, a) for a in acts]
        m = max(vs)
        return acts[self.rng.choice([i for i, v in enumerate(vs) if v == m])]

    def _update_q(self, s, a, r, s2, acts2, done):
        next_max = max(self._q(s2, a2) for a2 in acts2) if not done and acts2 else 0.0
        td = r + self.gamma * next_max
        self.q.setdefault(s, {})[a] = self._q(s, a) + self.alpha * (td - self._q(s, a))

    def act(self, obs, reward, done, infos):
        infos = infos or {}
        s = self._state(obs)
        acts = self._candidates(obs, infos)
        score = float(reward) if isinstance(reward, Number) else 0.0

        if self.prev_s is not None and self.prev_a is not None:
            r = score - (self.prev_score if self.prev_score is not None else score)
            if done and infos.get("won"):
                r += 5.0
            elif done and infos.get("lost"):
                r -= 2.0
            self._update_q(self.prev_s, self.prev_a, r, s, acts, done)

        if done:
            self.episodes.append({"env_name": self.env_name, "episode_idx": self.ep_idx, "num_steps": len(self.traj), "trajectory": self.traj})
            self.traj = []
            self.prev_s = self.prev_a = self.prev_score = None
            self.last_adm = []
            self.ep_idx += 1
            if self.autosave:
                self._save()

        a = self._best(s, acts) if acts and self.rng.rand() >= self.epsilon else (self.rng.choice(acts) if acts else "help")
        self.prev_s, self.prev_a, self.prev_score = s, a, score
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.traj.append({"observation": obs, "action": a, "reward": reward, "score": score, "done": done, "infos": _jsonable(infos)})

        if self.autosave and len(self.traj) % max(1, self.save_every) == 0:
            self._save()
        return a, {"prompt": None, "response": None, "nb_tokens": self.token_counter(text=obs)}

    def _save(self):
        d = os.path.dirname(self.trajectory_path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(self.trajectory_path, "w", encoding="utf-8") as f:
            json.dump({"agent_uid": self.uid, "updated_at": datetime.now(timezone.utc).isoformat(),
                       "episodes": self.episodes, "in_progress_episode": {"env_name": self.env_name, "episode_idx": self.ep_idx,
                       "num_steps": len(self.traj), "trajectory": self.traj}}, f, indent=2, ensure_ascii=True)
        if self.qtable_path:
            d = os.path.dirname(self.qtable_path)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(self.qtable_path, "w", encoding="utf-8") as f:
                json.dump(self.q, f, indent=2, ensure_ascii=True)


def build_argparser(parser=None):
    p = parser or argparse.ArgumentParser()
    g = p.add_argument_group("trajectory-collector")
    g.add_argument("--seed", type=int, default=20241001)
    g.add_argument("--trajectory-path", default=os.path.join("logs", "trajectories", "trajectory_collector.json"))
    g.add_argument("--autosave", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--save-every", type=int, default=25)
    g.add_argument("--qtable-path", default=None)
    g.add_argument("--alpha", type=float, default=0.2)
    g.add_argument("--gamma", type=float, default=0.95)
    g.add_argument("--epsilon", type=float, default=0.3)
    g.add_argument("--epsilon-min", type=float, default=0.05)
    g.add_argument("--epsilon-decay", type=float, default=0.999)
    return p


register("trajectory-collector", "Q-learning collector", TrajectoryCollectorAgent, build_argparser)
