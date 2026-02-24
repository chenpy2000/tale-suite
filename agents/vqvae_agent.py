import argparse
import random
import sys
from pathlib import Path

import torch

import tales
from tales.agent import register
from tales.token import get_token_counter

_ROOT = Path(__file__).resolve().parent.parent
_LA = _ROOT / "latent-action"
if str(_LA) not in sys.path:
    sys.path.insert(0, str(_LA))
from vqvae import VQVAE, ActionVocab


class VQVAEAgent(tales.Agent):
    def __init__(self, **kwargs):
        self.ckpt = kwargs.get("checkpoint_path")
        self.w = kwargs.get("window_size")
        self.seed = kwargs.get("seed", 20241001)
        self.token_counter = get_token_counter()
        if not self.ckpt:
            raise ValueError("need --checkpoint-path")
        path = Path(self.ckpt)
        if not path.is_absolute():
            path = _ROOT / path
        if not path.exists():
            raise FileNotFoundError(path)
        ckpt = torch.load(path, map_location="cpu")
        a = ckpt.get("args", {})
        self.w = self.w or int(a.get("window_size", 5))
        trajectories_dir = str(_LA / "data" / "trajectories_cleaned")
        self.model = VQVAE(text_model_name=a.get("text_model", "all-MiniLM-L6-v2"), action_vocab=ActionVocab(ckpt["action_vocab"]),
                          trajectories_root=trajectories_dir, latent_dim=a.get("latent_dim", 256),
                          num_codes=a.get("num_codes", 128), commitment_beta=a.get("commitment_beta", 0.25))
        self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.model.eval()
        self.rng = random.Random(self.seed)
        self.obs_h, self.act_h = [], []

    @property
    def uid(self):
        return f"VQVAE_{Path(self.ckpt).stem}_w{self.w}_s{self.seed}"

    @property
    def params(self):
        return {"agent_type": "vqvae", "checkpoint_path": self.ckpt, "window_size": self.w, "seed": self.seed}

    def reset(self, obs, info, env_name):
        self.obs_h, self.act_h = [], []

    def act(self, obs, reward, done, info):
        adm = info.get("admissible_commands") if isinstance(info, dict) else None
        self.obs_h.append(obs or "")
        obs_seq = self.obs_h[-self.w:]
        act_seq = self.act_h[-(self.w - 1):] + [ActionVocab.PAD_TOKEN] * (self.w - len(self.act_h) + 1)
        while len(obs_seq) < self.w:
            obs_seq = [""] + obs_seq
        while len(act_seq) < self.w:
            act_seq = [ActionVocab.PAD_TOKEN] + act_seq

        with torch.no_grad():
            obs_b, act_b = self.model._normalize_batch([obs_seq], [act_seq])
            obs_emb, act_emb = self.model._encode_text_batch(obs_b, act_b)
            z = self.model.encoder(obs_emb, act_emb)
            q, _, _ = self.model.quantizer(z)
            logits = self.model.decoder(q, obs_emb)
            pred_id = logits[0, -1, :].argmax().item()
        a = self.model.action_vocab.decode(pred_id)
        if a in (ActionVocab.PAD_TOKEN, ActionVocab.UNK_TOKEN):
            a = None
        if adm is not None:
            adm = [str(x) for x in adm]
            if adm and (not a or a not in adm):
                a = self.rng.choice(adm)
            elif not adm:
                a = a or "wait"
        if a is None:
            a = "look"
        self.act_h.append(a)
        if len(self.act_h) > self.w * 2:
            self.obs_h = self.obs_h[-self.w:]
            self.act_h = self.act_h[-self.w:]
        return str(a), {"prompt": None, "response": None, "nb_tokens": self.token_counter(text=obs or "")}


def build_argparser(parser=None):
    p = parser or argparse.ArgumentParser()
    g = p.add_argument_group("vqvae")
    g.add_argument("--checkpoint-path", default="latent-action/checkpoints/vqvae_checkpoint.pt")
    g.add_argument("--window-size", type=int, default=None)
    g.add_argument("--seed", type=int, default=20241001)
    return p


register("vqvae", "VQ-VAE agent", VQVAEAgent, build_argparser)
