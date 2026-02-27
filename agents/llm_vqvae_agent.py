# LLM + VQ-VAE agent. LLM reasons, VQ-VAE scores admissible actions.

import argparse
import random
import sys
from pathlib import Path

import requests
import torch

import tales
from tales.agent import register
from tales.token import get_token_counter

_ROOT = Path(__file__).resolve().parent.parent
_LA = _ROOT / "latent-action"
if str(_LA) not in sys.path:
    sys.path.insert(0, str(_LA))
from vqvae import ActionVocab, VQVAE

SYSTEM_PROMPT = (
    "You are playing a text-based game. Given the observation and admissible commands, "
    "reason briefly about what to do, then output a *single* action (exactly one command from the list). "
    "Output only the action, nothing else."
)


class LLMVQVAEAgent(tales.Agent):
    def __init__(self, api_key, api_url, model, vqvae_checkpoint, window_size=None, llm_weight=0.3, seed=20241001, **kwargs):
        if not api_key:
            raise ValueError("--api-key is required for llm-vqvae agent")
        self.api_key = api_key
        self.api_url = api_url.rstrip("/") + "/v1/chat/completions"
        self.model_name = model
        self.llm_weight = llm_weight
        self.seed = seed
        self.rng = random.Random(seed)
        self.token_counter = get_token_counter()

        path = Path(vqvae_checkpoint)
        if not path.is_absolute():
            path = _ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"VQ-VAE checkpoint not found: {path}")

        ckpt = torch.load(path, map_location="cpu")
        args = ckpt.get("args", {})
        traj_dir = str(_LA / "data" / "trajectories_cleaned")
        if not Path(traj_dir).exists():
            traj_dir = str(_LA / "data" / "trajectories")
        self.vqvae = VQVAE(
            text_model_name=args.get("text_model", "all-MiniLM-L6-v2"),
            action_vocab=ActionVocab(ckpt["action_vocab"]),
            trajectories_root=traj_dir,
            latent_dim=args.get("latent_dim", 256),
            num_codes=args.get("num_codes", 128),
            commitment_beta=args.get("commitment_beta", 0.25),
        )
        sd = ckpt["model_state_dict"]
        remap = {
            "encoder.input_proj": "encoder.proj",
            "encoder.positional_encoding.pe": "encoder.pe",
            "encoder.output_proj": "encoder.out",
            "decoder.option_proj": "decoder.opt_proj",
            "decoder.decoder": "decoder.lstm",
            "decoder.output_head": "decoder.head",
        }
        for old, new in remap.items():
            for k in list(sd.keys()):
                if k.startswith(old + "."):
                    sd[new + k[len(old):]] = sd.pop(k)
        self.vqvae.load_state_dict(sd, strict=False)
        self.vqvae.eval()
        self.vqvae.to("cuda" if torch.cuda.is_available() else "cpu")

        self.w = window_size or args.get("window_size", 5)
        self.obs_h, self.act_h = [], []
        self.history = []

    @property
    def uid(self):
        return f"LLMVQVAE_{self.model_name}_w{self.llm_weight}_s{self.seed}"

    @property
    def params(self):
        return {
            "agent_type": "llm_vqvae",
            "model": self.model_name,
            "llm_weight": self.llm_weight,
            "seed": self.seed,
        }

    def reset(self, obs, info, env_name):
        self.obs_h, self.act_h = [], []
        self.history = []

    def _llm_call(self, messages):
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        payload = {"model": self.model_name, "messages": messages, "max_tokens": 150, "temperature": 0.0}
        r = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()

    def act(self, obs, reward, done, info):
        if done:
            self.reset(obs, info, None)
        adm = info.get("admissible_commands") if isinstance(info, dict) else None
        adm = [str(a) for a in adm] if adm else []

        self.obs_h.append(obs or "")
        obs_seq = self.obs_h[-self.w:]
        act_seq = self.act_h[-(self.w - 1):] + [ActionVocab.PAD_TOKEN] * (self.w - len(self.act_h) + 1)
        while len(obs_seq) < self.w:
            obs_seq = [""] + obs_seq
        while len(act_seq) < self.w:
            act_seq = [ActionVocab.PAD_TOKEN] + act_seq

        llm_action = None
        prompt = f"{obs}\n\nAdmissible commands: {', '.join(adm)}\n\nOutput one action:" if adm else obs
        if adm:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for o, a in self.history[-5:]:
                messages.append({"role": "user", "content": o})
                messages.append({"role": "assistant", "content": a})
            messages.append({"role": "user", "content": prompt})
            try:
                llm_action = self._llm_call(messages)
                for cmd in adm:
                    if cmd.lower() in llm_action.lower() or llm_action.lower() in cmd.lower():
                        llm_action = cmd
                        break
                if llm_action not in adm:
                    llm_action = None
            except Exception:
                llm_action = None

        with torch.no_grad():
            obs_b, act_b = self.vqvae._normalize_batch([obs_seq], [act_seq])
            obs_emb, act_emb = self.vqvae._encode_text_batch(obs_b, act_b)
            z = self.vqvae.encoder(obs_emb, act_emb)
            q, _, _ = self.vqvae.quantizer(z)
            logits = self.vqvae.decoder(q, obs_emb)[0, -1, :]

        if adm:
            scores = {}
            stoi, unk = self.vqvae.action_vocab.stoi, self.vqvae.action_vocab.stoi[ActionVocab.UNK_TOKEN]
            for a in adm:
                s = str(a).strip()
                idx = stoi.get(s, unk)
                if idx == unk and s.lower() != s:
                    idx = stoi.get(s.lower(), unk)
                scores[a] = float(logits[idx])
            if llm_action and llm_action in scores:
                scores[llm_action] += self.llm_weight * 10
            a = max(scores, key=scores.get)
        else:
            pred_id = logits.argmax().item()
            a = self.vqvae.action_vocab.decode(pred_id)
            if a in (ActionVocab.PAD_TOKEN, ActionVocab.UNK_TOKEN):
                a = "look"

        if adm and a not in adm:
            a = self.rng.choice(adm)
        if a in (ActionVocab.PAD_TOKEN, ActionVocab.UNK_TOKEN):
            a = "look"

        self.act_h.append(a)
        self.history.append((prompt, a))
        if len(self.act_h) > self.w * 2:
            self.obs_h = self.obs_h[-self.w:]
            self.act_h = self.act_h[-self.w:]

        return str(a), {"prompt": None, "response": llm_action, "nb_tokens": self.token_counter(text=obs or "")}


def build_argparser(parser=None):
    p = parser or argparse.ArgumentParser()
    g = p.add_argument_group("llm-vqvae")
    g.add_argument("--api-key", required=True)
    g.add_argument("--api-url", default="https://tritonai-api.ucsd.edu")
    g.add_argument("--model", default="api-llama-4-scout")
    g.add_argument("--vqvae-checkpoint", default="latent-action/checkpoints/vqvae_checkpoint.pt")
    g.add_argument("--window-size", type=int, default=None)
    g.add_argument("--llm-weight", type=float, default=0.3)
    g.add_argument("--seed", type=int, default=20241001)
    return p


register("llm-vqvae", "LLM + VQ-VAE", LLMVQVAEAgent, build_argparser)
