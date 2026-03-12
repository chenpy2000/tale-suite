"""Baseline LLM agent using TritonAI API (same as llm-vqvae but without VQ-VAE)."""
import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np

import tales
from tales.agent import register

_ROOT = Path(__file__).resolve().parent.parent

SYSTEM_PROMPT = (
    "You are an expert text adventure game player. Your goal is to complete the game by maximizing your score.\n\n"
    "You will receive:\n"
    "1. Current observation: what you see in the game right now\n"
    "2. Valid commands: the ONLY actions that will work in this situation\n\n"
    "Output format: Reply with ONLY the exact command from the valid commands list, nothing else. No explanation, no punctuation."
)


class LLMTritonAgent(tales.Agent):
    """Plain LLM over TritonAI (no VQ-VAE). Same API as llm-vqvae for fair comparison."""

    def __init__(
        self,
        api_key,
        api_url,
        model,
        seed=20241001,
        **kwargs,
    ):
        self.api_key = api_key
        self.api_url = api_url.rstrip("/")
        if not self.api_url.endswith("/v1/chat/completions"):
            self.api_url = self.api_url.rstrip("/") + "/v1/chat/completions"
        self.model_name = model
        self.rng = np.random.RandomState(seed)
        print(f"[llm-triton] Baseline LLM: {model} (no VQ-VAE)")

    @property
    def uid(self):
        return f"LLMTriton_{self.model_name}"

    @property
    def params(self):
        return {"agent_type": "llm-triton", "model": self.model_name}

    def reset(self, obs, info, env_name=None):
        pass

    def _query_llm(self, prompt):
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 64,
        }
        req = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        choices = data.get("choices", [])
        if not choices:
            return ""
        content = (choices[0].get("message") or {}).get("content")
        return (content or "").strip()

    def _align_to_admissible(self, action, admissible):
        """Align LLM output to an admissible command. If no match, return random."""
        if not admissible:
            return ""
        action = (action or "").strip()
        if action in admissible:
            return action
        low = action.lower()
        for cmd in admissible:
            if cmd.lower() == low:
                return cmd
        for cmd in admissible:
            if low in cmd.lower() or cmd.lower() in low:
                return cmd
        return self.rng.choice(admissible)

    def act(self, obs, reward, done, infos):
        admissible = list(infos.get("admissible_commands") or [])
        if not admissible:
            return "wait", {"nb_tokens": 0}

        # Limit commands in prompt (same as llm-vqvae)
        show = admissible[:30]
        cmd_list = "\n".join(f"- {c}" for c in show)
        if len(admissible) > 30:
            cmd_list += f"\n... and {len(admissible) - 30} more"

        prompt = f"""Current observation:
{obs}

Valid commands (choose exactly one):
{cmd_list}

Your action:"""

        raw = self._query_llm(prompt)
        action = self._align_to_admissible(raw, admissible)
        return action, {"nb_tokens": 0}


def build_argparser(parser=None):
    parser = parser or argparse.ArgumentParser()
    g = parser.add_argument_group("LLM Triton baseline")
    g.add_argument("--api-key", required=True, help="TritonAI API key")
    g.add_argument("--api-url", default="https://tritonai-api.ucsd.edu")
    g.add_argument("--model", default="api-gpt-oss-120b")
    return parser


register(
    name="llm-triton",
    desc="Baseline LLM via TritonAI (same API as llm-vqvae, no VQ-VAE).",
    klass=LLMTritonAgent,
    add_arguments=build_argparser,
)
