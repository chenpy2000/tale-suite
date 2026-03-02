import argparse
import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

import numpy as np

import tales
from memory import MemoryCompressor, ObservationParser, OptionModule, Retriever, Scratchpad
from tales.agent import register
from tales.token import get_token_counter


class MemoryAgent(tales.Agent):
    """Custom memory agent with scratchpad, compression, optional retrieval, and options."""

    def __init__(self, **kwargs):
        self.seed = kwargs.get("seed", 20241001)
        self.rng = np.random.RandomState(self.seed)
        self.token_counter = get_token_counter()

        self.variant = kwargs.get("memory_variant", "structured")
        self.compress_every = int(kwargs.get("compress_every", 8))
        self.max_context_items = int(kwargs.get("max_context_items", 12))
        self.use_llm_policy = bool(kwargs.get("use_llm_policy", False))
        self.llm_api_url = kwargs.get(
            "llm_api_url", "https://tritonai-api.ucsd.edu/v1/chat/completions"
        )
        self.llm_model = kwargs.get("llm_model", "api-llama-4-scout")
        self.llm_api_key = kwargs.get("llm_api_key")
        self.llm_api_key_env = kwargs.get("llm_api_key_env", "TRITON_API_KEY")
        self.llm_timeout = int(kwargs.get("llm_timeout", 30))
        self.use_llm_parser = bool(kwargs.get("use_llm_parser", False))
        self.use_llm_compressor = bool(kwargs.get("use_llm_compressor", False))
        self.parser_model = kwargs.get("parser_model") or self.llm_model
        self.compressor_model = kwargs.get("compressor_model") or self.llm_model

        self.use_scratchpad = self.variant in {
            "structured",
            "rag",
            "option",
        }
        self.use_compression = self.variant in {
            "compressed",
            "structured",
            "rag",
            "option",
        }
        self.use_retrieval = kwargs.get("use_retrieval", self.variant == "rag")
        self.use_option_module = kwargs.get(
            "use_option_module", self.variant == "option"
        )

        compression_strategy = kwargs.get("compression_strategy")
        if compression_strategy is None:
            compression_strategy = (
                "free_text" if self.variant == "compressed" else "structured"
            )

        parser = ObservationParser(
            use_llm=self.use_llm_parser,
            llm_api_url=self.llm_api_url,
            llm_model=self.parser_model,
            llm_api_key=self.llm_api_key,
            llm_api_key_env=self.llm_api_key_env,
            llm_timeout=self.llm_timeout,
        )
        self.scratchpad = Scratchpad(parser=parser) if self.use_scratchpad else None
        self.compressor = (
            MemoryCompressor(
                strategy=compression_strategy,
                max_items=self.max_context_items,
                use_llm=self.use_llm_compressor,
                llm_api_url=self.llm_api_url,
                llm_model=self.compressor_model,
                llm_api_key=self.llm_api_key,
                llm_api_key_env=self.llm_api_key_env,
                llm_timeout=self.llm_timeout,
            )
            if self.use_compression
            else None
        )
        self.retriever = Retriever(top_k=3) if self.use_retrieval else None
        self.option_module = OptionModule(enabled=self.use_option_module)

        self.history: List[Dict[str, Any]] = []
        self.raw_memory: List[str] = []
        self.compressed_history: List[Dict[str, Any]] = []
        self.step_count = 0

        # fmt:off
        self.default_actions = [
            "north", "south", "east", "west", "up", "down",
            "look", "inventory", "examine", "open", "close",
            "take", "drop", "use", "help", "wait", "YES",
        ]
        # fmt:on

    @property
    def uid(self):
        return (
            f"MemoryAgent_v{self.variant}"
            f"_s{self.seed}"
            f"_ce{self.compress_every}"
            f"_retr{int(self.use_retrieval)}"
            f"_opt{int(self.use_option_module)}"
            f"_llm{int(self.use_llm_policy)}"
            f"_lp{int(self.use_llm_parser)}"
            f"_lc{int(self.use_llm_compressor)}"
        )

    @property
    def params(self):
        return {
            "agent_type": "memory",
            "memory_variant": self.variant,
            "seed": self.seed,
            "compress_every": self.compress_every,
            "use_retrieval": self.use_retrieval,
            "use_option_module": self.use_option_module,
            "use_llm_policy": self.use_llm_policy,
            "llm_model": self.llm_model,
            "use_llm_parser": self.use_llm_parser,
            "use_llm_compressor": self.use_llm_compressor,
        }

    def reset(self, obs, info, env):
        self.step_count = 0
        self.history = []
        self.raw_memory = []
        self.compressed_history = []
        if self.scratchpad is not None:
            self.scratchpad.reset()
            self.scratchpad.update(obs)
        self.raw_memory.append(obs)

    def act(self, obs, reward, done, info):
        self.step_count += 1
        updates = {}
        self.raw_memory.append(obs)

        if self.scratchpad is not None:
            updates = self.scratchpad.update(obs)

        if self._should_compress():
            compressed = self._compress_now()
            if compressed is not None:
                self.compressed_history.append(compressed)

        retrieved = []
        if self.retriever is not None:
            retrieved = self.retriever.query(obs, self.compressed_history)

        action = self._generate_action(obs, info, updates, retrieved)
        self.history.append(
            {
                "step": self.step_count,
                "observation": obs,
                "action": action,
                "reward": reward,
                "done": done,
            }
        )

        stats = {
            "prompt": self._build_prompt_debug(obs, updates, retrieved),
            "response": action,
            "nb_tokens": self.token_counter(text=obs),
        }
        return action, stats

    def _should_compress(self) -> bool:
        if self.compressor is None:
            return False
        return self.step_count % max(self.compress_every, 1) == 0

    def _compress_now(self) -> Optional[Dict[str, Any]]:
        if self.compressor is None:
            return None
        if self.scratchpad is not None:
            return self.scratchpad.compress(self.compressor)
        return self._compress_raw_history()

    def _compress_raw_history(self) -> Dict[str, Any]:
        recent = self.raw_memory[-self.max_context_items :]
        summary = " | ".join(line[:120].replace("\n", " ") for line in recent)
        return {
            "strategy": "free_text",
            "summary": summary or "No history yet.",
            "top_facts": [],
            "top_rules": [],
        }

    def _generate_action(
        self,
        obs: str,
        info: Dict[str, Any],
        updates: Dict[str, Any],
        retrieved: List[Dict[str, Any]],
    ) -> str:
        admissible = list(info.get("admissible_commands") or [])

        option = self.option_module.choose_option(obs)
        option_candidates = self.option_module.option_actions(option)

        heuristics = self._heuristic_actions(obs, updates, retrieved)
        candidates = option_candidates + heuristics

        if self.use_llm_policy:
            llm_action = self._llm_action(obs, updates, retrieved, admissible, candidates)
            if llm_action:
                return llm_action

        if admissible:
            admissible_set = set(admissible)
            for candidate in candidates:
                if candidate in admissible_set:
                    return candidate
            return str(self.rng.choice(admissible))

        for candidate in candidates:
            if candidate:
                return candidate

        return str(self.rng.choice(self.default_actions))

    def _heuristic_actions(
        self, obs: str, updates: Dict[str, Any], retrieved: List[Dict[str, Any]]
    ) -> List[str]:
        lowered = (obs or "").lower()
        actions: List[str] = []

        if any(x in lowered for x in ["what do i do", "stuck", "can't"]):
            actions.append("help")

        if "locked" in lowered:
            actions.extend(["examine door", "unlock door", "open door"])

        if "inventory" not in lowered and self.step_count % 9 == 0:
            actions.append("inventory")

        nouns = re.findall(r"\b[a-z]{4,}\b", lowered)
        if nouns:
            focus = nouns[0]
            actions.extend([f"examine {focus}", f"take {focus}"])

        if updates.get("failures"):
            actions.append("look")

        if retrieved:
            actions.append("look")

        actions.extend(["north", "south", "east", "west"])
        return actions

    def _llm_action(
        self,
        obs: str,
        updates: Dict[str, Any],
        retrieved: List[Dict[str, Any]],
        admissible: List[str],
        candidates: List[str],
    ) -> Optional[str]:
        api_key = self.llm_api_key or os.getenv(self.llm_api_key_env)
        if not api_key:
            return None

        try:
            prompt = self._build_llm_prompt(obs, updates, retrieved, admissible, candidates)
            response_json = self._query_llm(prompt, api_key)
            raw = self._extract_action_text(response_json)
            action = self._sanitize_action(raw)
            if not action:
                return None

            if admissible:
                return self._align_to_admissible(action, admissible)
            return action
        except Exception:
            return None

    def _build_llm_prompt(
        self,
        obs: str,
        updates: Dict[str, Any],
        retrieved: List[Dict[str, Any]],
        admissible: List[str],
        candidates: List[str],
    ) -> str:
        snapshot = {}
        if self.scratchpad is not None:
            exported = self.scratchpad.export()
            snapshot = {
                "facts": exported["facts"][-self.max_context_items :],
                "rules": exported["rules"][-self.max_context_items :],
                "preconditions": exported["preconditions"][-self.max_context_items :],
                "failures": exported["failures"][-8:],
                "successes": exported["successes"][-8:],
            }

        compressed = self.compressed_history[-3:]
        admissible_text = ", ".join(admissible[:50]) if admissible else "N/A"
        candidate_text = ", ".join(candidates[:20]) if candidates else "N/A"

        return (
            "You are an agent in a text game. Return exactly one short command.\n"
            "Do not explain. No punctuation unless needed.\n\n"
            f"Observation:\n{obs}\n\n"
            f"Scratchpad:\n{json.dumps(snapshot, ensure_ascii=True)}\n\n"
            f"Compressed history:\n{json.dumps(compressed, ensure_ascii=True)}\n\n"
            f"Retrieved memory:\n{json.dumps(retrieved[:3], ensure_ascii=True)}\n\n"
            f"Heuristic candidates:\n{candidate_text}\n\n"
            f"Admissible commands:\n{admissible_text}\n"
        )

    def _query_llm(self, prompt: str, api_key: str) -> Dict[str, Any]:
        payload = {
            "model": self.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 32,
        }
        req = urllib.request.Request(
            self.llm_api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.llm_timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"LLM HTTP {exc.code}: {body}") from exc

    @staticmethod
    def _extract_action_text(response_json: Dict[str, Any]) -> str:
        choices = response_json.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content", "")
        return str(content).strip()

    @staticmethod
    def _sanitize_action(text: str) -> str:
        if not text:
            return ""
        # keep first non-empty line as single action
        for line in str(text).splitlines():
            clean = line.strip()
            if clean:
                return clean[:120]
        return ""

    def _align_to_admissible(self, action: str, admissible: List[str]) -> Optional[str]:
        if action in admissible:
            return action

        lowered_action = action.lower()
        for cmd in admissible:
            if cmd.lower() == lowered_action:
                return cmd
        for cmd in admissible:
            if cmd.lower() in lowered_action or lowered_action in cmd.lower():
                return cmd
        if admissible:
            return str(self.rng.choice(admissible))
        return action

    def _build_prompt_debug(
        self, obs: str, updates: Dict[str, Any], retrieved: List[Dict[str, Any]]
    ) -> str:
        parts = [f"obs={obs[:200]}"]
        if updates:
            parts.append(f"updates={updates}")
        if self.scratchpad is not None:
            snapshot = self.scratchpad.export()
            brief = {
                "facts": snapshot["facts"][:4],
                "rules": snapshot["rules"][:4],
                "failures": snapshot["failures"][-4:],
            }
            parts.append(f"scratchpad={brief}")
        if retrieved:
            parts.append(f"retrieved={retrieved[:2]}")
        return "\n".join(parts)


def build_argparser(parser=None):
    parser = parser or argparse.ArgumentParser()
    group = parser.add_argument_group("MemoryAgent settings")

    group.add_argument(
        "--seed",
        type=int,
        default=20241001,
        help="Seed for action sampling. Default: %(default)s",
    )
    group.add_argument(
        "--memory-variant",
        choices=["baseline", "compressed", "structured", "rag", "option"],
        default="structured",
        help="Memory variant to run. Default: %(default)s",
    )
    group.add_argument(
        "--compress-every",
        type=int,
        default=8,
        help="Run compression every N steps. Default: %(default)s",
    )
    group.add_argument(
        "--compression-strategy",
        choices=["free_text", "structured"],
        default=None,
        help="Compression strategy override. Default: chosen by memory variant.",
    )
    group.add_argument(
        "--use-retrieval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable retrieval from compressed history.",
    )
    group.add_argument(
        "--use-option-module",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable high-level option abstraction.",
    )
    group.add_argument(
        "--max-context-items",
        type=int,
        default=12,
        help="Max items retained by compressor. Default: %(default)s",
    )
    group.add_argument(
        "--use-llm-policy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable remote LLM API for action generation.",
    )
    group.add_argument(
        "--llm-api-url",
        default="https://tritonai-api.ucsd.edu/v1/chat/completions",
        help="Chat completions endpoint URL.",
    )
    group.add_argument(
        "--llm-model",
        default="api-llama-4-scout",
        help="Remote model identifier for the endpoint.",
    )
    group.add_argument(
        "--llm-api-key",
        default=None,
        help="API key string. Prefer env var for safety.",
    )
    group.add_argument(
        "--llm-api-key-env",
        default="TRITON_API_KEY",
        help="Environment variable name holding API key.",
    )
    group.add_argument(
        "--llm-timeout",
        type=int,
        default=30,
        help="LLM HTTP timeout in seconds. Default: %(default)s",
    )
    group.add_argument(
        "--use-llm-parser",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use LLM to parse observations into structured updates.",
    )
    group.add_argument(
        "--use-llm-compressor",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use LLM for scratchpad compression.",
    )
    group.add_argument(
        "--parser-model",
        default=None,
        help="Model for parser extraction. Default: use --llm-model",
    )
    group.add_argument(
        "--compressor-model",
        default=None,
        help="Model for compression. Default: use --llm-model",
    )

    return parser


register(
    name="memory-agent",
    desc=(
        "Custom memory agent with scratchpad updates, periodic compression, "
        "optional retrieval, and optional option abstraction."
    ),
    klass=MemoryAgent,
    add_arguments=build_argparser,
)
