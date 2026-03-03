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
        self.prompt_variant = kwargs.get("prompt_variant", "default")

        # Variant "history" or "naive": compress raw history without scratchpad
        # Variant "compressed": use scratchpad + compress
        # Variant "structured": use scratchpad, structured compression
        self.use_scratchpad = self.variant in {
            "compressed",
            "structured",
            "rag",
            "option",
        }
        self.use_compression = self.variant in {
            "history",      # NEW: naive variant - compress raw history without scratchpad
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
                prompt_variant=self.prompt_variant,
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
        self.task_related_actions: List[str] = []  # Dynamic task-specific actions
        self.task_entities: List[str] = []  # Key entities from recipe/goal
        self.last_compression_info: Dict[str, Any] = {}  # Store last compression LLM call
        self.task_recipe: str = ""  # Store extracted recipe/task

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
        self.task_related_actions = []
        self.task_entities = []
        self.last_compression_info = {}
        self.task_recipe = ""  # Reset cached recipe
        if self.scratchpad is not None:
            self.scratchpad.reset()
            self.scratchpad.update(obs)
        self.raw_memory.append(obs)

    def _extract_task_info(self) -> str:
        """Extract task/recipe information from scratchpad facts."""
        if self.task_recipe:
            return self.task_recipe
        
        if self.scratchpad is None:
            return ""
        
        exported = self.scratchpad.export()
        facts = exported.get("facts", [])
        
        # Look for recipe-related facts
        recipe_facts = []
        for fact in facts:
            fact_lower = fact.lower()
            if any(keyword in fact_lower for keyword in ["recipe", "ingredient", "directions", "dice", "chop", "roast", "prepare meal", "cook"]):
                recipe_facts.append(fact)
        
        if recipe_facts:
            self.task_recipe = "\\n".join(recipe_facts[:10])  # Cache it
            return self.task_recipe
        
        return ""

    def act(self, obs, reward, done, info):
        self.step_count += 1
        updates = {}
        self.raw_memory.append(obs)

        if self.scratchpad is not None:
            updates = self.scratchpad.update(obs)

        compression_info = {}
        if self._should_compress():
            compressed = self._compress_now()
            if compressed is not None:
                self.compressed_history.append(compressed)
                # Store compression info if it has LLM call details
                if "_llm_prompt" in compressed:
                    compression_info = {
                        "compression_prompt": compressed.get("_llm_prompt", ""),
                        "compression_response": compressed.get("_llm_response", ""),
                        "compression_usage": compressed.get("_llm_usage", {}),
                    }
                    self.last_compression_info = compression_info

        retrieved = []
        if self.retriever is not None:
            retrieved = self.retriever.query(obs, self.compressed_history)

        # Call _generate_action_with_stats to get both action and LLM stats
        action, llm_stats = self._generate_action_with_stats(obs, info, updates, retrieved)
        
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
            "nb_tokens": llm_stats.get("nb_tokens_prompt", 0) + llm_stats.get("nb_tokens_response", 0),
            "nb_tokens_prompt": llm_stats.get("nb_tokens_prompt", 0),
            "nb_tokens_response": llm_stats.get("nb_tokens_response", 0),
            "nb_tokens_thinking": llm_stats.get("nb_tokens_thinking", 0),
            "thinking": llm_stats.get("thinking", None),
        }
        
        # Add compression info if it happened this step
        if compression_info:
            stats["compression_prompt"] = compression_info.get("compression_prompt", "")
            stats["compression_response"] = compression_info.get("compression_response", "")
            stats["compression_usage"] = compression_info.get("compression_usage", {})
        
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
        """
        Compress raw history without using scratchpad.
        This is the most naive variant - directly give history to LLM compressor.
        """
        if self.compressor is not None:
            return self.compressor.compress_raw_history(self.history)
        
        # Fallback if no compressor
        recent = self.raw_memory[-self.max_context_items :]
        summary = " | ".join(line[:120].replace("\n", " ") for line in recent)
        return {
            "strategy": "free_text",
            "summary": summary or "No history yet.",
            "top_facts": [],
            "top_rules": [],
        }

    def _generate_action_with_stats(
        self,
        obs: str,
        info: Dict[str, Any],
        updates: Dict[str, Any],
        retrieved: List[Dict[str, Any]],
    ) -> tuple:
        """Generate action and return (action, stats_dict) with token usage info."""
        admissible = list(info.get("admissible_commands") or [])
        llm_stats = {
            "nb_tokens_prompt": 0,
            "nb_tokens_response": 0,
            "nb_tokens_thinking": 0,
            "thinking": None,
        }

        option = self.option_module.choose_option(obs)
        option_candidates = self.option_module.option_actions(option)

        heuristics = self._heuristic_actions(obs, updates, retrieved)
        candidates = option_candidates + heuristics

        if self.use_llm_policy:
            llm_action, llm_response_stats = self._llm_action_with_stats(
                obs, updates, retrieved, admissible, candidates
            )
            if llm_action:
                return llm_action, llm_response_stats
            llm_stats = llm_response_stats  # Keep stats even if action was None

        # Fallback to heuristics
        if admissible:
            admissible_set = set(admissible)
            for candidate in candidates:
                if candidate in admissible_set:
                    return candidate, llm_stats
            return str(self.rng.choice(admissible)), llm_stats

        for candidate in candidates:
            if candidate:
                return candidate, llm_stats

        final_action = str(self.rng.choice(self.default_actions))
        return final_action, llm_stats

    def _heuristic_actions(
        self, obs: str, updates: Dict[str, Any], retrieved: List[Dict[str, Any]]
    ) -> List[str]:
        lowered = (obs or "").lower()
        actions: List[str] = []

        # Extract LLM-recommended actions from retrieved compressed memory
        llm_recommended = []
        llm_avoid = []
        if retrieved:
            for mem in retrieved:
                llm_recommended.extend(mem.get("recommended_actions", []))
                llm_avoid.extend(mem.get("avoid_actions", []))
        
        # Remove duplicates
        llm_recommended = list(dict.fromkeys(llm_recommended))[:12]
        llm_avoid = list(dict.fromkeys(llm_avoid))

        if any(x in lowered for x in ["what do i do", "stuck", "can't"]):
            actions.append("help")

        if "locked" in lowered:
            actions.extend(["examine door", "unlock door", "open door"])

        if "inventory" not in lowered and self.step_count % 9 == 0:
            actions.append("inventory")

        # Extract entities from observations (avoid verbs like "open", "close")
        # Common verbs to skip
        skip_words = {
            "open", "close", "take", "drop", "examine", "look", "read", 
            "scan", "find", "make", "like", "have", "here", "were", "over", 
            "that", "this", "your", "there", "something", "nothing"
        }
        nouns = [w for w in re.findall(r"\b[a-z]{4,}\b", lowered) if w not in skip_words]
        
        # Extract recipe ingredients if present
        if "ingredients:" in lowered:
            # Find lines after "ingredients:" and before "directions:"
            ingredients_section = lowered.split("ingredients:")
            if len(ingredients_section) > 1:
                ingredients_text = ingredients_section[1].split("directions:")[0]
                # Extract ingredient names (simple words)
                recipe_items = re.findall(r"\b([a-z]{4,})\b", ingredients_text)
                for item in recipe_items[:3]:  # First 3 ingredients
                    if item not in skip_words:
                        actions.extend([f"take {item}", f"examine {item}", f"examine fridge"])
        
        # Extract from environment description (fridge, stove, counter, etc.)
        if nouns:
            focus = nouns[0]
            if focus not in skip_words:
                actions.extend([f"examine {focus}", f"take {focus}"])

        if updates.get("failures"):
            actions.append("look")

        if retrieved:
            actions.append("look")

        # Priority 1: LLM-recommended actions from compressed memory
        if llm_recommended:
            actions.extend(llm_recommended)
        
        # Priority 2: Task-related actions from static extraction
        if self.task_related_actions:
            actions.extend(self.task_related_actions[:8])
        
        # Priority 3: Navigation as fallback (lower priority)
        actions.extend(["north", "south", "east", "west"])
        
        # Filter out actions that LLM says to avoid
        if llm_avoid:
            actions = [a for a in actions if a not in llm_avoid]
        
        return actions

    def _llm_action_with_stats(
        self,
        obs: str,
        updates: Dict[str, Any],
        retrieved: List[Dict[str, Any]],
        admissible: List[str],
        candidates: List[str],
    ) -> tuple:
        """Call LLM and return (action, stats_dict) with token usage."""
        stats = {
            "nb_tokens_prompt": 0,
            "nb_tokens_response": 0,
            "nb_tokens_thinking": 0,
            "thinking": None,
        }
        
        api_key = self.llm_api_key or os.getenv(self.llm_api_key_env)
        if not api_key:
            return None, stats

        try:
            prompt = self._build_llm_prompt(obs, updates, retrieved, admissible, candidates)
            response_json = self._query_llm(prompt, api_key)
            
            # Extract action
            raw = self._extract_action_text(response_json)
            action = self._sanitize_action(raw)
            
            # Extract stats from API response
            if response_json and "usage" in response_json:
                usage = response_json["usage"]
                stats["nb_tokens_prompt"] = usage.get("prompt_tokens", 0)
                stats["nb_tokens_response"] = usage.get("completion_tokens", 0)
                # For reasoning tokens, check if model returns them
                stats["nb_tokens_thinking"] = usage.get("reasoning_tokens", 0)
            
            # Extract reasoning content if available
            if response_json and "choices" in response_json and response_json["choices"]:
                message = response_json["choices"][0].get("message", {})
                reasoning = message.get("reasoning_content", "")
                if reasoning:
                    stats["thinking"] = reasoning
                    # If reasoning_tokens not in usage, estimate from content
                    if stats["nb_tokens_thinking"] == 0 and reasoning:
                        stats["nb_tokens_thinking"] = len(reasoning.split())
            
            if not action:
                return None, stats

            if admissible:
                final_action = self._align_to_admissible(action, admissible)
                return final_action, stats
            return action, stats
            
        except Exception as e:
            return None, stats

    def _build_llm_prompt(
        self,
        obs: str,
        updates: Dict[str, Any],
        retrieved: List[Dict[str, Any]],
        admissible: List[str],
        candidates: List[str],
    ) -> str:
        parts = []
        
        # Core instruction
        parts.append("You are playing a text-based cooking game. Your GOAL is to follow the recipe and prepare a meal.")
        parts.append("\nSTRATEGY:")
        parts.append("1. If you see a recipe, follow its steps in order (e.g., find ingredients, chop, roast, prepare meal)")
        parts.append("2. Interact with objects you can see BEFORE exploring (e.g., 'take apple', 'examine fridge')")
        parts.append("3. If stuck or getting errors, use 'look' or 'inventory' to reorient")
        parts.append("4. Avoid repeating failed actions\n")
        
        # Extract and show task/recipe from scratchpad facts
        task_info = self._extract_task_info()
        if task_info:
            parts.append(f"TASK/RECIPE:\n{task_info}\n")
        
        # Show memory guidance from compression (if available)
        # This is variant-agnostic: compression.py decides what to extract based on prompt_variant
        if self.compressed_history:
            last_compression = self.compressed_history[-1]
            
            # Show recommended actions if compressed memory provides them
            if "recommended_actions" in last_compression and last_compression["recommended_actions"]:
                rec = last_compression["recommended_actions"][:8]
                parts.append(f"Memory Guidance - Recommended: {', '.join(rec)}")
            
            # Show actions to avoid if provided
            if "avoid_actions" in last_compression and last_compression["avoid_actions"]:
                avoid = last_compression["avoid_actions"][:5]
                parts.append(f"Memory Guidance - Avoid: {', '.join(avoid)}")
            
            # Show strategy hint if provided
            if "strategy_hint" in last_compression and last_compression["strategy_hint"]:
                parts.append(f"Memory Guidance - Strategy: {last_compression['strategy_hint']}")
            
            # Show summary if provided (for variants that don't extract actions)
            if "summary" in last_compression and last_compression["summary"] and not last_compression.get("recommended_actions"):
                summary = last_compression["summary"]
                if len(summary) > 200:
                    summary = summary[:200] + "..."
                parts.append(f"Memory Summary: {summary}")
            
            parts.append("")
        
        # Always show recent action history for immediate context
        if self.history:
            recent = self.history[-5:]
            parts.append("Recent actions:")
            for h in recent:
                parts.append(f"  Step {h['step']}: {h['action']} → reward={h['reward']:.1f}")
            parts.append("")
        
        # Current observation
        parts.append(f"Current observation:\n{obs}\n")
        
        # Show available actions
        parts.append("\nOutput exactly ONE action command. Output ONLY the command with no explanation.")
        
        # Admissible commands (if provided)
        if admissible:
            adm_text = ", ".join(admissible[:15])
            parts.append(f"\nValid commands: {adm_text}")
        
        # Filter and prioritize candidates
        # Deprioritize pure navigation if there are other options
        priority_candidates = [c for c in candidates if c not in ['north', 'south', 'east', 'west']]
        nav_candidates = [c for c in candidates if c in ['north', 'south', 'east', 'west']]
        
        if priority_candidates:
            parts.append(f"Suggested actions (prioritized): {', '.join(priority_candidates[:6])}\n")
        if nav_candidates and len(priority_candidates) < 3:
            parts.append(f"Navigation options: {', '.join(nav_candidates)}\n")
        
        parts.append("\nYour command:")
        
        return "\n".join(parts)

    def _query_llm(self, prompt: str, api_key: str) -> Dict[str, Any]:
        payload = {
            "model": self.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 256,  # Increased to allow both reasoning + action content
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
        parts = [f"obs={obs}"]
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
        choices=["baseline", "history", "compressed", "structured", "rag", "option"],
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
    group.add_argument(
        "--prompt-variant",
        default="default",
        help="Compression prompt variant (default, v1, v2, v3). Default: %(default)s",
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
