import json
import os
import urllib.request
from typing import Any, Dict, List


class MemoryCompressor:
    """Compresses scratchpad memory with free-text or structured strategy."""

    def __init__(
        self,
        strategy: str = "structured",
        max_items: int = 12,
        use_llm: bool = False,
        llm_api_url: str = "https://tritonai-api.ucsd.edu/v1/chat/completions",
        llm_model: str = "api-llama-4-scout",
        llm_api_key: str | None = None,
        llm_api_key_env: str = "TRITON_API_KEY",
        llm_timeout: int = 30,
        prompt_variant: str = "default",
    ):
        self.strategy = strategy
        self.max_items = max_items
        self.use_llm = use_llm
        self.llm_api_url = llm_api_url
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key
        self.llm_api_key_env = llm_api_key_env
        self.llm_timeout = llm_timeout
        self.prompt_variant = prompt_variant

    def compress(self, scratchpad_state: Any) -> Dict[str, Any]:
        if self.use_llm:
            llm_out = self._compress_with_llm(scratchpad_state)
            if llm_out is not None:
                return llm_out

        if self.strategy == "free_text":
            return self._free_text_compress(scratchpad_state)
        if self.strategy == "structured":
            return self._structured_compress(scratchpad_state)
        raise ValueError(f"Unsupported compression strategy: {self.strategy}")

    def compress_raw_history(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compress raw history directly without using scratchpad.
        This is the most naive variant: just give LLM the raw observations, actions, feedback.
        
        Args:
            history: List of dicts with keys: step, observation, action, reward, done
        
        Returns:
            Compression result with recommended_actions, avoid_actions, strategy_hint
        """
        if not self.use_llm:
            # Fallback: simple text summary
            recent = history[-self.max_items:]
            summary_lines = []
            for h in recent:
                obs = h.get("observation", "")[:80].replace("\n", " ")
                action = h.get("action", "")
                summary_lines.append(f"Action: {action} -> {obs}")
            return {
                "strategy": "free_text",
                "summary": " | ".join(summary_lines) or "No history yet.",
                "top_facts": [],
                "top_rules": [],
            }
        
        # Use LLM to compress raw history
        api_key = self.llm_api_key or os.getenv(self.llm_api_key_env)
        if not api_key:
            return {
                "strategy": "free_text",
                "summary": "No API key",
                "top_facts": [],
                "top_rules": [],
            }
        
        # Build prompt from raw history
        recent = history[-self.max_items:]
        history_text = []
        for i, h in enumerate(recent, 1):
            obs = h.get("observation", "").strip()
            action = h.get("action", "")
            history_text.append(f"Step {i}:\nObservation: {obs}\nAction: {action}\n")
        
        prompt = f"""You are analyzing a text-based game agent's history to extract actionable information.

Game History (most recent {len(recent)} steps):
{''.join(history_text)}

Based on this history, extract:
1. recommended_actions: List of 3-5 actions that seem promising or useful
2. avoid_actions: List of 3-5 actions that failed or seem unproductive
3. strategy_hint: A brief strategy suggestion (1-2 sentences)

Return a JSON object with these three fields.
"""
        
        payload = {
            "model": self.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 500,
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
                raw = json.loads(resp.read().decode("utf-8"))
            
            content = self._extract_content(raw)
            usage = raw.get("usage", {})
            
            obj = self._extract_json_object(content)
            if obj is None:
                obj = {
                    "recommended_actions": [],
                    "avoid_actions": [],
                    "strategy_hint": content[:200] if content else "No strategy extracted",
                }
            
            # Add standard fields
            result = {
                "strategy": "raw_history_v0",
                "recommended_actions": obj.get("recommended_actions", []),
                "avoid_actions": obj.get("avoid_actions", []),
                "strategy_hint": obj.get("strategy_hint", ""),
                "top_facts": [],
                "top_rules": [],
                "_llm_prompt": prompt,
                "_llm_response": content,
                "_llm_usage": usage,
            }
            return result
            
        except Exception as e:
            return {
                "strategy": "raw_history_v0",
                "summary": f"LLM compression failed: {e}",
                "recommended_actions": [],
                "avoid_actions": [],
                "strategy_hint": "",
                "top_facts": [],
                "top_rules": [],
            }

    def _free_text_compress(self, state: Any) -> Dict[str, Any]:
        important = sorted(
            state.importance.items(), key=lambda kv: kv[1], reverse=True
        )[: self.max_items]
        key_points = [k for k, _ in important]

        summary = " | ".join(key_points)
        if not summary:
            summary = "No salient memory yet."

        return {
            "strategy": "free_text",
            "summary": summary,
            "top_facts": list(state.facts[: self.max_items]),
            "top_rules": list(state.rules[: self.max_items]),
        }

    def _structured_compress(self, state: Any) -> Dict[str, Any]:
        fact_scores = [(f, state.importance.get(f, 0.0)) for f in state.facts]
        rule_scores = [(r, state.importance.get(r, 0.0)) for r in state.rules]

        top_facts = [
            fact
            for fact, _ in sorted(fact_scores, key=lambda kv: kv[1], reverse=True)[
                : self.max_items
            ]
        ]
        top_rules = [
            rule
            for rule, _ in sorted(rule_scores, key=lambda kv: kv[1], reverse=True)[
                : self.max_items
            ]
        ]

        frequent_failures = self._top_k(state.failures, k=6)

        return {
            "strategy": "structured",
            "top_facts": top_facts,
            "top_rules": top_rules,
            "preconditions": list(state.preconditions[: self.max_items]),
            "frequent_failures": frequent_failures,
            "successes": list(state.successes[-self.max_items :]),
        }

    def _compress_with_llm(self, state: Any) -> Dict[str, Any] | None:
        # Route to experimental versions if variant is specified
        if self.prompt_variant != "default":
            return self._compress_with_llm_experimental(state, self.prompt_variant)
        
        api_key = self.llm_api_key or os.getenv(self.llm_api_key_env)
        if not api_key:
            return None

        scratchpad = {
            "facts": list(state.facts),
            "rules": list(state.rules),
            "preconditions": list(state.preconditions),
            "failures": list(state.failures[-20:]),
            "successes": list(state.successes[-20:]),
            "importance": dict(state.importance),
        }
        prompt = (
            f"Compress this game scratchpad using strategy={self.strategy}.\n"
            "Return strict JSON with keys: strategy, top_facts, top_rules, preconditions, frequent_failures, successes, summary.\n"
            "Use arrays for all *_facts/rules/preconditions/failures/successes keys.\n"
            "Keep concise and high-signal.\n\n"
            f"Scratchpad JSON:\n{json.dumps(scratchpad, ensure_ascii=True)}\n"
        )
        payload = {
            "model": self.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 500,
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
                raw = json.loads(resp.read().decode("utf-8"))
            content = self._extract_content(raw)
            obj = self._extract_json_object(content)
            if obj is None:
                return None
            return self._normalize_compression(obj)
        except Exception:
            return None

    @staticmethod
    def _top_k(items: List[str], k: int) -> List[str]:
        if not items:
            return []
        counts: Dict[str, int] = {}
        for item in items:
            counts[item] = counts.get(item, 0) + 1
        return [
            item for item, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:k]
        ]

    @staticmethod
    def _extract_content(response_json: Dict[str, Any]) -> str:
        choices = response_json.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return str(message.get("content", "")).strip()

    @staticmethod
    def _extract_json_object(text: str) -> Dict[str, Any] | None:
        try:
            return json.loads(text)
        except Exception:
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                return None
        return None

    def _normalize_compression(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        out = {
            "strategy": str(obj.get("strategy", self.strategy)),
            "summary": str(obj.get("summary", "")),
            "top_facts": self._as_list(obj.get("top_facts", [])),
            "top_rules": self._as_list(obj.get("top_rules", [])),
            "preconditions": self._as_list(obj.get("preconditions", [])),
            "frequent_failures": self._as_list(obj.get("frequent_failures", [])),
            "successes": self._as_list(obj.get("successes", [])),
            # New fields for action extraction (variant v1)
            "recommended_actions": self._as_list(obj.get("recommended_actions", [])),
            "avoid_actions": self._as_list(obj.get("avoid_actions", [])),
            "key_entities": self._as_list(obj.get("key_entities", [])),
            "task_verbs": self._as_list(obj.get("task_verbs", [])),
            "strategy_hint": str(obj.get("strategy_hint", "")),
        }
        # Keep bounds stable.
        for key in [
            "top_facts",
            "top_rules",
            "preconditions",
            "frequent_failures",
            "successes",
        ]:
            out[key] = out[key][: self.max_items]
        # Bound new action-related fields
        for key in ["recommended_actions", "avoid_actions", "key_entities", "task_verbs"]:
            if key in out:
                out[key] = out[key][: self.max_items]
        return out

    @staticmethod
    def _as_list(value: Any) -> List[str]:
        if not isinstance(value, list):
            value = [value]
        return [str(x).strip() for x in value if str(x).strip()]

    def _compress_with_llm_experimental(self, state: Any, variant: str) -> Dict[str, Any] | None:
        """Route to specific experimental compression variant."""
        if variant == "v1":
            return self._compress_variant_v1(state)
        elif variant == "v2":
            return self._compress_variant_v2(state)
        elif variant == "v3":
            return self._compress_variant_v3(state)
        else:
            # Fallback to default if unknown variant
            return None

    def _compress_variant_v1(self, state: Any) -> Dict[str, Any] | None:
        """Experimental variant v1: Extract actionable information from memory."""
        api_key = self.llm_api_key or os.getenv(self.llm_api_key_env)
        if not api_key:
            return None

        # CUSTOMIZE SCRATCHPAD CONTENT FOR V1
        scratchpad = {
            "facts": list(state.facts),
            "rules": list(state.rules),
            "preconditions": list(state.preconditions),
            "failures": list(state.failures[-30:]),
            "successes": list(state.successes[-30:]),
            "importance": dict(state.importance),
        }
        
        # CUSTOMIZE PROMPT FOR V1 - Focus on extracting actionable information
        prompt = (
            "You are analyzing game memory to extract actionable information.\n\n"
            "TASK: Extract the following from the scratchpad:\n"
            "1. recommended_actions: Actions that should be tried (based on recipe/goal, successful patterns, preconditions)\n"
            "2. avoid_actions: Actions that repeatedly failed and should be avoided\n"
            "3. key_entities: Important objects/locations mentioned in facts (ingredients, tools, containers)\n"
            "4. task_verbs: Key verbs from the recipe/goal (e.g., chop, roast, prepare, take, examine)\n"
            "5. top_facts: Most important facts to remember\n"
            "6. top_rules: Most important rules learned\n"
            "7. strategy_hint: One-sentence hint about what to do next\n\n"
            "ANALYSIS GUIDELINES:\n"
            "- For recommended_actions: Look at recipe/goal in facts, combine task_verbs with key_entities\n"
            "- For avoid_actions: Find actions that failed multiple times (e.g., 'north' if blocked)\n"
            "- For key_entities: Extract nouns from facts (cheese, knife, oven, cookbook, etc.)\n"
            "- For task_verbs: Extract action verbs from rules/preconditions (chop, roast, take, etc.)\n"
            "- Prioritize recipe-related actions over navigation\n\n"
            "Return strict JSON with these EXACT keys:\n"
            "{\n"
            '  "strategy": "action_extraction",\n'
            '  "recommended_actions": ["action1", "action2", ...],  // 8-12 specific actions to try\n'
            '  "avoid_actions": ["action1", "action2", ...],  // 3-5 actions that failed repeatedly\n'
            '  "key_entities": ["entity1", "entity2", ...],  // 5-8 important objects/locations\n'
            '  "task_verbs": ["verb1", "verb2", ...],  // 4-6 key action verbs\n'
            '  "top_facts": ["fact1", "fact2", ...],  // Top facts\n'
            '  "top_rules": ["rule1", "rule2", ...],  // Top rules\n'
            '  "strategy_hint": "One sentence about what to do next",\n'
            '  "summary": "Brief summary of current progress"\n'
            "}\n\n"
            f"Scratchpad:\n{json.dumps(scratchpad, ensure_ascii=True, indent=2)}\n"
        )
        
        payload = {
            "model": self.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 500,
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
                raw = json.loads(resp.read().decode("utf-8"))
            content = self._extract_content(raw)
            obj = self._extract_json_object(content)
            if obj is None:
                return None
            result = self._normalize_compression(obj)
            
            # Store LLM call info for logging (with underscore prefix to not interfere with compression data)
            result["_llm_prompt"] = prompt
            result["_llm_response"] = content
            result["_llm_usage"] = raw.get("usage", {})
            
            return result
        except Exception:
            return None

    def _compress_variant_v2(self, state: Any) -> Dict[str, Any] | None:
        """Experimental variant v2: Customize your prompt/scratchpad here."""
        api_key = self.llm_api_key or os.getenv(self.llm_api_key_env)
        if not api_key:
            return None

        # CUSTOMIZE SCRATCHPAD CONTENT FOR V2
        scratchpad = {
            "facts": list(state.facts),
            "rules": list(state.rules),
            "preconditions": list(state.preconditions),
            "failures": list(state.failures[-20:]),
            "successes": list(state.successes[-20:]),
            "importance": dict(state.importance),
        }
        
        # CUSTOMIZE PROMPT FOR V2
        prompt = (
            f"[VARIANT V2] Compress this game scratchpad using strategy={self.strategy}.\n"
            "Return strict JSON with keys: strategy, top_facts, top_rules, preconditions, frequent_failures, successes, summary.\n"
            "Use arrays for all *_facts/rules/preconditions/failures/successes keys.\n"
            "Keep concise and high-signal.\n\n"
            f"Scratchpad JSON:\n{json.dumps(scratchpad, ensure_ascii=True)}\n"
        )
        
        payload = {
            "model": self.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 500,
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
                raw = json.loads(resp.read().decode("utf-8"))
            content = self._extract_content(raw)
            obj = self._extract_json_object(content)
            if obj is None:
                return None
            return self._normalize_compression(obj)
        except Exception:
            return None

    def _compress_variant_v3(self, state: Any) -> Dict[str, Any] | None:
        """Experimental variant v3: Customize your prompt/scratchpad here."""
        api_key = self.llm_api_key or os.getenv(self.llm_api_key_env)
        if not api_key:
            return None

        # CUSTOMIZE SCRATCHPAD CONTENT FOR V3
        scratchpad = {
            "facts": list(state.facts),
            "rules": list(state.rules),
            "preconditions": list(state.preconditions),
            "failures": list(state.failures[-20:]),
            "successes": list(state.successes[-20:]),
            "importance": dict(state.importance),
        }
        
        # CUSTOMIZE PROMPT FOR V3
        prompt = (
            f"[VARIANT V3] Compress this game scratchpad using strategy={self.strategy}.\n"
            "Return strict JSON with keys: strategy, top_facts, top_rules, preconditions, frequent_failures, successes, summary.\n"
            "Use arrays for all *_facts/rules/preconditions/failures/successes keys.\n"
            "Keep concise and high-signal.\n\n"
            f"Scratchpad JSON:\n{json.dumps(scratchpad, ensure_ascii=True)}\n"
        )
        
        payload = {
            "model": self.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 500,
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
                raw = json.loads(resp.read().decode("utf-8"))
            content = self._extract_content(raw)
            obj = self._extract_json_object(content)
            if obj is None:
                return None
            return self._normalize_compression(obj)
        except Exception:
            return None
