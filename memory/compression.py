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
    ):
        self.strategy = strategy
        self.max_items = max_items
        self.use_llm = use_llm
        self.llm_api_url = llm_api_url
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key
        self.llm_api_key_env = llm_api_key_env
        self.llm_timeout = llm_timeout

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
        return out

    @staticmethod
    def _as_list(value: Any) -> List[str]:
        if not isinstance(value, list):
            value = [value]
        return [str(x).strip() for x in value if str(x).strip()]
