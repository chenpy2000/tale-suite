import re
import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List


class ObservationParser:
    """Heuristic parser that extracts structured updates from text observations."""

    def __init__(
        self,
        use_llm: bool = False,
        llm_api_url: str = "https://tritonai-api.ucsd.edu/v1/chat/completions",
        llm_model: str = "api-llama-4-scout",
        llm_api_key: str | None = None,
        llm_api_key_env: str = "TRITON_API_KEY",
        llm_timeout: int = 30,
    ):
        self.use_llm = use_llm
        self.llm_api_url = llm_api_url
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key
        self.llm_api_key_env = llm_api_key_env
        self.llm_timeout = llm_timeout

    def parse(self, observation: str) -> Dict[str, Any]:
        if self.use_llm:
            llm_updates = self._parse_with_llm(observation)
            if llm_updates is not None:
                return llm_updates

        text = observation or ""
        lowered = text.lower()

        facts = self._extract_facts(text, lowered)
        rules = self._extract_rules(text, lowered)
        failures = self._extract_failures(text, lowered)
        successes = self._extract_successes(text, lowered)
        preconditions = self._extract_preconditions(text, lowered)

        return {
            "facts": facts,
            "rules": rules,
            "failures": failures,
            "successes": successes,
            "preconditions": preconditions,
        }

    def _parse_with_llm(self, observation: str) -> Dict[str, Any] | None:
        api_key = self.llm_api_key or os.getenv(self.llm_api_key_env)
        if not api_key:
            return None

        prompt = (
            "Extract structured game-memory updates from this observation.\n"
            "Return strict JSON with keys: facts, rules, failures, successes, preconditions.\n"
            "Each value must be a JSON array of short strings. No extra keys.\n\n"
            f"Observation:\n{observation}\n"
        )
        payload = {
            "model": self.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 300,
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
            return self._normalize_updates(obj)
        except Exception:
            return None

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

    @staticmethod
    def _normalize_updates(obj: Dict[str, Any]) -> Dict[str, Any]:
        keys = ["facts", "rules", "failures", "successes", "preconditions"]
        normalized = {}
        for key in keys:
            value = obj.get(key, [])
            if not isinstance(value, list):
                value = [str(value)]
            normalized[key] = [str(x).strip() for x in value if str(x).strip()]
        return normalized

    def _extract_facts(self, text: str, lowered: str) -> List[str]:
        facts: List[str] = []

        location = re.search(
            r"(?:you are in|you are at|location:)\s+([^\n\.;]+)", lowered
        )
        if location:
            facts.append(f"location:{location.group(1).strip()}")

        if "inventory" in lowered or "carrying" in lowered:
            items = re.findall(r"\b(?:a|an|the)\s+([a-z][a-z\- ]{1,30})\b", lowered)
            for item in items[:8]:
                facts.append(f"item_seen:{item.strip()}")

        exits = re.findall(r"\b(north|south|east|west|up|down)\b", lowered)
        if exits:
            uniq_exits = sorted(set(exits))
            facts.append(f"exits:{','.join(uniq_exits)}")

        return facts

    def _extract_rules(self, _text: str, lowered: str) -> List[str]:
        rules: List[str] = []
        if "door is locked" in lowered:
            rules.append("locked_door_requires_unlock")
        if "can't go that way" in lowered or "cannot go" in lowered:
            rules.append("movement_blocked_invalid_direction")
        if "need" in lowered and "to" in lowered:
            rules.append("precondition_hint_present")
        return rules

    def _extract_failures(self, _text: str, lowered: str) -> List[str]:
        patterns = [
            "you can't",
            "cannot",
            "nothing happens",
            "not possible",
            "doesn't work",
            "invalid",
            "failed",
            "no effect",
        ]
        return [p for p in patterns if p in lowered]

    def _extract_successes(self, _text: str, lowered: str) -> List[str]:
        patterns = [
            "you open",
            "you unlock",
            "you take",
            "you pick up",
            "you win",
            "completed",
            "score",
        ]
        return [p for p in patterns if p in lowered]

    def _extract_preconditions(self, _text: str, lowered: str) -> List[str]:
        preconditions: List[str] = []
        match = re.search(r"you need (?:a|an|the)?\s*([^\n\.;]+)", lowered)
        if match:
            preconditions.append(match.group(1).strip())
        return preconditions
