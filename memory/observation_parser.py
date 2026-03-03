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
        if re.search(r"\byou need\b.{0,80}\bto\b", lowered):
            rules.append("precondition_hint_present")
        return rules

    def _extract_failures(self, text: str, _lowered: str) -> List[str]:
        failures: List[str] = []
        failure_patterns = [
            ("action_rejected", r"^(you can't|you cannot)\b"),
            ("nothing_happened", r"^nothing happens\b"),
            ("not_possible", r"^(that's|this is) not possible\b"),
            ("doesnt_work", r"^that doesn't work\b"),
            ("invalid_command", r"^(invalid|i don't understand|unknown command)\b"),
            ("missing_required_item", r"^you (don't|do not) (have|carry)\b"),
            ("no_effect", r"^no effect\b"),
            ("action_failed", r"^failed\b"),
            ("burned_item", r"^you burned\b"),
            ("terminal_loss", r"^\*+\s*you lost!?"),
            ("terminal_loss", r"^would you like to quit\??\b"),
            ("terminal_loss", r"^game over\b"),
            ("terminal_loss", r"^you (died|are dead)\b"),
        ]

        for raw_line in text.splitlines():
            line = " ".join(raw_line.strip().lower().split())
            if not line:
                continue
            for label, pattern in failure_patterns:
                if re.search(pattern, line):
                    failures.append(label)
                    break

        return self._dedupe(failures)

    def _extract_successes(self, _text: str, lowered: str) -> List[str]:
        successes: List[str] = []
        success_patterns = [
            ("you open", r"^you open\b"),
            ("you unlock", r"^you unlock\b"),
            ("you take", r"^you (take|pick up)\b"),
            ("you win", r"^\*+\s*you won!?"),
            ("score_increased", r"^your score has just gone up\b"),
            ("completed", r"\b(completed|objective complete)\b"),
        ]

        for raw_line in lowered.splitlines():
            line = " ".join(raw_line.strip().split())
            if not line:
                continue
            for label, pattern in success_patterns:
                if re.search(pattern, line):
                    successes.append(label)
                    break

        return self._dedupe(successes)

    def _extract_preconditions(self, _text: str, lowered: str) -> List[str]:
        preconditions: List[str] = []

        # Capture explicit "need X to do Y" constraints.
        for need, goal in re.findall(
            r"you need\s+([a-z0-9\- ']{1,50}?)\s+to\s+([a-z0-9\- ']{1,60})",
            lowered,
        ):
            need_clean = " ".join(need.split())
            goal_clean = " ".join(goal.split())
            if need_clean and goal_clean:
                preconditions.append(f"{need_clean} -> {goal_clean}")

        # Capture "you need to VERB ..." while filtering navigational flavor text.
        for goal in re.findall(r"you need to\s+([a-z0-9\- ']{1,60})", lowered):
            goal_clean = " ".join(goal.split())
            if goal_clean and "exit without a door" not in goal_clean:
                preconditions.append(f"to {goal_clean}")

        return self._dedupe(preconditions)

    @staticmethod
    def _dedupe(items: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for item in items:
            key = item.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(item.strip())
        return out
