from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .observation_parser import ObservationParser


@dataclass
class ScratchpadState:
    facts: List[str] = field(default_factory=list)
    rules: List[str] = field(default_factory=list)
    preconditions: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)
    successes: List[str] = field(default_factory=list)
    importance: Dict[str, float] = field(default_factory=dict)
    compressed_history: List[Dict[str, Any]] = field(default_factory=list)


class Scratchpad:
    """Holds evolving structured memory and importance weights."""

    def __init__(self, parser: Optional[ObservationParser] = None):
        self.parser = parser or ObservationParser()
        self.state = ScratchpadState()
        self.step = 0
        self.failure_counts = defaultdict(int)

    def reset(self) -> None:
        self.state = ScratchpadState()
        self.step = 0
        self.failure_counts.clear()

    def update(self, observation: str) -> Dict[str, Any]:
        self.step += 1
        updates = self.parser.parse(observation)
        self._apply_updates(updates)
        return updates

    def compress(self, compressor: Any) -> Dict[str, Any]:
        compressed_state = compressor.compress(self.state)
        self.state.compressed_history.append(compressed_state)
        self._load_from_compressed(compressed_state)
        return compressed_state

    def export(self) -> Dict[str, Any]:
        return {
            "facts": list(self.state.facts),
            "rules": list(self.state.rules),
            "preconditions": list(self.state.preconditions),
            "failures": list(self.state.failures),
            "successes": list(self.state.successes),
            "importance": dict(self.state.importance),
            "compressed_history": list(self.state.compressed_history),
        }

    def _apply_updates(self, updates: Dict[str, Any]) -> None:
        self._merge_unique(self.state.facts, updates.get("facts", []))
        self._merge_unique(self.state.rules, updates.get("rules", []))
        self._merge_unique(self.state.preconditions, updates.get("preconditions", []))
        self._merge_unique(self.state.successes, updates.get("successes", []))

        for failure in updates.get("failures", []):
            self.state.failures.append(failure)
            self.failure_counts[failure] += 1

        self._recompute_importance(updates)

    def _recompute_importance(self, updates: Dict[str, Any]) -> None:
        for fact in updates.get("facts", []):
            self.state.importance[fact] = self.state.importance.get(fact, 0.0) + 1.0

        for rule in updates.get("rules", []):
            self.state.importance[rule] = self.state.importance.get(rule, 0.0) + 1.5

        for precondition in updates.get("preconditions", []):
            key = f"need:{precondition}"
            self.state.importance[key] = self.state.importance.get(key, 0.0) + 2.0

        for failure, count in self.failure_counts.items():
            self.state.importance[f"failure:{failure}"] = 1.0 + float(count)

    def _load_from_compressed(self, compressed_state: Dict[str, Any]) -> None:
        top_facts = compressed_state.get("top_facts", [])
        top_rules = compressed_state.get("top_rules", [])
        self.state.facts = list(top_facts)
        self.state.rules = list(top_rules)

    @staticmethod
    def _merge_unique(target: List[str], additions: List[str]) -> None:
        seen = set(target)
        for item in additions:
            if item not in seen:
                target.append(item)
                seen.add(item)
