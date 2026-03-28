import re
from typing import Any, Dict, List


class Retriever:
    """Simple lexical retriever over compressed memory entries."""

    def __init__(self, top_k: int = 3):
        self.top_k = top_k

    def query(self, observation: str, compressed_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not compressed_history:
            return []

        obs_tokens = self._tokenize(observation)
        scored = []

        for item in compressed_history:
            memory_text = self._to_text(item)
            mem_tokens = self._tokenize(memory_text)
            overlap = len(obs_tokens & mem_tokens)
            union = len(obs_tokens | mem_tokens) or 1
            score = overlap / union
            scored.append((score, item))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for score, item in scored[: self.top_k] if score > 0]

    @staticmethod
    def _to_text(item: Dict[str, Any]) -> str:
        chunks = []
        for value in item.values():
            if isinstance(value, list):
                chunks.extend(str(x) for x in value)
            else:
                chunks.append(str(value))
        return " ".join(chunks)

    @staticmethod
    def _tokenize(text: Any) -> set:
        return set(re.findall(r"[a-z0-9_]+", str(text).lower()))
