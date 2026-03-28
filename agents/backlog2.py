import argparse
import re
from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, List, Optional, Set, Tuple

import llm
import numpy as np
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

import tales
from tales.agent import register
from tales.token import get_token_counter
from tales.utils import (
    format_messages_to_markdown,
    is_recoverable_error,
    merge_messages,
    messages2conversation,
)


GOAL_MAKER_SYSTEM_PROMPT = (
    "You are a goal maker for a text-based game agent.\n"
    "Your job is to infer ONE long-term goal from the game's initial observation.\n\n"
    "Rules:\n"
    " - Use the game's opening instruction/objective if it is explicitly stated.\n"
    " - The long-term goal should be high-level and stable across the episode.\n"
    " - Do NOT produce a short-term actionable task like.\n"
    " - If there is a location mentioned in the opening observation that is likely to be important for the game, it can be included in the long-term goal.\n"
    " - Do NOT include explanations.\n"
    " - If the opening observation does not specify a clear final objective, output exactly: achieve as much score as possible\n\n"
    "Example:\n"
    "Opening observation: 'You are hungry! Let's cook a delicious meal. Check the cookbook in the kitchen for the recipe. Once done, enjoy your meal!'\n"
    "Output: Find the kitchen and prepare meal and then eat the meal to avoid hunger\n"
)

ACTION_RECOMMENDER_SYSTEM_PROMPT = (
    "You are an action recommender for a text-based game agent.\n"
    "At each step your job is to do TWO things:\n"
    " 1. judge the outcome of the previous action, if there was a previous action\n"
    " 2. recommend exactly ONE next action\n\n"
    "Outcome labels for the previous action:\n"
    " - useful_scoring: the previous action directly increased score or clearly completed a scoring milestone\n"
    " - useful_non_scoring: the previous action helped progress without increasing score (new room, new information, opened access, useful object)\n"
    " - useless: the previous action produced no useful progress\n"
    " - failed: the previous action directly caused failure, loss, death, or restart\n\n"
    "How to judge the previous action:\n"
    " - Judge only the immediately previous action explicitly provided in the prompt, not an entire task.\n"
    " - Use the latest observation/feedback as evidence.\n"
    " - If score increased after the previous action, prefer useful_scoring.\n"
    " - If the action revealed new information, opened a path, moved to a useful new room, or obtained a useful item, prefer useful_non_scoring.\n"
    " - Entering a new room that helps search for a goal-relevant location usually counts as useful_non_scoring.\n"
    " - If the action had no meaningful effect, or only repeated already-known information, prefer useless.\n"
    " - If the action directly triggered a loss, restart, or obviously bad irreversible result, prefer failed.\n\n"
    "Decision rules for the next action:\n"
    " - Use the long-term goal as the overall objective.\n"
    " - Use the map summary to navigate systematically.\n"
    " - Avoid repeating failed actions in the same relevant location/context unless there is strong new evidence.\n"
    " - Avoid repeating useless actions unless the observation has materially changed.\n"
    " - Previously useful actions are examples of good strategy, but do not repeat an already completed useful action in the current episode unless the observation indicates it is relevant again.\n"
    " - Prefer actions that help reach or exploit goal-relevant locations and objects.\n"
    " - If the map summary lists untried exits from the current room, prefer one of them unless a known path to a more relevant target is available.\n"
    " - If a target location is known to be important (for example, kitchen), prioritize movement actions that help locate or reach it.\n"
    " - Do not interact with irrelevant objects just because they are mentioned in the observation.\n"
    " - Opening a door is not a goal by itself; only do it when it enables progress toward the long-term goal.\n"
    " - Output exactly one executable in-game action.\n\n"
    "IMPORTANT RULES:\n"
    " - History may include restarts and previous episodes; use it as context, but the action to classify is always the explicitly provided previous action.\n"
    " - If there is NO previous action yet (for example, the very first step of an episode), DO NOT output <previous_action> or <previous_action_outcome> or <evidence> tags.\n"
    " - If there IS a previous action, you MUST output <previous_action>, <previous_action_outcome>, and <evidence>.\n"
    " - Always include current location in this exact tag format: <location> ... <location>.\n"
    " - <evidence> must be short (<= 20 words) and quote concrete signals from the latest observation/feedback.\n"
    " - Output ONLY the required tags. No extra explanation text.\n\n"
    "Previous action outcome examples:\n"
    "Example 1:\n"
    "Previous action: take block of cheese from fridge\n"
    "Latest observation: 'You take the block of cheese from the fridge.' Score increased by 1.\n"
    "Judgement: useful_scoring\n\n"
    "Example 2:\n"
    "Previous action: examine cookbook\n"
    "Latest observation: recipe is shown with ingredients and directions, but score did not increase.\n"
    "Judgement: useful_non_scoring\n\n"
    "Example 3:\n"
    "Previous action: go west\n"
    "Latest observation: a new room is entered.\n"
    "Judgement: useful_non_scoring\n\n"
    "Example 4:\n"
    "Previous action: examine BBQ\n"
    "Latest observation: no new useful information, no score increase, no new access.\n"
    "Judgement: useless\n\n"
    "Example 5:\n"
    "Previous action: eat block of cheese\n"
    "Latest observation: 'You eat the block of cheese. Not bad.' then '*** You lost! ***' and restart.\n"
    "Judgement: failed\n\n"
    "Output format when there IS a previous action:\n"
    "<previous_action> ... <previous_action>\n"
    "<previous_action_outcome> useful_scoring/useful_non_scoring/useless/failed <previous_action_outcome>\n"
    "<location> ... <location>\n"
    "<evidence> ... <evidence>\n"
    "<next_action> ... <next_action>\n\n"
    "Output example when there IS a previous action:\n"
    "<previous_action> examine cookbook <previous_action>\n"
    "<previous_action_outcome> useful_non_scoring <previous_action_outcome>\n"
    "<location> kitchen <location>\n"
    "<evidence> recipe shown with ingredients and directions <evidence>\n"
    "<next_action> open fridge <next_action>\n\n"
    "Output format when there is NO previous action yet:\n"
    "<location> ... <location>\n"
    "<next_action> ... <next_action>\n\n"
    "Output example when there is NO previous action yet:\n"
    "<location> shed <location>\n"
    "<next_action> open barn door <next_action>\n"
)

_ACTION_OUTCOME_LABELS = {
    "useful_scoring",
    "useful_non_scoring",
    "useless",
    "failed",
}

_CARDINAL_DIRECTIONS = ("north", "south", "east", "west", "up", "down")
_REVERSE_DIRECTION = {
    "north": "south",
    "south": "north",
    "east": "west",
    "west": "east",
    "up": "down",
    "down": "up",
}

_COMMAND_PREFIXES = (
    "go ",
    "open ",
    "close ",
    "take ",
    "drop ",
    "put ",
    "insert ",
    "eat ",
    "drink ",
    "examine ",
    "look",
    "inventory",
    "read ",
    "unlock ",
    "lock ",
    "turn on ",
    "turn off ",
    "cook ",
    "prepare meal",
    "north",
    "south",
    "east",
    "west",
    "up",
    "down",
)


@dataclass
class RecommendResult:
    previous_action: Optional[str]
    previous_action_outcome: Optional[str]
    current_location: Optional[str]
    evidence: Optional[str]
    next_action: str
    raw: str


def _extract_repeated_tag(text: str, tag: str) -> Optional[str]:
    open_tag = f"<{tag}>"
    close_tag = f"</{tag}>"

    if open_tag not in text:
        return None

    if close_tag in text:
        m = re.search(re.escape(open_tag) + r"\s*(.*?)\s*" + re.escape(close_tag), text, re.DOTALL)
        return m.group(1).strip() if m else None

    first = text.find(open_tag)
    last = text.rfind(open_tag)
    if first == -1 or last == -1 or last == first:
        return None
    return text[first + len(open_tag):last].strip()


def _normalize_outcome_label(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s2 = s.strip().lower()
    for tok in re.split(r"[\s,;/]+", s2):
        if tok in _ACTION_OUTCOME_LABELS:
            return tok
    if s2 in _ACTION_OUTCOME_LABELS:
        return s2
    return None


def parse_recommend_output(text: str) -> RecommendResult:
    prev_action = _extract_repeated_tag(text, "previous_action")
    outcome_raw = _extract_repeated_tag(text, "previous_action_outcome")
    outcome = _normalize_outcome_label(outcome_raw)
    location = _extract_repeated_tag(text, "location")
    evidence = _extract_repeated_tag(text, "evidence")
    next_action = _extract_repeated_tag(text, "next_action")

    if next_action is None:
        lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
        candidates = []
        for ln in lines:
            low = ln.lower()
            if low.startswith("<previous_action") or low.startswith("</previous_action"):
                continue
            if low.startswith("<previous_action_outcome") or low.startswith("</previous_action_outcome"):
                continue
            if low.startswith("<location") or low.startswith("</location"):
                continue
            if low.startswith("<evidence") or low.startswith("</evidence"):
                continue
            if low.startswith("<next_action") or low.startswith("</next_action"):
                continue
            candidates.append(ln)
        command_like = [ln for ln in candidates if ln.lower().startswith(_COMMAND_PREFIXES)]
        next_action = command_like[0] if command_like else (candidates[0] if candidates else "")

    if next_action:
        next_action = re.sub(r"</?next_action>", "", next_action, flags=re.IGNORECASE).strip()

    return RecommendResult(
        previous_action=(prev_action or "").strip() or None,
        previous_action_outcome=outcome,
        current_location=(location or "").strip() or None,
        evidence=(evidence or "").strip() or None,
        next_action=(next_action or "").strip(),
        raw=text,
    )


class SimplePlanningAgent(tales.Agent):
    """
    Single-stage LLM agent:
      1) one-time goal maker
      2) per-step action recommender that classifies previous action and suggests next action
    """

    def __init__(self, *args, **kwargs):
        self.llm = kwargs["llm"]
        self.model = llm.get_model(self.llm)
        self.token_counter = get_token_counter(self.model)

        self.model.key = llm.get_key(
            kwargs.get("key"), kwargs["llm"], self.model.key_env_var
        ) or llm.get_key(None, self.model.needs_key, self.model.key_env_var)

        self.seed = kwargs["seed"]
        self.rng = np.random.RandomState(self.seed)

        self.history: List[Tuple[int, int, str, str]] = []
        self.episode_idx: int = 0

        self.context_limit = kwargs.get("context_limit")
        if self.context_limit is not None:
            assert self.context_limit > 0, "--context-limit must be greater than 0."

        self.conversation = kwargs["conversation"]
        self.recommend_temp = kwargs.get("recommend_temp", 0.0)

        self.long_term_goal: str = "achieve as much score as possible"
        self.current_location: Optional[str] = None

        self.map_graph: Dict[str, Dict] = {}
        self.last_action: Optional[str] = None
        self.last_location_before_action: Optional[str] = None
        self.last_action_episode: int = 0

        # Pending previous action to be judged on the next call.
        self.previous_action: Optional[str] = None
        self.previous_action_location: Optional[str] = None
        self.previous_action_episode: Optional[int] = None
        self.previous_action_step: Optional[int] = None

        # Deduplicated action memory keyed by (normalized action, normalized location).
        self.useful_scoring_actions: Dict[Tuple[str, Optional[str]], dict] = {}
        self.useful_non_scoring_actions: Dict[Tuple[str, Optional[str]], dict] = {}
        self.useless_actions: Dict[Tuple[str, Optional[str]], dict] = {}
        self.failed_actions: Dict[Tuple[str, Optional[str]], dict] = {}
        self.completed_current_episode_actions: Set[str] = set()

        # Global step counter; do not reset across restarts.
        self.step_idx: int = 0

    @property
    def uid(self):
        return (
            f"ActionRecommenderAgent_{self.llm}"
            f"_s{self.seed}"
            f"_c{self.context_limit}"
            f"_rt{self.recommend_temp}"
            f"_conv{self.conversation}"
        )

    @property
    def params(self):
        return {
            "agent_type": "action-recommender",
            "llm": self.llm,
            "seed": self.seed,
            "context_limit": self.context_limit,
            "recommend_temp": self.recommend_temp,
            "conversation": self.conversation,
        }

    @retry(
        retry=retry_if_exception(is_recoverable_error),
        wait=wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(100),
    )
    def _llm_call_from_conversation(self, conversation, *args, **kwargs):
        response = conversation.prompt(*args, **kwargs)
        response.duration_ms()
        return response

    def _llm_call_from_messages(self, messages, *args, **kwargs):
        allows_system_prompt = self.llm not in ["o1-mini", "o1-preview"]
        system = messages[0]["content"] if (allows_system_prompt and messages and messages[0]["role"] == "system") else None

        if not allows_system_prompt and messages and messages[0]["role"] == "system":
            messages = [dict(m) for m in messages]
            sys_msg = messages.pop(0)["content"]
            for i, msg in enumerate(messages):
                if msg["role"] == "user":
                    messages[i]["content"] = f"{sys_msg}\n\n{messages[i]['content']}"
                    break

        conversation = messages2conversation(self.model, messages)
        prompt = messages[-1]["content"]
        return self._llm_call_from_conversation(conversation, prompt=prompt, system=system, *args, **kwargs)

    def _build_llm_kwargs(self, *, temperature: float, max_tokens: int) -> dict:
        kwargs = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": self.seed,
            "stream": False,
        }
        if self.llm in [
            "claude-3.5-haiku",
            "claude-3.5-sonnet",
            "claude-3.5-sonnet-latest",
            "claude-3.7-sonnet",
        ]:
            kwargs.pop("seed", None)
        if "gemini" in self.llm or "gemma" in self.llm:
            kwargs.pop("seed", None)
            kwargs["max_output_tokens"] = kwargs.pop("max_tokens")
        return kwargs

    def reset(self, obs, info, env_name=None):
        self.episode_idx += 1
        self.completed_current_episode_actions = set()

        self.last_action = None
        self.last_location_before_action = None
        self.last_action_episode = 0

        # Keep pending previous action across reset so the next recommender call can judge it.
        self._reset_episode_local_map_state()

        self.current_location = self._infer_location_from_observation(obs)
        if self.current_location:
            self._ensure_room_node(self.current_location)
            self.map_graph[self.current_location]["visited_count"] += 1

    def _reset_episode_local_map_state(self):
        for room, node in self.map_graph.items():
            node["visited_count"] = 0
            node["objects"] = []
            for _, meta in node.get("exits", {}).items():
                meta["tried"] = False
                meta["mentioned"] = False
                if meta.get("target"):
                    meta["status"] = "open"
                elif meta.get("status") == "closed":
                    meta["status"] = "unknown"

    def _sanitize_observation_text(self, observation: str) -> str:
        text = (observation or "").strip("\n")
        if not text:
            return ""

        lines = text.splitlines()

        def is_room_header(s: str) -> bool:
            return bool(re.match(r"^\s*-\=\s*[^=\n]+?\s*\=-\s*$", s))

        def is_decorative_line(s: str) -> bool:
            stripped = s.strip()
            if not stripped:
                return True
            alpha = sum(ch.isalpha() for ch in stripped)
            digit = sum(ch.isdigit() for ch in stripped)
            space = sum(ch.isspace() for ch in stripped)
            symbol = len(stripped) - alpha - digit - space
            if len(stripped) >= 3 and alpha == 0 and digit == 0:
                return True
            if alpha > 0 and symbol > alpha * 1.5:
                return True
            lower = sum(ch.islower() for ch in stripped)
            upper = sum(ch.isupper() for ch in stripped)
            if alpha >= 6 and lower == 0 and upper >= 4 and symbol >= 2:
                return True
            return False

        def is_meaningful_text(s: str) -> bool:
            stripped = s.strip()
            if not stripped:
                return False
            if is_room_header(stripped):
                return True
            alpha = sum(ch.isalpha() for ch in stripped)
            lower = sum(ch.islower() for ch in stripped)
            words = re.findall(r"[A-Za-z]+", stripped)
            symbol = sum((not ch.isalnum()) and (not ch.isspace()) for ch in stripped)
            return alpha >= 3 and lower >= 1 and len(words) >= 2 and symbol <= max(alpha, 6)

        start_idx = None
        for i, line in enumerate(lines):
            s = line.strip()
            if not s:
                continue
            if is_meaningful_text(s):
                start_idx = i
                break
            if is_decorative_line(s):
                continue
            start_idx = i
            break

        if start_idx is None:
            return text.strip()

        cleaned = "\n".join(lines[start_idx:]).strip()
        cleaned = re.sub(r"^\n+", "", cleaned)
        return cleaned

    def _format_history(self, limit_turns: int, items=None) -> str:
        items = list(items) if items is not None else (self.history[-limit_turns:] if limit_turns > 0 else self.history)
        lines = []
        for ep, step, obs, act in items:
            o = obs.replace("\n", "\\n")
            a = act.replace("\n", "\\n")
            lines.append(f"episode {ep}-step {step} | observation: {o} | action: {a}")
        return "\n".join(lines) if lines else "(empty)"

    def _infer_location_from_observation(self, observation: str) -> Optional[str]:
        text = observation or ""
        matches = re.findall(r"-=\s*([^=\n]+?)\s*=-", text)
        for m in reversed(matches):
            loc = m.strip()
            if loc and loc.lower() != "restarting":
                return loc
        return None

    def _infer_target_room_hint(self) -> Optional[str]:
        goal = (self.long_term_goal or "").lower()
        for room in ["kitchen", "bedroom", "bathroom", "livingroom", "living room", "corridor", "backyard", "garden", "shed", "pantry"]:
            if room in goal:
                return room.replace(" ", "")
        return None

    def _ensure_room_node(self, room: str) -> Dict:
        room = (room or "").strip()
        if not room:
            return {}
        if room not in self.map_graph:
            self.map_graph[room] = {
                "exits": {
                    d: {"status": "unknown", "target": None, "tried": False, "mentioned": False}
                    for d in _CARDINAL_DIRECTIONS
                },
                "visited_count": 0,
                "objects": [],
            }
        else:
            exits = self.map_graph[room].setdefault("exits", {})
            for d in _CARDINAL_DIRECTIONS:
                meta = exits.setdefault(d, {"status": "unknown", "target": None, "tried": False, "mentioned": False})
                meta.setdefault("tried", False)
                meta.setdefault("mentioned", False)
        return self.map_graph[room]

    def _extract_move_direction(self, action: Optional[str]) -> Optional[str]:
        a = (action or "").strip().lower()
        if not a:
            return None
        if a in _CARDINAL_DIRECTIONS:
            return a
        m = re.fullmatch(r"go\s+(north|south|east|west|up|down)", a)
        return m.group(1) if m else None

    def _is_movement_blocked(self, text: str) -> bool:
        t = (text or "").lower()
        markers = [
            "you can't go that way",
            "can't go that way",
            "there is no exit",
            "you cannot go",
            "blocked",
            "closed",
            "not a direction",
            "you can't",
        ]
        return any(m in t for m in markers)

    def _extract_objects_from_observation(self, observation: str, admissible: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        text = observation or ""
        patterns = [
            r"\bYou can see ([^.]+)\.",
            r"\bYou see ([^.]+)\.",
            r"\bon the [^.]+ you (?:make out|can see|see) ([^.]+)\.",
            r"\bin the [^.]+ you (?:find|can see|see) ([^.]+)\.",
        ]
        for pat in patterns:
            for m in re.finditer(pat, text, flags=re.IGNORECASE):
                phrase = m.group(1).strip()
                parts = re.split(r",| and ", phrase)
                for p in parts:
                    obj = re.sub(r"^(a|an|the)\s+", "", p.strip(), flags=re.IGNORECASE)
                    obj = obj.strip(" .,:;")
                    if not obj:
                        continue
                    low = obj.lower()
                    if low in {"nothing", "anything", "piece of junk"}:
                        continue
                    if low not in seen:
                        seen.add(low)
                        out.append(obj)
        for cmd in admissible or []:
            c = cmd.strip().lower()
            m = re.match(r"^(?:examine|open|close|take|read|unlock|lock|cook|slice|dice|chop|fry|roast|grill)\s+(.+)$", c)
            if not m:
                continue
            obj = m.group(1)
            obj = re.split(r"\s+(?:with|using|in|into|on|from)\s+", obj)[0].strip()
            if not obj or obj in {"inventory", "look", "help"}:
                continue
            if obj in _CARDINAL_DIRECTIONS:
                continue
            if obj not in seen:
                seen.add(obj)
                out.append(obj)
        return out[:20]

    def _extract_explicit_exits_from_observation(self, observation: str) -> Set[str]:
        text = observation or ""
        found: Set[str] = set()
        for m in re.finditer(r"\bexits?\s*[:\-]\s*([^\n.]+)", text, flags=re.IGNORECASE):
            chunk = m.group(1).lower()
            for d in _CARDINAL_DIRECTIONS:
                if re.search(rf"\b{d}\b", chunk):
                    found.add(d)
        for d in _CARDINAL_DIRECTIONS:
            patterns = [
                rf"\byou can go {d}\b",
                rf"\ban? (?:open |closed )?(?:exit|door|passage|path)\b.*\b{d}\b",
                rf"\bleading {d}\b",
                rf"\btry going {d}\b",
                rf"\bexit to the {d}\b",
                rf"\bgoing {d}\b",
            ]
            if any(re.search(p, text, flags=re.IGNORECASE) for p in patterns):
                found.add(d)
        return found

    def _update_map_from_observation(self, observation: str, info: dict):
        room = self.current_location
        if not room:
            return
        node = self._ensure_room_node(room)
        if not node:
            return
        node["visited_count"] += 1
        admissible = info.get("admissible_commands") or []
        explicit_exits = self._extract_explicit_exits_from_observation(observation)
        for d in explicit_exits:
            node["exits"][d]["mentioned"] = True
            if node["exits"][d]["status"] == "unknown":
                node["exits"][d]["status"] = "open"
        for obj in self._extract_objects_from_observation(observation, admissible):
            if obj not in node["objects"]:
                node["objects"].append(obj)

    def _resolve_last_move_transition(self, observation: str):
        if self.last_action_episode != self.episode_idx:
            return
        move_dir = self._extract_move_direction(self.last_action)
        if not move_dir or not self.last_location_before_action:
            return
        from_room = self.last_location_before_action
        from_node = self._ensure_room_node(from_room)
        if not from_node:
            return
        from_node["exits"][move_dir]["mentioned"] = True
        from_node["exits"][move_dir]["tried"] = True

        to_room = self.current_location
        if to_room and to_room != from_room:
            from_node["exits"][move_dir]["status"] = "open"
            from_node["exits"][move_dir]["target"] = to_room
            to_node = self._ensure_room_node(to_room)
            rev = _REVERSE_DIRECTION.get(move_dir)
            if to_node and rev:
                to_node["exits"][rev]["status"] = "open"
                to_node["exits"][rev]["mentioned"] = True
                if to_node["exits"][rev]["target"] is None:
                    to_node["exits"][rev]["target"] = from_room
            return

        if self._is_movement_blocked(observation):
            from_node["exits"][move_dir]["status"] = "closed"
            if from_node["exits"][move_dir]["target"] == from_room:
                from_node["exits"][move_dir]["target"] = None

    def _find_room_by_keyword(self, keyword: str) -> Optional[str]:
        kw = (keyword or "").strip().lower()
        if not kw:
            return None
        for room in self.map_graph:
            if kw == room.lower() or kw in room.lower():
                return room
        return None

    def _shortest_path_directions(self, src_room: str, dst_room: str) -> Optional[List[str]]:
        if src_room not in self.map_graph or dst_room not in self.map_graph:
            return None
        q: Deque[Tuple[str, List[str]]] = deque([(src_room, [])])
        seen = {src_room}
        while q:
            room, path = q.popleft()
            if room == dst_room:
                return path
            node = self.map_graph.get(room, {})
            for d, meta in (node.get("exits", {}) or {}).items():
                target = meta.get("target")
                status = meta.get("status")
                if status != "open" or not target or target in seen:
                    continue
                seen.add(target)
                q.append((target, path + [d]))
        return None

    def _make_action_key(self, action: str, location: Optional[str]) -> Tuple[str, Optional[str]]:
        return (
            (action or "").strip().lower(),
            ((location or "").strip().lower() or None),
        )

    def _upsert_action_memory(self, bucket: Dict[Tuple[str, Optional[str]], dict], action: str, location: Optional[str]):
        key = self._make_action_key(action, location)
        bucket[key] = {
            "action": (action or "").strip(),
            "location": (location or "").strip() or None,
            "last_episode": self.episode_idx,
        }

    def _format_completed_current_episode_actions(self) -> str:
        if not self.completed_current_episode_actions:
            return "- (none)"
        return "\n".join(f"- {a}" for a in sorted(self.completed_current_episode_actions))

    def _build_map_summary(self, max_rooms: int = 8) -> str:
        if not self.map_graph:
            return "(empty)"
        lines: List[str] = []
        cur = self.current_location
        if cur and cur in self.map_graph:
            cur_node = self.map_graph[cur]
            untried_exits = [
                d for d, meta in cur_node["exits"].items()
                if meta.get("mentioned", False) and meta.get("status") != "closed" and not meta.get("tried", False)
            ]
            tried_exits = [d for d, meta in cur_node["exits"].items() if meta.get("tried", False)]
            lines.append(f"Current room: {cur}")
            lines.append("Untried exits from current room: " + (", ".join(untried_exits) if untried_exits else "(none)"))
            lines.append("Tried exits from current room: " + (", ".join(tried_exits) if tried_exits else "(none)"))

        target_room_hint = self._infer_target_room_hint()
        target_room = self._find_room_by_keyword(target_room_hint) if target_room_hint else None
        if cur and target_room:
            if cur == target_room:
                lines.append(f"Known path to {target_room}: already here.")
            else:
                path = self._shortest_path_directions(cur, target_room)
                if path:
                    lines.append(f"Known path to {target_room}: " + " -> ".join([f"go {d}" for d in path]))
                else:
                    lines.append(f"Known path to {target_room}: unknown.")

        rooms = sorted(
            self.map_graph.items(),
            key=lambda kv: ((kv[0] != (cur or "")), -kv[1].get("visited_count", 0), kv[0]),
        )[:max_rooms]
        for room_name, node in rooms:
            exits = []
            for d in _CARDINAL_DIRECTIONS:
                meta = node["exits"][d]
                status = meta.get("status", "unknown")
                target = meta.get("target")
                tried = bool(meta.get("tried", False))
                mentioned = bool(meta.get("mentioned", False))
                if status == "unknown" and not target and not tried and not mentioned:
                    continue
                tried_tag = "tried" if tried else "untried"
                if target:
                    exits.append(f"{d}:{status}/{tried_tag}->{target}")
                else:
                    exits.append(f"{d}:{status}/{tried_tag}")
            objs = ", ".join(node.get("objects", [])[:6]) if node.get("objects") else "(none)"
            exits_str = ", ".join(exits) if exits else "(none known)"
            lines.append(f"{room_name} (visited {node.get('visited_count', 0)}) | exits: {exits_str} | objects: {objs}")
        return "\n".join([f"- {ln}" for ln in lines]) if lines else "(empty)"

    def _render_action_records(self, records: Dict[Tuple[str, Optional[str]], dict], limit: int = 20) -> str:
        if not records:
            return "- (none)"
        items = sorted(
            records.values(),
            key=lambda x: ((x["location"] or ""), x["action"].lower()),
        )[:limit]
        lines = []
        for rec in items:
            loc = rec["location"] or "(unknown)"
            lines.append(f"- {rec['action']} (last episode {rec['last_episode']}, {loc})")
        return "\n".join(lines)

    def _store_previous_action_outcome(self, outcome: Optional[str]):
        if not self.previous_action or outcome not in _ACTION_OUTCOME_LABELS:
            return
        if outcome == "useful_scoring":
            self._upsert_action_memory(self.useful_scoring_actions, self.previous_action, self.previous_action_location)
            self.completed_current_episode_actions.add(self.previous_action)
        elif outcome == "useful_non_scoring":
            self._upsert_action_memory(self.useful_non_scoring_actions, self.previous_action, self.previous_action_location)
            self.completed_current_episode_actions.add(self.previous_action)
        elif outcome == "useless":
            self._upsert_action_memory(self.useless_actions, self.previous_action, self.previous_action_location)
        elif outcome == "failed":
            self._upsert_action_memory(self.failed_actions, self.previous_action, self.previous_action_location)

    def build_goal_messages(self, observation: str, info: dict) -> List[dict]:
        messages = [{"role": "system", "content": GOAL_MAKER_SYSTEM_PROMPT}]
        prompt = (
            f"Initial observation:\n{self._sanitize_observation_text(observation)}\n\n"
            f"Output the single long-term goal only.\n"
        )
        messages.append({"role": "user", "content": prompt})
        messages = merge_messages(messages)
        if not self.conversation:
            content = "".join([msg["content"] for msg in messages[1:]])
            messages = messages[:1] + [{"role": "user", "content": content}]
        return messages

    def build_recommend_messages(self, observation: str, info: dict) -> List[dict]:
        messages = [{"role": "system", "content": ACTION_RECOMMENDER_SYSTEM_PROMPT}]

        limit_turns = self.context_limit or (len(self.history) + 1)
        history_str = self._format_history(limit_turns)
        map_summary = self._build_map_summary()
        completed_actions = self._format_completed_current_episode_actions()
        admissible = info.get("admissible_commands") or []
        admissible_block = ""
        if admissible:
            admissible_block = "Admissible commands (subset):\n- " + "\n- ".join(admissible[:50]) + "\n\n"

        if self.previous_action is None:
            previous_action_block = (
                "Previous action to judge: (none)\n"
                "There is no previous action to classify yet.\n"
                "Since there is no previous action yet, do not output <previous_action>, <previous_action_outcome>, or <evidence> tags.\n\n"
            )
        else:
            previous_action_block = (
                f"Previous action to judge: {self.previous_action}\n"
                f"Previous action episode: {self.previous_action_episode}\n"
                f"Previous action step: {self.previous_action_step}\n"
                f"Previous action location: {self.previous_action_location or '(unknown)'}\n\n"
            )

        prompt = (
            f"Long-term goal: {self.long_term_goal}\n"
            f"Current Episode: {self.episode_idx}\n"
            f"Global Step: {self.step_idx + 1}\n"
            f"Current known location: {self.current_location or '(unknown)'}\n"
            f"{previous_action_block}"
            f"Map summary:\n{map_summary}\n\n"
            f"Completed useful actions in current episode:\n{completed_actions}\n\n"
            f"Previously useful scoring actions:\n{self._render_action_records(self.useful_scoring_actions)}\n\n"
            f"Previously useful non-scoring actions:\n{self._render_action_records(self.useful_non_scoring_actions)}\n\n"
            f"Previously useless actions:\n{self._render_action_records(self.useless_actions)}\n\n"
            f"Previously failed actions:\n{self._render_action_records(self.failed_actions)}\n\n"
            f"{admissible_block}"
            f"Recent history:\n{history_str}\n\n"
            f"Current observation:\n{self._sanitize_observation_text(observation)}\n\n"
            f"Remember: output only the required tags. If Previous action to judge is (none), output only <location> and <next_action>.\n"
        )

        messages.append({"role": "user", "content": prompt})
        messages = merge_messages(messages)
        if not self.conversation:
            content = "".join([msg["content"] for msg in messages[1:]])
            messages = messages[:1] + [{"role": "user", "content": content}]
        return messages

    def act(self, obs, reward, done, infos):
        observed_location = self._infer_location_from_observation(obs)
        if observed_location:
            self.current_location = observed_location
        self._resolve_last_move_transition(obs)
        self._update_map_from_observation(obs, infos)

        goal_messages = None
        goal_resp = None
        if self.episode_idx == 1 and self.step_idx == 0:
            goal_messages = self.build_goal_messages(obs, infos)
            goal_resp = self._llm_call_from_messages(goal_messages, **self._build_llm_kwargs(temperature=0.0, max_tokens=60))
            goal_text = goal_resp.text().strip()
            self.long_term_goal = goal_text.splitlines()[0].strip() if goal_text else "achieve as much score as possible"

        recommend_messages = self.build_recommend_messages(obs, infos)
        recommend_resp = self._llm_call_from_messages(
            recommend_messages,
            **self._build_llm_kwargs(temperature=self.recommend_temp, max_tokens=260),
        )
        recommend_text = recommend_resp.text().strip()
        recommend_result = parse_recommend_output(recommend_text)

        if recommend_result.current_location:
            self.current_location = recommend_result.current_location.strip()

        # Judge and store only the explicitly pending previous action. Then clear it.
        self._store_previous_action_outcome(recommend_result.previous_action_outcome)
        if self.previous_action is not None and recommend_result.previous_action_outcome is not None:
            self.previous_action = None
            self.previous_action_location = None
            self.previous_action_episode = None
            self.previous_action_step = None

        action = recommend_result.next_action.strip() if recommend_result.next_action else ""
        if not action:
            action = "help"

        global_step = self.step_idx + 1
        self.history.append((self.episode_idx, global_step, f"{obs}\n> ", f"{action}\n"))
        self.last_action = action
        self.last_location_before_action = self.current_location
        self.last_action_episode = self.episode_idx

        # New pending action to be judged on the next act() call, even across reset.
        self.previous_action = action
        self.previous_action_location = self.current_location
        self.previous_action_episode = self.episode_idx
        self.previous_action_step = global_step

        self.step_idx += 1

        stats = {
            "long_term_goal": self.long_term_goal,
            "goal_response": goal_resp.text() if goal_resp is not None else None,
            "nb_tokens_goal_prompt": self.token_counter(messages=goal_messages) if goal_messages is not None else 0,
            "nb_tokens_goal_response": self.token_counter(text=goal_resp.text()) if goal_resp is not None else 0,
            "recommend_prompt": format_messages_to_markdown(recommend_messages),
            "recommend_response": recommend_resp.text(),
            "previous_action": recommend_result.previous_action,
            "previous_action_outcome": recommend_result.previous_action_outcome,
            "recommend_evidence": recommend_result.evidence,
            "next_action": action,
            "nb_tokens_recommend_prompt": self.token_counter(messages=recommend_messages),
            "nb_tokens_recommend_response": self.token_counter(text=recommend_resp.text()),
            "current_location": self.current_location,
            "map_summary": self._build_map_summary(),
            "known_rooms": len(self.map_graph),
            "completed_current_episode_actions": sorted(self.completed_current_episode_actions),
            "useful_scoring_actions": list(self.useful_scoring_actions.values()),
            "useful_non_scoring_actions": list(self.useful_non_scoring_actions.values()),
            "useless_actions": list(self.useless_actions.values()),
            "failed_actions": list(self.failed_actions.values()),
            "pending_previous_action": self.previous_action,
            "pending_previous_action_location": self.previous_action_location,
            "pending_previous_action_episode": self.previous_action_episode,
            "pending_previous_action_step": self.previous_action_step,
        }
        stats["nb_tokens_recommend"] = stats["nb_tokens_recommend_prompt"] + stats["nb_tokens_recommend_response"]
        stats["nb_tokens_goal"] = stats["nb_tokens_goal_prompt"] + stats["nb_tokens_goal_response"]
        stats["nb_tokens"] = stats["nb_tokens_recommend"] + stats["nb_tokens_goal"]
        return action, stats


def build_argparser(parser=None):
    parser = parser or argparse.ArgumentParser()
    group = parser.add_argument_group("Action recommender agent settings")

    group.add_argument(
        "--llm",
        default="gpt-4o-mini",
        help="LLM to be used for evaluation. Default: %(default)s",
    )
    group.add_argument(
        "--seed",
        type=int,
        default=20241001,
        help="Seed for LLM (not all endpoints support this). Default: %(default)s",
    )
    group.add_argument(
        "--recommend-temp",
        type=float,
        default=0.0,
        help="Temperature for recommender stage. Default: %(default)s",
    )
    group.add_argument(
        "--context-limit",
        type=int,
        help="Limit context for LLM (in conversation turns). Default: no limit.",
    )
    group.add_argument(
        "--conversation",
        required=True,
        action=argparse.BooleanOptionalAction,
        help="Enable conversation mode. Otherwise, use single prompt.",
    )
    return parser


register(
    name="backlog2",
    desc="Single-stage action recommender with long-term goal and map memory.",
    klass=SimplePlanningAgent,
    add_arguments=build_argparser,
)
