import argparse
import re
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

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

# -----------------------------
# Prompts
# -----------------------------

GOAL_MAKER_SYSTEM_PROMPT = (
    "You are a goal maker for a text-based game agent.\n"
    "Your job is to infer ONE long-term goal from the game's initial observation.\n\n"
    "Rules:\n"
    " - Use the game's opening instruction/objective if it is explicitly stated.\n"
    " - The long-term goal should be high-level and stable across the episode.\n"
    " - Do NOT produce a short-term actionable task like 'go to kitchen'.\n"
    " - Do NOT include explanations.\n"
    " - If the opening observation does not specify a clear final objective, output exactly:\n"
    "   achieve as much score as possible\n\n"
    "Example:\n"
    "Opening observation: 'You are hungry! Let's cook a delicious meal. Check the cookbook in the kitchen for the recipe. Once done, enjoy your meal!'\n"
    "Output: Find the kitchen and prepare meal and then eat the meal to avoid hunger\n"
)

PLANNER_SYSTEM_PROMPT = (
    "You are a planner for a text-based game agent.\n"
    "Your ONLY job is to maintain a task backlog:\n"
    " - judge the state of the active task before planning\n"
    " - only switch to a different active task when the old one is marked completed_and_useful/completed_but_useless/failed/unprogressed; otherwise, keep it unchanged\n"
    " - a small list of candidate tasks\n\n"
    "Task requirements:\n"
    " - Short-term and actionable (can be tested with a few moves)\n"
    " - Verifiable from observations/feedback\n"
    " - Candidate tasks must directly help progress the long-term goal\n"
    " - Filter out irrelevant, redundant, or curiosity-only actions that do not help the long-term goal\n"
    " - Prefer the next short-horizon bottleneck step toward the long-term goal\n\n"
    "You must judge the active task state as one of:\n"
    " - progressing: evidence of progress but not finished)\n"
    " - completed_and_useful: finished and helped game progress (score/new information/new access/item)\n"
    " - completed_but_useless: finished but did not help progress or score\n"
    " - failed: caused loss/penalty or clearly wrong; OR the game restarted due to failure\n"
    " - unprogressed: no progress for ~10 steps AND clear evidence it's not promising\n\n"
    "State Judgement examples (Task / Action / Observation -> State):\n"
    " - Task: examine the cookbook on the table\n"
    "   Action: read cookbook\n"
    "   Observation: recipe is shown (ingredients + directions)\n"
    "   -> completed_and_useful (new information for planning)\n"
    " - Task: take the block of cheese\n"
    "   Action: take cheese\n"
    "   Observation: 'You take the block of cheese' + score increases\n"
    "   -> completed_and_useful\n"
    " - Task: finish meal with a risky step\n"
    "   Action: cook cheese with stove\n"
    "   Observation: 'You burned the block of cheese!' then game loss/restart\n"
    "   -> failed\n"
    " - Task: explore the environment generally\n"
    "   Actions over many steps: look/go east/go west/help with no new info and no score change\n"
    "   Observation trend: repeated dead-ends, no progress signal\n"
    "   -> unprogressed\n\n"
    "Navigation and exploration rules:\n"
    " - If the long-term goal or current observation explicitly mentions a target location (for example, kitchen), prioritize reaching that location before unrelated interactions.\n"
    " - Before reaching the target location, avoid spending actions on irrelevant objects, containers, or furniture unless they are required to unlock movement.\n"
    " - When the target location has not been found yet, candidate tasks should focus on navigation and systematic exploration.\n"
    " - Prefer unexplored exits and directions over re-checking already explored areas.\n"
    " - Opening a door is useful only if it helps access an unexplored or goal-relevant path.\n\n"
    "IMPORTANT RULES:\n"
    " - For planning future tasks, you may use the full history.\n"
    " - If task starts with find/locate/open/read/examine and observation confirms target discovered/opened/read, mark task as completed_* in this step.\n"
    " - Judge <active_task_state> using ONLY trajectories from Active Task Start Step to now.\n"
    " - Mark <active_task_state>=failed ONLY if the failure evidence is directly caused by attempting <active_task_before>.\n"
    " - If the failure came from a different action/task, DO NOT mark failed for <active_task_before>; use progressing or completed_but_useless.\n"
    " - Change active task when <active_task_state> is completed_and_useful/completed_but_useless/failed/unprogressed.\n"
    " - If <active_task_state>=progressing, <active_task_after> must be EXACTLY the same string as <active_task_before>.\n"
    " - Always include one short evidence line from latest observation/feedback for the state decision.\n"
    " - <evidence> must be <= 20 words and quote concrete signals about how you judge <active_task_before> (e.g., 'score +1', 'You can't go that way.').\n"
    " - Always include current location in this exact tag format: <location> ... <location>.\n"
    " - Every <candidate task> must be relevant to and advance the long-term goal.\n"
    " - Output ONLY required tags; no extra explanation text.\n"
    " - When changing active task, pick one of the candidate tasks if available. Otherwise, come up with a new task based on the observation.\n"
    "Output must follow the exact format:\n"
    "<active_task_before> ... <active_task_before>\n"
    "<active_task_state> progressing/completed_and_useful/completed_but_useless/failed/unprogressed <active_task_state>\n"
    "<location> ... <location>\n"
    "<evidence> ... <evidence>\n"
    "<active_task_after> ... <active_task_after>\n"
    "<candidate task> ... <candidate task>\n"
    "<candidate task> ... <candidate task>\n"
    "(repeat <candidate task> for each candidate task)\n"
    "Output Example (assuming active task before planning is 'find and take the block of cheese'):\n"
    "<active_task_before> find and take the block of cheese <active_task_before>\n"
    "<active_task_state> completed_and_useful <active_task_state>\n"
    "<location> kitchen <location>\n"
    "<evidence> 'You take the block of cheese.' and score increased by 1 <evidence>\n"
    "<active_task_after> find cookbook <active_task_after>\n"
    "<candidate task> examine cookbook <candidate task>\n"
    "<candidate task> some direction mentioned in cookbook <candidate task>\n"

)

ACTOR_SYSTEM_PROMPT = (
    "You are the action executor for a text-based game agent.\n"
    "You must choose exactly ONE next action based on the current observation, history, current task, and long-term goal.\n\n"

    "Decision hierarchy:\n"
    " - Long-term goal is the overall objective for the episode.\n"
    " - Current task is the immediate short-term subgoal chosen to advance the long-term goal.\n"
    " - Your action must directly advance the current task.\n"
    " - Use the long-term goal as a guardrail and tie-breaker: avoid irrelevant actions, and if multiple actions could help the current task, prefer the one that better supports the long-term goal.\n\n"

    "Action requirements:\n"
    " - Output exactly one concrete in-game action.\n"
    " - The action should be immediately executable in the game.\n"
    " - Prefer simple, standard text-game commands.\n"
    " - If the current task is clear, choose an action that makes concrete progress on it.\n"
    " - If the current task cannot currently be advanced directly, choose the best supporting action that helps unlock progress while still remaining relevant to the long-term goal.\n"
    " - Avoid actions that are irrelevant to both the current task and the long-term goal.\n"
    " - Do not explore out of curiosity unless exploration is necessary to make progress on the current task or long-term goal.\n"
    " - If an object, direction, location, or interaction target is explicitly mentioned in the observation and is relevant, prefer using it.\n\n"

    "When choosing actions:\n"
    " - Prefer actions that interact with newly observed objects, containers, tools, exits, ingredients, or instructions relevant to the task.\n"
    " - Prefer reading, opening, taking, moving, examining, or using objects when these are clearly helpful for the task.\n"
    " - If navigation is required, choose the single best movement action that progresses toward the task.\n"
    " - If multiple actions are plausible, choose the one with the highest immediate usefulness.\n\n"

    "Navigation discipline:\n"
    " - If a target location is known to be important (for example, kitchen), prioritize movement actions that help locate or reach it.\n"
    " - Do not interact with irrelevant objects just because they are mentioned in the observation.\n"
    " - Do not open containers or examine objects unless doing so is necessary for the current task or likely to reveal information relevant to the long-term goal.\n"
    " - Opening a door is not a goal by itself; only do it when it enables progress toward the current task or target location.\n"
    " - When exploring, prefer a systematic strategy: try an unvisited exit from the current room before revisiting known paths.\n"

    "Output format:\n"
    " - Return only the action string.\n"
    " - Do not provide explanation.\n"
    " - Do not provide multiple candidates.\n"
    " - Do not wrap the action in JSON, XML, markdown, or quotes.\n"
)


# -----------------------------
# Helpers: parsing planner/actor outputs
# -----------------------------

_ALLOWED_STATES = {
    "progressing",
    "completed_and_useful",
    "completed_but_useless",
    "failed",
    "unprogressed",
}


def _extract_repeated_tag(text: str, tag: str) -> Optional[str]:
    """
    Extract content between "<tag>" ... "<tag>" (same opening marker repeated),
    or between "<tag>" ... "</tag>" (standard open/close).
    """
    open_tag = f"<{tag}>"
    close_tag = f"</{tag}>"

    if open_tag not in text:
        return None

    if close_tag in text:
        m = re.search(re.escape(open_tag) + r"\s*(.*?)\s*" + re.escape(close_tag), text, re.DOTALL)
        return m.group(1).strip() if m else None

    # Repeated open markers: "<tag> ... <tag>"
    # Use first and last occurrence
    first = text.find(open_tag)
    last = text.rfind(open_tag)
    if first == -1 or last == -1 or last == first:
        return None
    content = text[first + len(open_tag):last]
    return content.strip()


def _extract_all_repeated_tags(text: str, tag: str) -> List[str]:
    """
    Extract all occurrences of "<tag> ... <tag>" (repeated marker) OR "<tag> ... </tag>".
    Returns list of stripped contents.
    """
    open_tag = f"<{tag}>"
    close_tag = f"</{tag}>"
    out: List[str] = []

    if close_tag in text:
        pattern = re.escape(open_tag) + r"\s*(.*?)\s*" + re.escape(close_tag)
        for m in re.finditer(pattern, text, re.DOTALL):
            s = (m.group(1) or "").strip()
            if s:
                out.append(s)
        return out

    # Repeated open markers
    pattern = re.escape(open_tag) + r"\s*(.*?)\s*" + re.escape(open_tag)
    for m in re.finditer(pattern, text, re.DOTALL):
        s = (m.group(1) or "").strip()
        if s:
            out.append(s)
    return out


def _normalize_state(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s2 = s.strip().lower()
    # allow extra words; pick the first matching token
    for tok in re.split(r"[\s,;/]+", s2):
        if tok in _ALLOWED_STATES:
            return tok
    if s2 in _ALLOWED_STATES:
        return s2
    return None


@dataclass
class PlanResult:
    old_state: Optional[str]
    current_location: Optional[str]
    new_current_task: Optional[str]
    candidate_tasks: List[str]
    raw: str


@dataclass
class ActResult:
    think: Optional[str]
    action: str
    raw: str


def parse_plan_output(text: str) -> PlanResult:
    old_state_raw = _extract_repeated_tag(text, "active_task_state")
    if old_state_raw is None:
        old_state_raw = _extract_repeated_tag(text, "old current task state")
    old_state = _normalize_state(old_state_raw)

    new_task = _extract_repeated_tag(text, "active_task_after")
    if new_task is None:
        new_task = _extract_repeated_tag(text, "new current task")
    current_location = _extract_repeated_tag(text, "location")
    candidates = _extract_all_repeated_tags(text, "candidate task")

    # Fallbacks: tolerate non-tag outputs.
    if old_state is None:
        m = re.search(
            r"\b(progressing|completed_and_useful|completed_but_useless|failed|unprogressed)\b",
            (text or "").lower(),
        )
        old_state = m.group(1) if m else None

    if new_task is None:
        m = re.search(r"active_task_after\s*[:\-]\s*(.+)", text, re.IGNORECASE)
        if m:
            new_task = m.group(1).strip()

    if current_location is None:
        m = re.search(r"\blocation\s*[:\-]\s*(.+)", text, re.IGNORECASE)
        if m:
            current_location = m.group(1).strip()

    if new_task is None:
        m = re.search(r"new current task\s*[:\-]\s*(.+)", text, re.IGNORECASE)
        if m:
            new_task = m.group(1).strip()

    if not candidates:
        m = re.search(r"candidate tasks?\s*[:\-]\s*(.+)", text, re.IGNORECASE | re.DOTALL)
        if m:
            tail = m.group(1)
            lines = [ln.strip(" -*\t") for ln in tail.splitlines() if ln.strip()]
            candidates = [ln for ln in lines if ln]

    if new_task is not None:
        new_task = new_task.strip()
    candidates = [c.strip() for c in candidates if c.strip()]

    return PlanResult(
        old_state=old_state,
        current_location=(current_location or "").strip() or None,
        new_current_task=new_task,
        candidate_tasks=candidates,
        raw=text,
    )


def parse_act_output(text: str) -> ActResult:
    think = _extract_repeated_tag(text, "think")
    action = _extract_repeated_tag(text, "action")

    # Fallback: if tags missing, pick the first plausible command line.
    if not action:
        lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
        cleaned = []
        for ln in lines:
            low = ln.lower()
            if low.startswith("<think") or low.startswith("</think") or low.startswith("<action") or low.startswith("</action"):
                continue
            if low.startswith("thought:") or low.startswith("reasoning:"):
                continue
            cleaned.append(ln)
        action = cleaned[0] if cleaned else ""

    # If model still returned a tagged/empty artifact, sanitize aggressively.
    if action:
        action = re.sub(r"</?action>", "", action, flags=re.IGNORECASE).strip()
        action = re.sub(r"</?think>", "", action, flags=re.IGNORECASE).strip()

    return ActResult(think=think, action=action.strip(), raw=text)


# -----------------------------
# Agent
# -----------------------------

class SimplePlanningAgent(tales.Agent):
    """
    Two-stage LLM agent:
      1) planner updates task backlog
      2) actor executes one action for current task
    """

    def __init__(self, *args, **kwargs):
        self.llm = kwargs["llm"]
        self.model = llm.get_model(self.llm)
        self.token_counter = get_token_counter(self.model)

        # Provide the API key, if one is needed and has been provided
        self.model.key = llm.get_key(
            kwargs.get("key"), kwargs["llm"], self.model.key_env_var
        ) or llm.get_key(None, self.model.needs_key, self.model.key_env_var)

        self.seed = kwargs["seed"]
        self.rng = np.random.RandomState(self.seed)

        # Full trajectory across episodes:
        # (episode_idx, step_idx_in_episode, observation, action)
        self.history: List[Tuple[int, int, str, str]] = []
        self.episode_idx: int = 0

        self.context_limit = kwargs.get("context_limit")
        if self.context_limit is not None:
            assert self.context_limit > 0, "--context-limit must be greater than 0."

        self.conversation = kwargs["conversation"]

        # Temperatures
        self.plan_temp = kwargs.get("plan_temp", 0.2)
        self.act_temp = kwargs.get("act_temp", 0.0)

        # Backlog state
        self.current_task: str = kwargs.get("init_task") or "Explore the environment (try 'look' or 'help')."
        self.current_location: Optional[str] = None
        self.candidate_tasks: List[str] = []
        self.long_term_goal: str = "achieve as much score as possible"
        self.completed_and_useful_tasks: Set[str] = set()
        self.completed_but_useless_tasks: Set[str] = set()
        # Backward compatibility for existing logs/bench tooling.
        self.success_tasks: Set[str] = self.completed_and_useful_tasks
        self.failed_tasks: Set[str] = set()
        self.unprogressed_tasks: Set[str] = set()

        # Progress tracking for "unprogressed"
        self.step_idx: int = 0
        self.current_task_start_step: int = 1
        self.last_progress_step: int = 0

    @property
    def uid(self):
        return (
            f"SimplePlanningAgent_{self.llm}"
            f"_s{self.seed}"
            f"_c{self.context_limit}"
            f"_pt{self.plan_temp}"
            f"_at{self.act_temp}"
            f"_conv{self.conversation}"
        )

    @property
    def params(self):
        return {
            "agent_type": "simple-planning",
            "llm": self.llm,
            "seed": self.seed,
            "context_limit": self.context_limit,
            "plan_temp": self.plan_temp,
            "act_temp": self.act_temp,
            "conversation": self.conversation,
        }

    @retry(
        retry=retry_if_exception(is_recoverable_error),
        wait=wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(100),
    )
    def _llm_call_from_conversation(self, conversation, *args, **kwargs):
        response = conversation.prompt(*args, **kwargs)
        response.duration_ms()  # Forces the response to be computed.
        return response

    def _llm_call_from_messages(self, messages, *args, **kwargs):
        conversation = messages2conversation(self.model, messages)
        prompt = messages[-1]["content"]

        # Some models do not allow system prompt; mimic llm.py behavior
        allows_system_prompt = self.llm not in ["o1-mini", "o1-preview"]
        system = messages[0]["content"] if (allows_system_prompt and messages and messages[0]["role"] == "system") else None

        if not allows_system_prompt and messages and messages[0]["role"] == "system":
            # Remove system, prepend to next user message if possible
            sys_msg = messages.pop(0)["content"]
            # Find first user message
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
        # Keep success/failed/unprogressed across restarts (as per design),
        # but re-seed step counters.
        self.episode_idx += 1
        self.step_idx = 0
        self.current_task_start_step = 1
        self.last_progress_step = 0
        self.current_location = self._infer_location_from_observation(obs)

    # -----------------------------
    # Message builders
    # -----------------------------

    def _format_history(self, limit_turns: int, *, items=None, start_step=None) -> str:
        # Show the last N steps
        items = list(items) if items is not None else (
            self.history[-limit_turns:] if limit_turns > 0 else self.history
        )
        lines = []
        for item in items:
            if len(item) == 4:
                ep, step, obs, act = item
            else:
                # Backward-compatible fallback if old tuple shape appears.
                ep, step = self.episode_idx, 0
                obs, act = item  # type: ignore[misc]
            o = obs.replace("\n", "\\n")
            a = act.replace("\n", "\\n")
            lines.append(f"episode {ep}-step {step} | observation: {o} | action: {a}")
        return "\n".join(lines) if lines else "(empty)"

    def _infer_location_from_observation(self, observation: str) -> Optional[str]:
        text = observation or ""
        matches = re.findall(r"-=\s*([^=\n]+?)\s*=-", text)
        for m in reversed(matches):
            loc = m.strip()
            if not loc:
                continue
            if loc.lower() == "restarting":
                continue
            return loc
        return None
    
    def build_goal_messages(self, observation: str, info: dict) -> List[dict]:
        messages = [{"role": "system", "content": GOAL_MAKER_SYSTEM_PROMPT}]

        prompt = (
            f"Initial observation:\n{observation}\n\n"
            f"Output the single long-term goal only.\n"
        )

        messages.append({"role": "user", "content": prompt})
        messages = merge_messages(messages)

        if not self.conversation:
            content = "".join([msg["content"] for msg in messages[1:]])
            messages = messages[:1] + [{"role": "user", "content": content}]

        return messages

    def build_plan_messages(self, observation: str, info: dict) -> List[dict]:
        messages = [{"role": "system", "content": PLANNER_SYSTEM_PROMPT}]

        # Planner gets full trajectory context for better long-horizon planning.
        limit_turns = len(self.history) + 1
        history_str = self._format_history(limit_turns)
        active_start_step = max(1, int(self.current_task_start_step))
        active_items = [
            item for item in self.history
            if len(item) == 4 and item[0] == self.episode_idx and item[1] >= active_start_step
        ]
        active_history_str = self._format_history(
            limit_turns=0,
            items=active_items,
            start_step=active_start_step,
        )

        restart_detected = "-= Restarting =-" in observation

        backlog_str = (
            f"Active task before planning: {self.current_task}\n"
            + (
                "Candidate tasks:\n" + "\n".join([f"- {t}" for t in self.candidate_tasks])
                if self.candidate_tasks
                else "Candidate tasks:\n- (none)"
            )
        )

        # Add minimal progress hints
        score = info.get("score")
        max_score = info.get("max_score")
        feedback = info.get("feedback", "")

        prev_sets = (
            f"All completed_and_useful tasks:\n- "
            + ("\n- ".join(sorted(self.completed_and_useful_tasks)) if self.completed_and_useful_tasks else "(none)")
            + "\n\n"
            f"All completed_but_useless tasks:\n- "
            + ("\n- ".join(sorted(self.completed_but_useless_tasks)) if self.completed_but_useless_tasks else "(none)")
            + "\n\n"
            f"All failed tasks:\n- " + ("\n- ".join(sorted(self.failed_tasks)) if self.failed_tasks else "(none)") + "\n\n"
            f"All unprogressed tasks:\n- " + ("\n- ".join(sorted(self.unprogressed_tasks)) if self.unprogressed_tasks else "(none)")
        )

        prompt = (
            f"Long-term goal: {self.long_term_goal}\n"
            f"Current Episode: {self.episode_idx}\n"
            f"Step: {self.step_idx + 1}\n"
            f"Active Task Start Step: {active_start_step}\n"
            f"Current Known Location: {self.current_location or '(unknown)'}\n"
            f"Restart detected: {restart_detected}\n\n"
            f"=== Current Backlog ===\n{backlog_str}\n\n"
            f"=== Full History Across Episodes (most recent {limit_turns} steps) ===\n{history_str}\n\n"
            f"=== Current Episode History Since Active Task Start Step (episode {self.episode_idx}, step {active_start_step}..now) ===\n{active_history_str}\n\n"
            f"=== Previous Task Outcomes ===\n{prev_sets}\n\n"
            f"=== Current Observation ===\n"
            f"(score: {score}, max_score: {max_score})\n"
            f"(feedback: {feedback})\n"
            f"{observation}\n"
        )

        messages.append({"role": "user", "content": prompt})
        messages = merge_messages(messages)

        if not self.conversation:
            # Merge all into one user message except system
            content = "".join([msg["content"] for msg in messages[1:]])
            messages = messages[:1] + [{"role": "user", "content": content}]

        return messages

    def build_act_messages(self, observation: str, info: dict) -> List[dict]:
        messages = [{"role": "system", "content": ACTOR_SYSTEM_PROMPT}]

        limit_turns = self.context_limit or (len(self.history) + 1)
        history_str = self._format_history(limit_turns)

        admissible = info.get("admissible_commands") or []
        admissible_block = ""
        if admissible:
            # Keep it short to avoid blowing context
            admissible_preview = admissible[:50]
            admissible_block = "Admissible commands (subset):\n- " + "\n- ".join(admissible_preview) + "\n\n"

        prompt = (
            f"Current Episode: {self.episode_idx}\n"
            f"Step: {self.step_idx + 1}\n"
            f"Long-term goal: {self.long_term_goal}\n"
            f"CURRENT TASK:\n{self.current_task}\n\n"
            f"{admissible_block}"
            f"Recent history:\n{history_str}\n\n"
            f"Observation:\n{observation}\n> "
        )

        messages.append({"role": "user", "content": prompt})
        messages = merge_messages(messages)

        if not self.conversation:
            content = "".join([msg["content"] for msg in messages[1:]])
            messages = messages[:1] + [{"role": "user", "content": content}]

        return messages

    # -----------------------------
    # Core step: plan then act
    # -----------------------------

    def _is_progress_signal(self, observation: str, info: dict, prev_score: Optional[int]) -> bool:
        """
        Simple heuristic to mark progress.
        The planner still decides task state, but this helps "10 steps no progress" logic.
        """
        score = info.get("score")
        if prev_score is not None and score is not None and score > prev_score:
            return True

        # Common game feedback patterns (very lightweight)
        text = (observation + "\n" + (info.get("feedback", "") or "")).lower()
        keywords = [
            "taken", "you take", "you pick up", "picked up",
            "opened", "unlocked", "you unlock",
            "score", "points",
            "you win", "won",
        ]
        return any(k in text for k in keywords)

    def act(self, obs, reward, done, infos):
        prev_score = infos.get("score")
        observed_location = self._infer_location_from_observation(obs)
        if observed_location:
            self.current_location = observed_location

        goal_messages = None
        goal_resp = None
        goal_text = None

        # 1) GOAL MAKER (one-time at episode 1, step 1)
        if self.episode_idx == 1 and self.step_idx == 0:
            goal_messages = self.build_goal_messages(obs, infos)
            goal_kwargs = self._build_llm_kwargs(temperature=0.2, max_tokens=60)
            goal_resp = self._llm_call_from_messages(goal_messages, **goal_kwargs)
            goal_text = goal_resp.text().strip()
            if goal_text:
                self.long_term_goal = goal_text.splitlines()[0].strip()
            else:
                self.long_term_goal = "achieve as much score as possible"

        # 2) PLAN
        plan_messages = self.build_plan_messages(obs, infos)
        plan_kwargs = self._build_llm_kwargs(temperature=self.plan_temp, max_tokens=300)
        plan_resp = self._llm_call_from_messages(plan_messages, **plan_kwargs)
        plan_text = plan_resp.text().strip()
        plan_result = parse_plan_output(plan_text)

        old_task = self.current_task
        old_state = plan_result.old_state
        if plan_result.current_location:
            self.current_location = plan_result.current_location.strip()

        # Update sets/backlog based on plan
        if old_state in {"completed_and_useful", "completed_but_useless", "failed", "unprogressed"}:
            if old_state == "completed_and_useful":
                self.completed_and_useful_tasks.add(old_task)
            elif old_state == "completed_but_useless":
                self.completed_but_useless_tasks.add(old_task)
            elif old_state == "failed":
                self.failed_tasks.add(old_task)
            else:
                self.unprogressed_tasks.add(old_task)

            # Replace current task if planner provided one
            if plan_result.new_current_task:
                self.current_task = plan_result.new_current_task
            else:
                # Fallback: choose first candidate if available
                if plan_result.candidate_tasks:
                    self.current_task = plan_result.candidate_tasks[0]
                else:
                    self.current_task = "Explore the environment (try 'look' or 'help' or examine relevant items)."

            self.current_task_start_step = self.step_idx + 1
            self.last_progress_step = self.step_idx  # reset window

        else:
            # progressing or unknown => keep current task
            # if planner tried to change it anyway, ignore per spec
            pass

        # Candidate tasks: replace (dedupe, remove current task)
        candidates = []
        seen = set()
        for t in plan_result.candidate_tasks:
            t2 = t.strip()
            if not t2 or t2 == self.current_task:
                continue
            if t2 not in seen:
                candidates.append(t2)
                seen.add(t2)
        self.candidate_tasks = candidates[:10]

        # 3) ACT
        act_messages = self.build_act_messages(obs, infos)
        act_kwargs = self._build_llm_kwargs(temperature=self.act_temp, max_tokens=120)
        act_resp = self._llm_call_from_messages(act_messages, **act_kwargs)
        act_text = act_resp.text().strip()
        act_result = parse_act_output(act_text)
        action = act_result.action.strip()
        if not action:
            action = "help"

        # Update local history
        self.history.append((self.episode_idx, self.step_idx + 1, f"{obs}\n> ", f"{action}\n"))

        # Update progress heuristic timestamps
        self.step_idx += 1
        if self._is_progress_signal(obs, infos, prev_score):
            self.last_progress_step = self.step_idx

        # Compute usage stats (separate plan vs act)
        stats = {
            # GOAL
            "long_term_goal": self.long_term_goal,
            "goal_response": goal_resp.text() if goal_resp is not None else None,
            "nb_tokens_goal_prompt": self.token_counter(messages=goal_messages) if goal_messages is not None else 0,
            "nb_tokens_goal_response": self.token_counter(text=goal_resp.text()) if goal_resp is not None else 0,
            # PLAN
            "plan_prompt": format_messages_to_markdown(plan_messages),
            "plan_response": plan_resp.text(),
            "plan_old_state": old_state,
            "plan_old_task": old_task,
            "plan_new_task": self.current_task,
            "plan_candidates": list(self.candidate_tasks),
            "nb_tokens_plan_prompt": self.token_counter(messages=plan_messages),
            "nb_tokens_plan_response": self.token_counter(text=plan_resp.text()),
            # ACT
            "act_prompt": format_messages_to_markdown(act_messages),
            "act_response": act_resp.text(),
            "thinking": act_result.think,
            "nb_tokens_act_prompt": self.token_counter(messages=act_messages),
            "nb_tokens_act_response": self.token_counter(text=act_resp.text()),
        }

        stats["nb_tokens_plan"] = stats["nb_tokens_plan_prompt"] + stats["nb_tokens_plan_response"]
        stats["nb_tokens_act"] = stats["nb_tokens_act_prompt"] + stats["nb_tokens_act_response"]
        stats["nb_tokens_goal"] = stats.get("nb_tokens_goal_prompt", 0) + stats.get("nb_tokens_goal_response", 0)
        stats["nb_tokens"] = stats["nb_tokens_plan"] + stats["nb_tokens_act"] + stats["nb_tokens_goal"]

        # Snapshots for benchmark printing
        stats["current_task"] = self.current_task
        stats["current_location"] = self.current_location
        stats["success_tasks"] = sorted(self.success_tasks)
        stats["completed_and_useful_tasks"] = sorted(self.completed_and_useful_tasks)
        stats["completed_but_useless_tasks"] = sorted(self.completed_but_useless_tasks)
        stats["failed_tasks"] = sorted(self.failed_tasks)
        stats["unprogressed_tasks"] = sorted(self.unprogressed_tasks)

        return action, stats


# -----------------------------
# Argparser / register
# -----------------------------

def build_argparser(parser=None):
    parser = parser or argparse.ArgumentParser()
    group = parser.add_argument_group("SimplePlanningAgent settings")

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
        "--plan-temp",
        type=float,
        default=0.2,
        help="Temperature for planner stage. Default: %(default)s",
    )
    group.add_argument(
        "--act-temp",
        type=float,
        default=0.0,
        help="Temperature for actor stage. Default: %(default)s",
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
    group.add_argument(
        "--init-task",
        type=str,
        default=None,
        help="Initial current task for planner. Default: auto.",
    )

    return parser


register(
    name="backlog",
    desc="Two-stage agent: planner maintains backlog, actor executes actions toward current task.",
    klass=SimplePlanningAgent,
    add_arguments=build_argparser,
)
