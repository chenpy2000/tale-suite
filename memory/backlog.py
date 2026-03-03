import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

WAITING = "waiting"
IN_PROGRESS = "in_progress"
COMPLETED = "completed"
FAILED = "failed"
UNPROGRESSED = "unprogressed"


@dataclass
class BacklogTask:
    task_id: int
    description: str
    status: str = WAITING
    attempts: int = 0
    last_update_step: int = 0
    created_step: int = 0
    active_since_step: int = 0
    terminal_step: int = 0
    cooldown_until_step: int = 0
    failure_reasons: List[str] = field(default_factory=list)
    last_failed_action: str = ""


class BacklogManager:
    """Maintains a small task backlog and one active task focus."""

    def __init__(
        self,
        max_waiting: int = 5,
        failure_threshold: int = 2,
        failed_cooldown_steps: int = 6,
    ):
        self.max_waiting = max(1, int(max_waiting))
        self.failure_threshold = max(1, int(failure_threshold))
        self.failed_cooldown_steps = max(1, int(failed_cooldown_steps))

        self.tasks: List[BacklogTask] = []
        self.current_task_id: Optional[int] = None
        self.next_task_id = 1

        self.last_action: Optional[str] = None
        self.last_action_task_id: Optional[int] = None
        self.failed_action_counts: Dict[str, int] = defaultdict(int)
        self.failure_events: List[Dict[str, Any]] = []
        self.episode_restarts = 0

    def reset_all(self) -> None:
        self.tasks = []
        self.current_task_id = None
        self.next_task_id = 1
        self.last_action = None
        self.last_action_task_id = None
        self.failed_action_counts = defaultdict(int)
        self.failure_events = []
        self.episode_restarts = 0

    def on_episode_restart(self, step: int) -> None:
        self.episode_restarts += 1
        if self.last_action_task_id is not None:
            pending_task = self._get_task(self.last_action_task_id)
            if pending_task is not None and pending_task.status == IN_PROGRESS:
                self._mark_failure(
                    pending_task,
                    step,
                    reasons=["episode_restart"],
                    last_action=self.last_action,
                    force_fail=True,
                )
            self._record_failure_event(
                step=step,
                reason="episode_restart",
                action=self.last_action,
                task=pending_task.description if pending_task else None,
                observation="",
            )
        for task in self.tasks:
            if task.status in {IN_PROGRESS, COMPLETED}:
                task.status = WAITING
                task.last_update_step = step
        self.current_task_id = None
        self.last_action = None
        self.last_action_task_id = None
        self._ensure_single_in_progress(step)
        self._activate_next_task(step)

    def update_from_observation(
        self,
        observation: str,
        updates: Dict[str, Any],
        step: int,
        llm_tasks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        self._apply_last_action_feedback(observation, updates, step)
        self._ensure_single_in_progress(step)

        for task_text in llm_tasks or []:
            self._add_task(task_text, step)

        self._activate_next_task(step)

        if self._waiting_count() == 0:
            self._recycle_failed_if_needed(step)

        return self.export()

    def record_action(self, action: str) -> None:
        self.last_action = (action or "").strip()
        self.last_action_task_id = self.current_task_id

    def apply_llm_task_judgement(
        self,
        status: str,
        step: int,
        reason: str = "",
        last_action: str = "",
    ) -> None:
        """Apply LLM judgement to the currently active task."""
        task = self.current_task()
        if task is None:
            return

        normalized = str(status or "").strip().lower()
        if normalized in {"keep", "continue"}:
            normalized = IN_PROGRESS
        if normalized in {"none", ""}:
            return

        if normalized == COMPLETED:
            task.status = COMPLETED
            task.last_update_step = step
            task.terminal_step = step
            task.attempts = 0
            self.current_task_id = None
            self._activate_next_task(step)
            return

        if normalized in {FAILED, UNPROGRESSED}:
            clean_reason = self._normalize_llm_reason(reason)
            failure_status = UNPROGRESSED if normalized == UNPROGRESSED else FAILED
            self._mark_failure(
                task,
                step,
                reasons=[clean_reason],
                last_action=last_action or self.last_action,
                force_fail=True,
                failure_status=failure_status,
            )
            self._activate_next_task(step)
            return

        if normalized == IN_PROGRESS:
            task.status = IN_PROGRESS
            task.last_update_step = step
            self.current_task_id = task.task_id
            return

    def prioritize_actions(
        self,
        admissible: List[str],
        candidates: List[str],
    ) -> List[str]:
        focused = self._task_actions(admissible)
        return self._dedupe(focused + candidates)

    def export(self) -> Dict[str, Any]:
        current = self.current_task()
        return {
            "current_task": current.description if current else None,
            "waiting": [t.description for t in self.tasks if t.status == WAITING],
            "in_progress": [t.description for t in self.tasks if t.status == IN_PROGRESS],
            "completed": [t.description for t in self.tasks if t.status == COMPLETED],
            "failed": [t.description for t in self.tasks if t.status == FAILED],
            "unprogressed": [t.description for t in self.tasks if t.status == UNPROGRESSED],
            "tasks": [
                {
                    "task_id": t.task_id,
                    "description": t.description,
                    "status": t.status,
                    "attempts": t.attempts,
                    "last_update_step": t.last_update_step,
                    "created_step": t.created_step,
                    "active_since_step": t.active_since_step,
                    "terminal_step": t.terminal_step,
                    "cooldown_until_step": t.cooldown_until_step,
                    "failure_reasons": list(t.failure_reasons[-6:]),
                    "last_failed_action": t.last_failed_action,
                }
                for t in self.tasks
            ],
            "max_waiting": self.max_waiting,
            "failure_threshold": self.failure_threshold,
            "failed_cooldown_steps": self.failed_cooldown_steps,
            "episode_restarts": self.episode_restarts,
            "risky_actions": self.risky_actions(),
            "failure_events": list(self.failure_events[-8:]),
        }

    def set_current_task(
        self,
        task_text: str,
        step: int,
        allow_create: bool = True,
    ) -> bool:
        normalized = self._normalize_task(task_text)
        if not normalized:
            return False

        target = self._find_task_by_description(normalized)
        if target is None:
            if not allow_create:
                return False
            if self._waiting_count() >= self.max_waiting:
                return False
            self._add_task(normalized, step)
            target = self._find_task_by_description(normalized)
            if target is None:
                return False

        if target.status in {FAILED, UNPROGRESSED} and step < target.cooldown_until_step:
            return False
        if target.status in {FAILED, UNPROGRESSED} and step >= target.cooldown_until_step:
            target.status = WAITING
            target.attempts = 0
            target.last_update_step = step

        current = self.current_task()
        if current is not None and current.task_id != target.task_id and current.status == IN_PROGRESS:
            current.status = WAITING
            current.last_update_step = step

        previous_status = target.status
        target.status = IN_PROGRESS
        target.last_update_step = step
        if target.active_since_step <= 0 or target.active_since_step > step or previous_status != IN_PROGRESS:
            target.active_since_step = step
        self.current_task_id = target.task_id
        self._ensure_single_in_progress(step)
        return True

    def current_task_step_window(self, step: int) -> Optional[Dict[str, int]]:
        task = self.current_task()
        if task is None:
            return None
        start = task.active_since_step or task.last_update_step or task.created_step or step
        return {"start": int(start), "end": int(step)}

    def current_task(self) -> Optional[BacklogTask]:
        if self.current_task_id is None:
            return None
        for task in self.tasks:
            if task.task_id == self.current_task_id:
                return task
        return None

    def waiting_count(self) -> int:
        return self._waiting_count()

    def has_waiting_capacity(self) -> bool:
        return self._waiting_count() < self.max_waiting

    def risky_actions(self, threshold: int = 2) -> List[str]:
        min_count = max(1, int(threshold))
        return sorted(
            action
            for action, count in self.failed_action_counts.items()
            if count >= min_count
        )

    def add_risky_actions(
        self,
        actions: List[str],
        step: int,
        weight: int = 1,
        reason: str = "llm_reflection",
        task: Optional[str] = None,
    ) -> None:
        boost = max(1, int(weight))
        for raw in actions:
            action = self._normalize_action(raw)
            if not action:
                continue
            self.failed_action_counts[action] += boost
            self._record_failure_event(
                step=step,
                reason=reason,
                action=action,
                task=task,
                observation="",
            )

    def recent_failed_actions(
        self,
        limit: int = 4,
        reasons: Optional[List[str]] = None,
    ) -> List[str]:
        wanted = {str(x).strip().lower() for x in (reasons or []) if str(x).strip()}
        out: List[str] = []
        seen = set()
        for event in reversed(self.failure_events):
            action = self._normalize_action(event.get("action") or "")
            reason = str(event.get("reason") or "").strip().lower()
            if not action:
                continue
            if wanted and reason not in wanted:
                continue
            if action in seen:
                continue
            seen.add(action)
            out.append(action)
            if len(out) >= max(1, int(limit)):
                break
        return out

    def _apply_last_action_feedback(
        self,
        observation: str,
        updates: Dict[str, Any],
        step: int,
    ) -> None:
        if not self.last_action:
            self.last_action_task_id = None
            return

        task: Optional[BacklogTask] = None
        if self.last_action_task_id is not None:
            task = self._get_task(self.last_action_task_id)

        failure_reasons = self._collect_failure_reasons(observation, updates)
        action = self._normalize_action(self.last_action)

        if task is None or task.status != IN_PROGRESS:
            if failure_reasons and action:
                self.failed_action_counts[action] += 1
                for reason in failure_reasons:
                    self._record_failure_event(
                        step=step,
                        reason=reason,
                        action=action,
                        task=None,
                        observation=observation,
                    )
            self.last_action = None
            self.last_action_task_id = None
            return

        lowered = (observation or "").lower()
        successes = updates.get("successes", [])
        failure_signal = bool(failure_reasons)
        terminal_failure = "terminal_loss" in failure_reasons or "episode_restart" in failure_reasons

        success_signal = bool(successes) or any(
            token in lowered
            for token in ["you open", "you unlock", "you take", "you pick up", "you win"]
        )

        if failure_signal and (terminal_failure or not success_signal):
            self._mark_failure(
                task,
                step,
                reasons=failure_reasons,
                last_action=self.last_action,
                force_fail=terminal_failure,
                observation=observation,
            )
        elif self._looks_like_task_completion(task, action, observation, successes):
            task.status = COMPLETED
            task.last_update_step = step
            task.terminal_step = step
            task.attempts = 0
            self.current_task_id = None

        self.last_action = None
        self.last_action_task_id = None

    def _looks_like_task_completion(
        self,
        task: BacklogTask,
        action: str,
        observation: str,
        successes: List[Any],
    ) -> bool:
        task_text = str(task.description or "").lower()
        action_tokens = set(re.findall(r"[a-z0-9_]+", str(action or "").lower()))
        task_tokens = set(re.findall(r"[a-z0-9_]+", task_text))
        obs_tokens = set(re.findall(r"[a-z0-9_]+", str(observation or "").lower()))
        success_tokens = {str(item).strip().lower() for item in successes if str(item).strip()}

        if not task_tokens:
            return bool(success_tokens)
        if "explore" in task_tokens:
            if action_tokens & {"north", "south", "east", "west", "up", "down", "go"}:
                return not self._looks_like_direct_failure(observation)
        if task_tokens & action_tokens:
            return True
        if len(task_tokens & obs_tokens) >= 2 and success_tokens:
            return True
        if success_tokens & {"score_increased", "you win", "completed"}:
            return True
        return False

    def _mark_failure(
        self,
        task: BacklogTask,
        step: int,
        reasons: List[str],
        last_action: Optional[str],
        force_fail: bool = False,
        observation: str = "",
        failure_status: str = FAILED,
    ) -> None:
        task.attempts += 1
        task.last_update_step = step

        action = (last_action or "").strip().lower()
        if action:
            task.last_failed_action = action
            self.failed_action_counts[action] += 1

        for reason in reasons:
            clean = str(reason).strip().lower()
            if clean and clean not in task.failure_reasons:
                task.failure_reasons.append(clean)
            self._record_failure_event(
                step=step,
                reason=clean,
                action=action or None,
                task=task.description,
                observation=observation,
            )

        if force_fail:
            task.attempts = max(task.attempts, self.failure_threshold)

        if task.attempts >= self.failure_threshold:
            terminal = UNPROGRESSED if failure_status == UNPROGRESSED else FAILED
            task.status = terminal
            task.terminal_step = step
            task.cooldown_until_step = step + self.failed_cooldown_steps
            self.current_task_id = None

    def _record_failure_event(
        self,
        step: int,
        reason: str,
        action: Optional[str],
        task: Optional[str],
        observation: str,
    ) -> None:
        self.failure_events.append(
            {
                "step": int(step),
                "reason": str(reason or "").strip() or "unknown_failure",
                "action": str(action or "").strip() or None,
                "task": str(task or "").strip() or None,
                "observation": self._normalize_task(observation),
            }
        )
        if len(self.failure_events) > 64:
            self.failure_events = self.failure_events[-64:]

    @staticmethod
    def _normalize_llm_reason(text: str) -> str:
        cleaned = " ".join(str(text or "").strip().lower().split())
        if not cleaned:
            return "llm_marked_failed"
        return f"llm:{cleaned[:80]}"

    def _collect_failure_reasons(
        self,
        observation: str,
        updates: Dict[str, Any],
    ) -> List[str]:
        reasons = [str(item).strip().lower() for item in updates.get("failures", []) if str(item).strip()]
        lowered = (observation or "").lower()

        if "you burned" in lowered and "burned_item" not in reasons:
            reasons.append("burned_item")
        if self._looks_like_terminal_failure(observation) and "terminal_loss" not in reasons:
            reasons.append("terminal_loss")
        if self._looks_like_direct_failure(observation) and "action_rejected" not in reasons:
            reasons.append("action_rejected")

        return self._dedupe(reasons)

    def _add_task(self, text: str, step: int) -> None:
        normalized = self._normalize_task(text)
        if not normalized:
            return

        for task in self.tasks:
            existing = self._normalize_task(task.description)
            if existing == normalized:
                return
            if self._task_similarity(existing, normalized) >= 0.8:
                return

        if self._waiting_count() >= self.max_waiting:
            return

        self.tasks.append(
            BacklogTask(
                task_id=self.next_task_id,
                description=normalized,
                status=WAITING,
                attempts=0,
                last_update_step=step,
                created_step=step,
            )
        )
        self.next_task_id += 1

    def _activate_next_task(self, step: int) -> None:
        current = self.current_task()
        if current is not None and current.status == IN_PROGRESS:
            return

        self.current_task_id = None
        for task in self.tasks:
            if task.status == WAITING:
                task.status = IN_PROGRESS
                task.last_update_step = step
                task.active_since_step = step
                self.current_task_id = task.task_id
                return

    def _ensure_single_in_progress(self, step: int) -> None:
        in_progress = [task for task in self.tasks if task.status == IN_PROGRESS]
        if not in_progress:
            if self.current_task_id is not None:
                current = self._get_task(self.current_task_id)
                if current is None or current.status != IN_PROGRESS:
                    self.current_task_id = None
            return

        if len(in_progress) == 1:
            self.current_task_id = in_progress[0].task_id
            return

        keep = self._get_task(self.current_task_id) if self.current_task_id is not None else None
        if keep is None or keep.status != IN_PROGRESS:
            keep = in_progress[0]

        for task in in_progress:
            if task.task_id == keep.task_id:
                continue
            task.status = WAITING
            task.last_update_step = step

        self.current_task_id = keep.task_id

    def _recycle_failed_if_needed(self, step: int) -> None:
        if self.current_task_id is not None:
            return

        for task in self.tasks:
            if task.status in {FAILED, UNPROGRESSED} and step >= task.cooldown_until_step:
                task.status = WAITING
                task.attempts = 0
                task.last_update_step = step
                self._activate_next_task(step)
                return

    def _task_actions(self, admissible: List[str]) -> List[str]:
        task = self.current_task()
        if task is None:
            return []

        if not admissible:
            return []

        task_text = task.description.lower()
        keywords = set(re.findall(r"[a-z0-9_]+", task_text))

        scored = []
        for cmd in admissible:
            lowered = cmd.lower()
            cmd_tokens = set(re.findall(r"[a-z0-9_]+", lowered))
            overlap = len(keywords & cmd_tokens)
            if overlap > 0:
                scored.append((overlap, cmd))

            if "explore" in task_text:
                for direction in ["north", "south", "east", "west", "up", "down"]:
                    if direction in task_text and lowered == direction:
                        scored.append((100, cmd))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [cmd for _, cmd in scored]

    def _waiting_count(self) -> int:
        return sum(1 for task in self.tasks if task.status == WAITING)

    def _get_task(self, task_id: int) -> Optional[BacklogTask]:
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None

    def _find_task_by_description(self, normalized: str) -> Optional[BacklogTask]:
        for task in self.tasks:
            existing = self._normalize_task(task.description)
            if existing == normalized:
                return task
            if self._task_similarity(existing, normalized) >= 0.85:
                return task
        return None

    @staticmethod
    def _normalize_task(text: str) -> str:
        cleaned = " ".join(str(text).strip().split())
        return cleaned[:160]

    @staticmethod
    def _normalize_action(text: str) -> str:
        cleaned = " ".join(str(text).strip().lower().split())
        return cleaned[:120]

    @staticmethod
    def _looks_like_direct_failure(observation: str) -> bool:
        patterns = [
            r"^(you can't|you cannot)\b",
            r"^nothing happens\b",
            r"^(that's|this is) not possible\b",
            r"^that doesn't work\b",
            r"^(invalid|i don't understand|unknown command)\b",
            r"^no effect\b",
            r"^failed\b",
        ]
        for raw_line in str(observation or "").splitlines():
            line = " ".join(raw_line.strip().lower().split())
            if not line:
                continue
            for pattern in patterns:
                if re.search(pattern, line):
                    return True
        return False

    @staticmethod
    def _looks_like_terminal_failure(observation: str) -> bool:
        patterns = [
            r"^\*+\s*you lost!?\b",
            r"^would you like to quit\??\b",
            r"^game over\b",
            r"^you (died|are dead)\b",
        ]
        for raw_line in str(observation or "").splitlines():
            line = " ".join(raw_line.strip().lower().split())
            if not line:
                continue
            for pattern in patterns:
                if re.search(pattern, line):
                    return True
        return False

    @staticmethod
    def _task_similarity(a: str, b: str) -> float:
        tokens_a = set(re.findall(r"[a-z0-9_]+", a.lower()))
        tokens_b = set(re.findall(r"[a-z0-9_]+", b.lower()))
        if not tokens_a or not tokens_b:
            return 0.0
        overlap = len(tokens_a & tokens_b)
        denom = float(min(len(tokens_a), len(tokens_b)))
        return overlap / denom if denom > 0 else 0.0

    @staticmethod
    def _dedupe(values: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for value in values:
            key = (value or "").strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(value)
        return out
