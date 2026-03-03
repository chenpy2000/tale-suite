import argparse
import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

import tales
from memory import (
    BacklogManager,
    MemoryCompressor,
    ObservationParser,
    OptionModule,
    Retriever,
    Scratchpad,
)
from tales.agent import register
from tales.token import get_token_counter


class PlanningAgent(tales.Agent):
    """Planning agent with a task backlog, scratchpad, compression, and retrieval."""

    def __init__(self, **kwargs):
        self.seed = kwargs.get("seed", 20241001)
        self.rng = np.random.RandomState(self.seed)
        self.token_counter = get_token_counter()

        self.variant = kwargs.get("memory_variant", "structured")
        self.compress_every = int(kwargs.get("compress_every", 8))
        self.max_context_items = int(kwargs.get("max_context_items", 12))
        self.use_llm_policy = bool(kwargs.get("use_llm_policy", False))
        self.max_prompt_chars = int(kwargs.get("max_prompt_chars", 6500))
        self.max_repeat_action = int(kwargs.get("max_repeat_action", 2))
        self.bootstrap_task_limit = int(kwargs.get("bootstrap_task_limit", 3))
        self.prompt_history_turns = int(kwargs.get("prompt_history_turns", 10))
        reflection_flag = kwargs.get("use_llm_reflection")
        self.use_llm_reflection = self.use_llm_policy if reflection_flag is None else bool(reflection_flag)
        self.reflect_every = int(kwargs.get("reflect_every", 2))
        self.max_reflection_notes = int(kwargs.get("max_reflection_notes", 12))
        self.max_reflection_records = int(kwargs.get("max_reflection_records", 32))
        self.enable_score_replay = bool(kwargs.get("enable_score_replay", True))
        self.max_replay_steps = int(kwargs.get("max_replay_steps", 8))
        self.llm_force_json_output = bool(kwargs.get("llm_force_json_output", True))
        self.llm_json_retries = max(0, int(kwargs.get("llm_json_retries", 1)))
        self.enable_auto_unprogressed = bool(kwargs.get("enable_auto_unprogressed", True))
        self.task_stagnation_steps = max(3, int(kwargs.get("task_stagnation_steps", 8)))
        self.task_stagnation_max_unique_obs = max(
            1, int(kwargs.get("task_stagnation_max_unique_obs", 3))
        )
        self.task_stagnation_max_unique_actions = max(
            1, int(kwargs.get("task_stagnation_max_unique_actions", 3))
        )
        
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

        self.use_scratchpad = self.variant in {"structured", "rag", "option"}
        self.use_compression = self.variant in {"compressed", "structured", "rag", "option"}
        self.use_retrieval = kwargs.get("use_retrieval", self.variant == "rag")
        self.use_option_module = kwargs.get("use_option_module", self.variant == "option")

        compression_strategy = kwargs.get("compression_strategy")
        if compression_strategy is None:
            compression_strategy = "free_text" if self.variant == "compressed" else "structured"

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
            )
            if self.use_compression
            else None
        )
        self.retriever = Retriever(top_k=3) if self.use_retrieval else None
        self.option_module = OptionModule(enabled=self.use_option_module)
        
        # Initialize BacklogManager
        self.backlog = BacklogManager(
            max_waiting=int(kwargs.get("max_waiting_tasks", 5)),
            failure_threshold=int(kwargs.get("task_failure_threshold", 2)),
            failed_cooldown_steps=int(kwargs.get("task_failed_cooldown_steps", 6)),
        )

        self.history: List[Dict[str, Any]] = []
        self.raw_memory: List[str] = []
        self.compressed_history: List[Dict[str, Any]] = []
        self.reflection_notes: List[str] = []
        self.reflection_archive: List[Dict[str, Any]] = []
        self.llm_score_points: List[Dict[str, Any]] = []
        self.score_events: List[Dict[str, Any]] = []
        self.best_scoring_prefix: List[str] = []
        self.replay_prefix: List[str] = []
        self.replay_cursor = 0
        self.episode_scored = False
        self.last_known_score = 0
        self.episode_index = 0
        self.last_llm_thought = ""
        self.step_count = 0
        self.stagnation_task = ""
        self.stagnation_no_progress_steps = 0
        self.stagnation_obs_window: List[str] = []
        self.stagnation_action_window: List[str] = []

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
            f"PlanningAgent_v{self.variant}"
            f"_s{self.seed}"
            f"_ce{self.compress_every}"
            f"_llm{int(self.use_llm_policy)}"
        )

    @property
    def params(self):
        return {
            "agent_type": "planning",
            "memory_variant": self.variant,
            "seed": self.seed,
            "use_llm_policy": self.use_llm_policy,
            "llm_model": self.llm_model,
        }

    def reset(self, obs, info, env):
        self.step_count = 0
        self.episode_index += 1
        self.history = []
        self.raw_memory = []
        self.compressed_history = []
        self.episode_scored = False
        self.replay_cursor = 0
        self.replay_prefix = list(self.best_scoring_prefix)
        self.last_known_score = int((info or {}).get("score", 0))
        self.last_llm_thought = ""
        if self.scratchpad is not None:
            self.scratchpad.reset()
            self.scratchpad.update(obs)
        self.raw_memory.append(obs)
        self._reset_stagnation_state("")
        
        # Retain backlog state across episodes (semantic restart)
        self.backlog.on_episode_restart(self.step_count)

    def act(self, obs, reward, done, info):
        self.step_count += 1
        updates = {}
        self.raw_memory.append(obs)
        current_score = int((info or {}).get("score", self.last_known_score))
        score_delta = current_score - self.last_known_score
        self.last_known_score = current_score

        if self.scratchpad is not None:
            updates = self.scratchpad.update(obs)

        self._record_score_event(score_delta, obs, updates, current_score)

        # Evaluate last action's success/failure & update current task
        self.backlog.update_from_observation(obs, updates, self.step_count)
        self._update_replay_prefix_from_success(updates, score_delta)

        # Keep backlog alive even if LLM does not emit structured tasks.
        self._inject_bootstrap_tasks(obs, updates, info)
        self._maybe_auto_mark_unprogressed(obs, updates, score_delta)

        if self._should_compress():
            compressed = self._compress_now()
            if compressed is not None:
                self.compressed_history.append(compressed)

        retrieved = []
        if self.retriever is not None:
            retrieved = self.retriever.query(obs, self.compressed_history)

        action, new_tasks = self._generate_action_and_tasks(obs, info, updates, retrieved)

        # Inject new LLM-generated tasks if capacity allows
        if new_tasks:
            self.backlog.update_from_observation(obs, updates, self.step_count, llm_tasks=new_tasks)

        # Record action so the backlog can evaluate it on the next step
        self.backlog.record_action(action)

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
            "thinking": self.last_llm_thought,
            "nb_tokens": self.token_counter(text=obs),
        }
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
        recent = self.raw_memory[-self.max_context_items :]
        summary = " | ".join(line[:120].replace("\n", " ") for line in recent)
        return {
            "strategy": "free_text",
            "summary": summary or "No history yet.",
            "top_facts": [],
            "top_rules": [],
        }

    def _inject_bootstrap_tasks(
        self, obs: str, updates: Dict[str, Any], info: Dict[str, Any]
    ) -> None:
        if self.backlog.current_task() is not None:
            return
        admissible = list(info.get("admissible_commands") or [])
        tasks = self._filter_tasks_from_reflection(self._bootstrap_tasks(obs, updates, admissible))
        if not tasks:
            return
        self.backlog.update_from_observation(
            obs, updates, self.step_count, llm_tasks=tasks[: self.bootstrap_task_limit]
        )

    def _bootstrap_tasks(
        self, obs: str, updates: Dict[str, Any], admissible: List[str]
    ) -> List[str]:
        tasks: List[str] = []
        preconditions = list(updates.get("preconditions") or [])
        for item in preconditions[:2]:
            tasks.append(f"satisfy precondition: {item}")

        directions = self._admissible_directions(admissible)
        for direction in directions:
            if self._direction_recently_tried(direction):
                continue
            tasks.append(f"explore {direction}")
            if len(tasks) >= self.bootstrap_task_limit:
                break

        lowered_admissible = [cmd.lower() for cmd in admissible]
        if any(cmd.startswith("open ") for cmd in lowered_admissible):
            tasks.append("open unexplored doors or containers")
        if any(cmd.startswith("examine ") for cmd in lowered_admissible):
            tasks.append("inspect salient objects in the current area")
        if any(cmd.startswith("take ") for cmd in lowered_admissible):
            tasks.append("collect potentially useful items")
        if updates.get("failures"):
            tasks.append("recover from recent failed action with a different approach")
        if not tasks and ("goal" in obs.lower() or "objective" in obs.lower()):
            tasks.append("follow the explicit objective from the observation")

        return self._dedupe_text(tasks)[: self.bootstrap_task_limit]

    def _admissible_directions(self, admissible: List[str]) -> List[str]:
        directions: List[str] = []
        for cmd in admissible:
            lowered = cmd.strip().lower()
            if lowered in {"north", "south", "east", "west", "up", "down"}:
                directions.append(lowered)
            elif lowered.startswith("go "):
                tail = lowered.split(" ", 1)[1]
                if tail in {"north", "south", "east", "west", "up", "down"}:
                    directions.append(tail)
        return self._dedupe_text(directions)

    def _direction_recently_tried(self, direction: str, horizon: int = 8) -> bool:
        for item in self.history[-horizon:]:
            action = str(item.get("action", "")).strip().lower()
            if action in {direction, f"go {direction}"}:
                return True
        return False

    def _blocked_actions_for_observation(self, obs: str) -> Set[str]:
        current_sig = self._observation_signature(obs)
        action_counts: Dict[str, int] = {}

        for item in reversed(self.history):
            if self._observation_signature(str(item.get("observation", ""))) != current_sig:
                break
            action = str(item.get("action", "")).strip().lower()
            if not action:
                continue
            action_counts[action] = action_counts.get(action, 0) + 1

        return {action for action, count in action_counts.items() if count >= self.max_repeat_action}

    def _blocked_actions(self, obs: str) -> Set[str]:
        repeated = self._blocked_actions_for_observation(obs)
        risky = {value.strip().lower() for value in self.backlog.risky_actions() if str(value).strip()}
        terminal_failed = set(
            self.backlog.recent_failed_actions(
                limit=3,
                reasons=["terminal_loss", "burned_item", "episode_restart"],
            )
        )
        return repeated.union(risky).union(terminal_failed)

    def _reset_stagnation_state(self, task_name: str) -> None:
        self.stagnation_task = str(task_name or "").strip()
        self.stagnation_no_progress_steps = 0
        self.stagnation_obs_window = []
        self.stagnation_action_window = []

    @staticmethod
    def _most_common_nonempty(values: List[str]) -> str:
        counts: Dict[str, int] = {}
        best = ""
        best_count = 0
        for raw in values:
            value = str(raw or "").strip().lower()
            if not value:
                continue
            count = counts.get(value, 0) + 1
            counts[value] = count
            if count > best_count:
                best = value
                best_count = count
        return best

    def _is_progress_signal(
        self,
        obs: str,
        updates: Dict[str, Any],
        score_delta: int,
    ) -> bool:
        if int(score_delta) != 0:
            return True
        if updates.get("successes"):
            return True
        if updates.get("facts") or updates.get("rules") or updates.get("preconditions"):
            return True

        lowered = str(obs or "").lower()
        positive_tokens = [
            "you open",
            "you unlock",
            "you close",
            "you take",
            "you pick up",
            "you put",
            "you drop",
            "you enter",
            "you arrive",
            "you move",
            "you go ",
            "you roasted",
            "you fried",
            "you cooked",
            "you chopped",
            "you sliced",
            "you prepared",
            "you win",
        ]
        negative_tokens = [
            "you can't",
            "you cannot",
            "nothing happens",
            "not possible",
            "that doesn't work",
            "i don't understand",
            "unknown command",
            "no effect",
            "failed",
            "you lost",
            "game over",
            "you died",
            "would you like to quit",
        ]
        if any(token in lowered for token in positive_tokens) and not any(
            token in lowered for token in negative_tokens
        ):
            return True
        return False

    def _maybe_auto_mark_unprogressed(
        self,
        obs: str,
        updates: Dict[str, Any],
        score_delta: int,
    ) -> None:
        if not self.enable_auto_unprogressed:
            return

        task = self.backlog.current_task()
        if task is None:
            self._reset_stagnation_state("")
            return

        task_name = str(task.description or "").strip()
        if not task_name:
            self._reset_stagnation_state("")
            return

        if task_name != self.stagnation_task:
            self._reset_stagnation_state(task_name)

        progress = self._is_progress_signal(obs, updates, score_delta)
        obs_sig = self._observation_signature(obs)
        last_action = ""
        if self.history:
            last_action = str(self.history[-1].get("action", "")).strip().lower()

        self.stagnation_obs_window.append(obs_sig)
        self.stagnation_action_window.append(last_action)
        self.stagnation_obs_window = self.stagnation_obs_window[-self.task_stagnation_steps :]
        self.stagnation_action_window = self.stagnation_action_window[-self.task_stagnation_steps :]

        if progress:
            self.stagnation_no_progress_steps = 0
            return

        self.stagnation_no_progress_steps += 1
        if self.stagnation_no_progress_steps < self.task_stagnation_steps:
            return

        recent_obs = [value for value in self.stagnation_obs_window if value]
        recent_actions = [value for value in self.stagnation_action_window if value]
        unique_obs = len(set(recent_obs))
        unique_actions = len(set(recent_actions))
        if (
            unique_obs > self.task_stagnation_max_unique_obs
            and unique_actions > self.task_stagnation_max_unique_actions
        ):
            return

        common_action = self._most_common_nonempty(recent_actions)
        reason = self._clip_text(
            "auto_stagnation:"
            f"{self.stagnation_no_progress_steps}_steps_no_progress;"
            f"unique_obs={unique_obs};unique_actions={unique_actions}",
            180,
        )
        step_window = self.backlog.current_task_step_window(self.step_count) or {
            "start": max(1, self.step_count - self.task_stagnation_steps + 1),
            "end": self.step_count,
        }
        self.backlog.apply_llm_task_judgement(
            status="unprogressed",
            step=self.step_count,
            reason=reason,
            last_action=last_action,
        )
        self._append_reflection_entry(
            {
                "task": task_name,
                "status": "unprogressed",
                "reason": reason,
                "outcome": "auto-detected stagnation with no meaningful progress",
                "step_start": int(step_window.get("start", self.step_count)),
                "step_end": int(step_window.get("end", self.step_count)),
                "score_delta": 0,
                "triggered_restart": False,
                "repeatable_after_restart": False,
                "avoid_actions": [common_action] if common_action else [],
            }
        )

        if common_action and recent_actions.count(common_action) >= max(2, self.task_stagnation_steps // 2):
            self.backlog.add_risky_actions(
                [common_action],
                step=self.step_count,
                weight=1,
                reason="auto_stagnation",
                task=task_name,
            )

        if reason:
            self.reflection_notes.append(reason)
            self.reflection_notes = self._dedupe_text(self.reflection_notes)[
                -self.max_reflection_notes :
            ]

        next_task = self.backlog.export().get("current_task") or ""
        self._reset_stagnation_state(str(next_task))

    def _update_replay_prefix_from_success(self, updates: Dict[str, Any], score_delta: int) -> None:
        if self.episode_scored:
            return
        if not self.history:
            return
        successes = [str(x).strip().lower() for x in (updates.get("successes") or [])]
        if score_delta <= 0 and not any("score" in item for item in successes):
            return
        self.episode_scored = True
        prefix = [
            str(item.get("action", "")).strip()
            for item in self.history
            if str(item.get("action", "")).strip()
        ]
        prefix = prefix[: self.max_replay_steps]
        if prefix and (
            not self.best_scoring_prefix or len(prefix) < len(self.best_scoring_prefix)
        ):
            self.best_scoring_prefix = list(prefix)
            self.replay_prefix = list(prefix)
            self.replay_cursor = 0

    def _record_score_event(
        self,
        score_delta: int,
        obs: str,
        updates: Dict[str, Any],
        current_score: int,
    ) -> None:
        if score_delta == 0:
            return
        last_action = ""
        if self.history:
            last_action = str(self.history[-1].get("action", "")).strip()
        event = {
            "step": int(self.step_count),
            "episode": int(self.episode_index),
            "delta": int(score_delta),
            "score": int(current_score),
            "action": last_action,
            "signal": self._score_signal_text(obs, updates),
        }
        self.score_events.append(event)
        self.score_events = self.score_events[-64:]

        point = {
            "step": int(self.step_count),
            "episode": int(self.episode_index),
            "kind": "gain" if score_delta > 0 else "loss",
            "point": self._clip_text(
                f"{last_action or 'unknown action'} | delta {score_delta:+d}",
                120,
            ),
        }
        self._append_score_point(point)

    @staticmethod
    def _score_signal_text(obs: str, updates: Dict[str, Any]) -> str:
        lines = [line.strip() for line in str(obs or "").splitlines() if line.strip()]
        for line in lines:
            lowered = line.lower()
            if "score" in lowered or "you lost" in lowered or "you won" in lowered:
                return line[:160]
        failures = [str(x).strip() for x in (updates.get("failures") or []) if str(x).strip()]
        if failures:
            return ", ".join(failures[:3])[:160]
        return ""

    def _append_score_point(self, entry: Dict[str, Any]) -> None:
        point_text = str(entry.get("point", "")).strip().lower()
        kind = str(entry.get("kind", "")).strip().lower()
        step = int(entry.get("step", self.step_count))
        episode = int(entry.get("episode", self.episode_index))
        if not point_text or kind not in {"gain", "loss"}:
            return
        for idx in range(len(self.llm_score_points) - 1, -1, -1):
            prev = self.llm_score_points[idx]
            if (
                str(prev.get("point", "")).strip().lower() == point_text
                and str(prev.get("kind", "")).strip().lower() == kind
                and int(prev.get("episode", -1)) == episode
            ):
                self.llm_score_points[idx] = {
                    **prev,
                    "step": step,
                }
                return
        self.llm_score_points.append(
            {
                "step": step,
                "episode": episode,
                "kind": kind,
                "point": self._clip_text(str(entry.get("point", "")), 120),
            }
        )
        self.llm_score_points = self.llm_score_points[-self.max_reflection_notes :]

    def _maybe_replay_action(
        self,
        admissible: List[str],
        blocked_actions: Set[str],
    ) -> Optional[str]:
        if not self.enable_score_replay or self.episode_scored:
            return None
        if not self.replay_prefix or self.replay_cursor >= len(self.replay_prefix):
            return None
        if self.step_count > max(self.max_replay_steps + 2, 4):
            return None

        while self.replay_cursor < len(self.replay_prefix):
            candidate = self.replay_prefix[self.replay_cursor].strip()
            self.replay_cursor += 1
            if not candidate:
                continue
            normalized = candidate.lower()
            if normalized in blocked_actions:
                continue
            if admissible and candidate not in admissible:
                self.replay_cursor = len(self.replay_prefix)
                return None
            return candidate
        return None

    def _peek_replay_action(
        self,
        admissible: List[str],
        blocked_actions: Set[str],
    ) -> Optional[str]:
        if not self.enable_score_replay or self.episode_scored:
            return None
        if not self.replay_prefix or self.replay_cursor >= len(self.replay_prefix):
            return None
        if self.step_count > max(self.max_replay_steps + 2, 4):
            return None

        cursor = self.replay_cursor
        while cursor < len(self.replay_prefix):
            candidate = self.replay_prefix[cursor].strip()
            cursor += 1
            if not candidate:
                continue
            normalized = candidate.lower()
            if normalized in blocked_actions:
                continue
            if admissible and candidate not in admissible:
                return None
            return candidate
        return None

    def _generate_action_and_tasks(
        self,
        obs: str,
        info: Dict[str, Any],
        updates: Dict[str, Any],
        retrieved: List[Dict[str, Any]],
    ) -> Tuple[str, List[str]]:
        admissible = list(info.get("admissible_commands") or [])
        blocked_actions = self._blocked_actions(obs)
        self.last_llm_thought = ""
        replay_action: Optional[str] = None
        if self.use_llm_policy:
            replay_action = self._peek_replay_action(admissible, blocked_actions)
        else:
            replay_action = self._maybe_replay_action(admissible, blocked_actions)
            if replay_action:
                return replay_action, []

        option = self.option_module.choose_option(obs)
        option_candidates = self.option_module.option_actions(option)
        heuristics = self._heuristic_actions(obs, updates, retrieved)
        
        # Prioritize candidates relevant to the current backlog task
        candidates = self.backlog.prioritize_actions(
            admissible, option_candidates + heuristics
        )
        if blocked_actions:
            candidates = [
                candidate
                for candidate in candidates
                if candidate.strip().lower() not in blocked_actions
            ]
        if replay_action and replay_action.strip().lower() not in blocked_actions:
            candidates = [replay_action] + candidates

        new_tasks: List[str] = []

        if self.use_llm_policy:
            llm_decision = self._llm_decide_step(
                obs, updates, retrieved, admissible, candidates
            )
            self.last_llm_thought = str(llm_decision.get("thought", "")).strip()
            llm_tasks = list(llm_decision.get("new_tasks") or [])
            if llm_tasks:
                new_tasks.extend(llm_tasks[: self.bootstrap_task_limit])
            self._apply_llm_decision(llm_decision, updates)

            llm_action = str(llm_decision.get("action", "")).strip()
            if llm_action:
                action = self._avoid_blocked_action(
                    llm_action, admissible, candidates, blocked_actions
                )
                if action:
                    return action, self._dedupe_text(new_tasks)[: self.bootstrap_task_limit]

        if self.backlog.current_task() is None or updates.get("failures"):
            new_tasks.extend(self._bootstrap_tasks(obs, updates, admissible))
        new_tasks = self._filter_tasks_from_reflection(new_tasks)[: self.bootstrap_task_limit]

        # Fallback to prioritized candidates
        if admissible:
            admissible_set = set(admissible)
            for candidate in candidates:
                if candidate in admissible_set:
                    return candidate, new_tasks
            return self._diversifying_admissible(admissible, blocked_actions, candidates), new_tasks

        for candidate in candidates:
            if candidate and candidate.strip().lower() not in blocked_actions:
                return candidate, new_tasks

        for fallback in self.default_actions:
            if fallback.strip().lower() not in blocked_actions:
                return fallback, new_tasks
        return str(self.rng.choice(self.default_actions)), new_tasks

    def _heuristic_actions(
        self, obs: str, updates: Dict[str, Any], retrieved: List[Dict[str, Any]]
    ) -> List[str]:
        lowered = (obs or "").lower()
        actions: List[str] = []

        if any(x in lowered for x in ["what do i do", "stuck", "can't"]):
            actions.append("help")
        if "locked" in lowered:
            actions.extend(["examine door", "unlock door", "open door"])
        if "inventory" not in lowered and self.step_count % 9 == 0:
            actions.append("inventory")

        nouns = re.findall(r"\b[a-z]{4,}\b", lowered)
        if nouns:
            focus = nouns[0]
            actions.extend([f"examine {focus}", f"take {focus}"])

        if updates.get("failures") or retrieved:
            actions.append("look")

        actions.extend(["north", "south", "east", "west"])
        return actions

    def _llm_decide_step(
        self,
        obs: str,
        updates: Dict[str, Any],
        retrieved: List[Dict[str, Any]],
        admissible: List[str],
        candidates: List[str],
    ) -> Dict[str, Any]:
        decision = {
            "thought": "",
            "task_status": "none",
            "task_reason": "",
            "next_task": "",
            "new_tasks": [],
            "notes": [],
            "reflection_entries": [],
            "avoid_actions": [],
            "score_points": [],
            "action": "",
        }
        api_key = self.llm_api_key or os.getenv(self.llm_api_key_env)
        if not api_key:
            return decision

        try:
            prompt = self._build_llm_prompt(obs, updates, retrieved, admissible, candidates)
            response_json = self._query_llm(
                prompt,
                api_key,
                strict_json=self.llm_force_json_output,
                max_tokens=480,
            )
            content = self._extract_content(response_json)
            obj = self._extract_json_object(content)
            if not obj and self.llm_json_retries > 0:
                repair_prompt = self._build_json_repair_prompt(content, admissible)
                for _ in range(self.llm_json_retries):
                    try:
                        response_json = self._query_llm(
                            repair_prompt,
                            api_key,
                            strict_json=True,
                            max_tokens=380,
                        )
                    except Exception:
                        break
                    content = self._extract_content(response_json)
                    obj = self._extract_json_object(content)
                    if obj:
                        break
                    repair_prompt = self._build_json_repair_prompt(content, admissible)
            if not obj:
                raw_action = self._sanitize_action(content)
                if admissible and raw_action:
                    raw_action = self._align_to_admissible(raw_action, admissible)
                decision["action"] = raw_action
                return decision
            return self._parse_llm_decision(obj, admissible)
        except Exception:
            return decision

    def _parse_llm_decision(
        self,
        obj: Dict[str, Any],
        admissible: List[str],
    ) -> Dict[str, Any]:
        thought = self._clip_text(str(obj.get("thought", "")).strip(), 220)

        backlog_update = obj.get("backlog_update", {})
        task_update = obj.get("task_update", {})
        if not isinstance(backlog_update, dict) and isinstance(task_update, dict):
            backlog_update = task_update

        new_tasks: List[str] = []
        task_status = "none"
        task_reason = ""
        next_task = ""
        if isinstance(backlog_update, dict):
            task_status = str(
                backlog_update.get("current_task_status")
                or backlog_update.get("status")
                or ""
            ).strip().lower()
            task_reason = self._clip_text(
                str(
                    backlog_update.get("current_task_reason")
                    or backlog_update.get("reason")
                    or ""
                ).strip(),
                160,
            )
            next_task = self._clip_text(
                str(
                    backlog_update.get("next_current_task")
                    or backlog_update.get("next_task")
                    or ""
                ).strip(),
                160,
            )
            new_tasks.extend(self._coerce_text_list(backlog_update.get("tasks_to_add", [])))
            new_tasks.extend(self._coerce_text_list(backlog_update.get("new_tasks", [])))
        elif backlog_update:
            task_status = str(backlog_update).strip().lower()

        if isinstance(task_update, dict):
            if not task_status:
                task_status = str(task_update.get("status", "")).strip().lower()
            if not task_reason:
                task_reason = self._clip_text(str(task_update.get("reason", "")).strip(), 160)
            if not next_task:
                next_task = self._clip_text(str(task_update.get("next_task", "")).strip(), 160)
            new_tasks.extend(self._coerce_text_list(task_update.get("new_tasks", [])))
        elif task_update and not task_status:
            task_status = str(task_update).strip().lower()

        if not next_task:
            next_task = self._clip_text(
                str(obj.get("next_current_task") or obj.get("next_task") or "").strip(),
                160,
            )

        new_tasks.extend(self._coerce_text_list(obj.get("new_tasks", [])))
        new_tasks = self._filter_tasks_from_reflection(new_tasks)[: self.bootstrap_task_limit]
        task_status = self._normalize_task_status(task_status)

        reflection = obj.get("reflection_update", {})
        if not isinstance(reflection, dict):
            reflection = obj.get("reflection", {})
        notes: List[str] = []
        avoid_actions: List[str] = []
        score_points: List[Dict[str, Any]] = []
        reflection_entries: List[Dict[str, Any]] = []
        if isinstance(reflection, dict):
            notes = self._dedupe_text(self._coerce_text_list(reflection.get("notes", [])))[:2]
            avoid_actions = self._dedupe_text(
                self._coerce_text_list(reflection.get("avoid_actions", []))
            )[:4]
            score_points = self._parse_score_points(reflection.get("score_points", []))
            reflection_entries = self._parse_reflection_entries(
                reflection.get("entries", reflection.get("archive", [])),
                fallback_status=task_status,
                fallback_reason=task_reason,
            )
        if not notes:
            notes = self._dedupe_text(self._coerce_text_list(obj.get("notes", [])))[:2]
        if not avoid_actions:
            avoid_actions = self._dedupe_text(self._coerce_text_list(obj.get("avoid_actions", [])))[:4]
        if not score_points:
            score_points = self._parse_score_points(obj.get("score_points", []))

        raw_action = self._sanitize_action(str(obj.get("action", "")))
        action = raw_action
        if action and admissible:
            action = self._align_to_admissible(action, admissible)

        return {
            "thought": thought,
            "task_status": task_status,
            "task_reason": task_reason,
            "next_task": next_task,
            "new_tasks": new_tasks,
            "notes": notes,
            "reflection_entries": reflection_entries,
            "avoid_actions": avoid_actions,
            "score_points": score_points,
            "action": action,
        }

    def _parse_score_points(self, raw_points: Any) -> List[Dict[str, Any]]:
        values = raw_points
        if not isinstance(values, list):
            values = [values]
        parsed: List[Dict[str, Any]] = []
        for value in values[:4]:
            if isinstance(value, dict):
                kind = str(value.get("kind") or value.get("type") or "").strip().lower()
                point = str(value.get("point") or value.get("text") or "").strip()
                step = self._safe_int(value.get("step"), self.step_count)
                episode = self._safe_int(value.get("episode"), self.episode_index)
            else:
                text = str(value).strip()
                if not text:
                    continue
                lowered = text.lower()
                kind = "loss" if ("loss" in lowered or lowered.startswith("-")) else "gain"
                point = text
                step = self.step_count
                episode = self.episode_index
            if kind not in {"gain", "loss"} or not point:
                continue
            parsed.append(
                {
                    "step": int(step),
                    "episode": int(episode),
                    "kind": kind,
                    "point": self._clip_text(point, 120),
                }
            )
        return parsed

    def _parse_reflection_entries(
        self,
        raw_entries: Any,
        fallback_status: str,
        fallback_reason: str,
    ) -> List[Dict[str, Any]]:
        values = raw_entries
        if not isinstance(values, list):
            values = [values]

        task = self.backlog.current_task()
        default_task = task.description if task is not None else ""
        step_window = self.backlog.current_task_step_window(self.step_count) or {
            "start": self.step_count,
            "end": self.step_count,
        }
        entries: List[Dict[str, Any]] = []
        for value in values[:3]:
            if isinstance(value, dict):
                status = self._normalize_task_status(
                    str(value.get("status") or fallback_status or "").strip().lower()
                )
                if status not in {"completed", "failed", "unprogressed"}:
                    continue
                task_text = self._clip_text(
                    str(value.get("task") or value.get("task_name") or default_task).strip(),
                    160,
                )
                reason = self._clip_text(
                    str(value.get("reason") or fallback_reason or "").strip(),
                    220,
                )
                outcome = self._clip_text(
                    str(value.get("outcome") or value.get("result") or "").strip(),
                    200,
                )
                step_start = self._safe_int(value.get("step_start"), step_window["start"])
                step_end = self._safe_int(value.get("step_end"), step_window["end"])
                if step_end < step_start:
                    step_start, step_end = step_end, step_start
                score_delta = value.get("score_delta", None)
                score_delta_int: Optional[int] = None
                if score_delta is not None and str(score_delta).strip() != "":
                    score_delta_int = self._safe_int(score_delta, 0)
                entries.append(
                    {
                        "task": task_text,
                        "status": status,
                        "reason": reason,
                        "outcome": outcome,
                        "step_start": int(step_start),
                        "step_end": int(step_end),
                        "score_delta": score_delta_int,
                        "triggered_restart": self._coerce_bool(value.get("triggered_restart")),
                        "repeatable_after_restart": self._coerce_bool(
                            value.get("repeatable_after_restart")
                        ),
                        "avoid_actions": self._coerce_text_list(value.get("avoid_actions", []))[:4],
                    }
                )
                continue

            text = str(value).strip()
            if not text:
                continue
            status = self._normalize_task_status(fallback_status)
            if status not in {"completed", "failed", "unprogressed"}:
                continue
            entries.append(
                {
                    "task": default_task,
                    "status": status,
                    "reason": self._clip_text(text, 220),
                    "outcome": "",
                    "step_start": int(step_window["start"]),
                    "step_end": int(step_window["end"]),
                    "score_delta": None,
                    "triggered_restart": False,
                    "repeatable_after_restart": False,
                    "avoid_actions": [],
                }
            )
        return entries

    def _apply_llm_decision(
        self,
        decision: Dict[str, Any],
        updates: Dict[str, Any],
    ) -> None:
        status = self._normalize_task_status(str(decision.get("task_status", "")).strip().lower())
        reason = self._clip_text(str(decision.get("task_reason", "")).strip(), 220)
        previous_task_obj = self.backlog.current_task()
        previous_task = previous_task_obj.description if previous_task_obj is not None else ""
        previous_window = self.backlog.current_task_step_window(self.step_count) or {
            "start": self.step_count,
            "end": self.step_count,
        }
        if status in {"in_progress", "completed", "failed", "unprogressed", "none"}:
            last_action = ""
            if self.history:
                last_action = str(self.history[-1].get("action", "")).strip()
            self.backlog.apply_llm_task_judgement(
                status=status,
                step=self.step_count,
                reason=reason,
                last_action=last_action,
            )

        next_task = self._clip_text(str(decision.get("next_task", "")).strip(), 160)
        if next_task:
            self.backlog.set_current_task(next_task, step=self.step_count, allow_create=True)

        if not self._should_capture_reflection(updates, decision):
            return

        notes = self._dedupe_text(list(decision.get("notes", []) or []))[:2]
        if notes:
            self.reflection_notes.extend(notes)
            self.reflection_notes = self._dedupe_text(self.reflection_notes)[-self.max_reflection_notes :]

        for point in list(decision.get("score_points", []) or []):
            if isinstance(point, dict):
                self._append_score_point(point)

        entries = list(decision.get("reflection_entries", []) or [])
        if not entries and status in {"completed", "failed", "unprogressed"} and previous_task:
            entries = [
                self._synthesize_reflection_entry(
                    status=status,
                    reason=reason,
                    updates=updates,
                    task=previous_task,
                    step_window=previous_window,
                )
            ]
        for entry in entries:
            if isinstance(entry, dict):
                self._append_reflection_entry(entry)

        entry_avoid_actions: List[str] = []
        for entry in entries:
            if isinstance(entry, dict):
                entry_avoid_actions.extend(self._coerce_text_list(entry.get("avoid_actions", [])))
        avoid_actions = self._dedupe_text(
            list(decision.get("avoid_actions", []) or []) + entry_avoid_actions
        )[:4]
        if avoid_actions:
            weight = 2 if "terminal_loss" in (updates.get("failures") or []) else 1
            self.backlog.add_risky_actions(
                avoid_actions,
                step=self.step_count,
                weight=weight,
                reason="llm_step_reflection",
                task=self.backlog.export().get("current_task"),
            )

    def _synthesize_reflection_entry(
        self,
        status: str,
        reason: str,
        updates: Dict[str, Any],
        task: str,
        step_window: Dict[str, int],
    ) -> Dict[str, Any]:
        failures = [str(item).strip() for item in (updates.get("failures") or []) if str(item).strip()]
        latest_delta = 0
        if self.score_events and int(self.score_events[-1].get("step", -1)) == self.step_count:
            latest_delta = int(self.score_events[-1].get("delta", 0))
        outcome_parts: List[str] = []
        if latest_delta != 0:
            outcome_parts.append(f"score_delta={latest_delta:+d}")
        if failures:
            outcome_parts.append("failures=" + ", ".join(failures[:3]))
        if not outcome_parts:
            outcome_parts.append("no_immediate_score_change")
        triggered_restart = any(
            reason_key in {item.lower() for item in failures}
            for reason_key in ["terminal_loss", "episode_restart"]
        )
        return {
            "task": self._clip_text(task, 160),
            "status": status,
            "reason": self._clip_text(reason or "status updated by llm", 220),
            "outcome": self._clip_text("; ".join(outcome_parts), 200),
            "step_start": int(step_window.get("start", self.step_count)),
            "step_end": int(step_window.get("end", self.step_count)),
            "score_delta": latest_delta if latest_delta != 0 else None,
            "triggered_restart": bool(triggered_restart),
            "repeatable_after_restart": bool(latest_delta > 0),
            "avoid_actions": [],
        }

    def _append_reflection_entry(self, raw_entry: Dict[str, Any]) -> None:
        task = self._clip_text(str(raw_entry.get("task", "")).strip(), 160)
        status = self._normalize_task_status(str(raw_entry.get("status", "")).strip().lower())
        if status not in {"completed", "failed", "unprogressed"} or not task:
            return
        entry = {
            "step": int(self.step_count),
            "episode": int(self.episode_index),
            "task": task,
            "status": status,
            "reason": self._clip_text(str(raw_entry.get("reason", "")).strip(), 220),
            "outcome": self._clip_text(str(raw_entry.get("outcome", "")).strip(), 200),
            "step_start": self._safe_int(raw_entry.get("step_start"), self.step_count),
            "step_end": self._safe_int(raw_entry.get("step_end"), self.step_count),
            "score_delta": raw_entry.get("score_delta", None),
            "triggered_restart": self._coerce_bool(raw_entry.get("triggered_restart")),
            "repeatable_after_restart": self._coerce_bool(
                raw_entry.get("repeatable_after_restart")
            ),
            "avoid_actions": self._coerce_text_list(raw_entry.get("avoid_actions", []))[:4],
        }

        dedupe_key = (
            entry["task"].lower(),
            entry["status"],
            int(entry["step_start"]),
            int(entry["step_end"]),
        )
        for idx in range(len(self.reflection_archive) - 1, -1, -1):
            prev = self.reflection_archive[idx]
            prev_key = (
                str(prev.get("task", "")).lower(),
                str(prev.get("status", "")),
                self._safe_int(prev.get("step_start"), -1),
                self._safe_int(prev.get("step_end"), -1),
            )
            if prev_key == dedupe_key:
                self.reflection_archive[idx] = entry
                break
        else:
            self.reflection_archive.append(entry)

        self.reflection_archive = self.reflection_archive[-self.max_reflection_records :]

    def _should_capture_reflection(
        self,
        updates: Dict[str, Any],
        decision: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if not self.use_llm_reflection:
            return False
        if decision:
            if decision.get("reflection_entries"):
                return True
            status = self._normalize_task_status(str(decision.get("task_status", "")).strip().lower())
            if status in {"completed", "failed", "unprogressed"}:
                return True
            if decision.get("notes") or decision.get("score_points") or decision.get("avoid_actions"):
                return True
        if updates.get("failures"):
            return True
        return self.step_count % max(self.reflect_every, 1) == 0

    def _build_llm_prompt(
        self,
        obs: str,
        updates: Dict[str, Any],
        retrieved: List[Dict[str, Any]],
        admissible: List[str],
        candidates: List[str],
    ) -> str:
        snapshot = self._build_memory_snapshot(obs)
        compressed = self.compressed_history[-2:]
        admissible_slice = admissible[:70]
        candidate_slice = candidates[:28]
        blocked_actions = sorted(self._blocked_actions(obs))
        recent_turns = self._recent_turns_for_prompt(limit=self.prompt_history_turns)

        backlog_export = self.backlog.export()
        current_task = backlog_export.get("current_task", None)
        current_window = self.backlog.current_task_step_window(self.step_count)
        backlog_view = {
            "current_task": current_task,
            "current_task_step_window": current_window,
            "waiting": backlog_export.get("waiting", [])[:6],
            "in_progress": backlog_export.get("in_progress", [])[:2],
            "failed": backlog_export.get("failed", [])[:6],
            "unprogressed": backlog_export.get("unprogressed", [])[:6],
            "completed_recent": backlog_export.get("completed", [])[-6:],
            "risky_actions": backlog_export.get("risky_actions", [])[:6],
            "recent_failures": backlog_export.get("failure_events", [])[-3:],
            "task_table": backlog_export.get("tasks", [])[:8],
        }
        score_view = {
            "episode_index": self.episode_index,
            "current_score": self.last_known_score,
            "recent_score_events": self.score_events[-6:],
            "known_score_points": self.llm_score_points[-8:],
            "best_scoring_prefix": self.best_scoring_prefix[: self.max_replay_steps],
        }
        reflection_view = {
            "notes": self.reflection_notes[-8:],
            "archive": self.reflection_archive[-8:],
        }

        prompt = (
            "You are a planning + reflection policy for ANY interactive text environment.\n"
            "You must run one internal chain-of-thought pass, then output STRICT JSON only.\n"
            "Output must be valid JSON parseable by json.loads (double quotes, no trailing commas).\n"
            "Every step must follow 4 phases in this order:\n"
            "1) Analyze old backlog + history(step, observation, action) + current observation + old reflection.\n"
            "2) Update backlog: maintain waiting tasks and one focused current task. If evidence shows current task is completed, failed, or unprogressed, mark it explicitly. Then choose next current task from waiting/new tasks, or propose a new likely-effective task.\n"
            "3) Update reflection: when a task was marked completed/failed/unprogressed, append why, result (score gain/loss, restart/loss risk, low feasibility), and step range. Track repeatable score opportunities after restarts and avoid failure patterns.\n"
            "4) Output one environment action that advances the current task.\n\n"
            "Return exactly this JSON schema:\n"
            "{\n"
            '  "thought": "concise reasoning summary (<= 1 sentence)",\n'
            '  "backlog_update": {\n'
            '    "current_task_status": "in_progress|completed|failed|unprogressed|none",\n'
            '    "current_task_reason": "evidence-based reason",\n'
            '    "tasks_to_add": ["0-3 candidate tasks"],\n'
            '    "next_current_task": "task to focus next (optional if unchanged)"\n'
            "  },\n"
            '  "reflection_update": {\n'
            '    "notes": ["0-2 concise principles"],\n'
            '    "entries": [\n'
            '      {"task":"...", "status":"completed|failed|unprogressed", "reason":"...", "outcome":"...", "step_start":0, "step_end":0, "score_delta":0, "triggered_restart":false, "repeatable_after_restart":false}\n'
            "    ],\n"
            '    "score_points": [{"step":0, "kind":"gain|loss", "point":"..."}],\n'
            '    "avoid_actions": ["0-4 risky commands to avoid"]\n'
            "  },\n"
            '  "action": "exactly one command copied from admissible commands",\n'
            '  "new_tasks": ["optional extra tasks, merged with backlog_update.tasks_to_add"]\n'
            "}\n\n"
            "Hard rules:\n"
            "- Never output markdown or extra text.\n"
            "- The action MUST exactly match one admissible command.\n"
            "- Focus on current_task until evidence supports completed/failed/unprogressed.\n"
            "- If current_task is None, propose at least one useful generic task and choose one as next_current_task.\n"
            "- If status is completed/failed/unprogressed, reflection_update.entries should include at least one matching archive item.\n"
            "- Use unprogressed when repeated attempts show no meaningful progress and success is unlikely with current approach.\n"
            "- Use reflection_update.score_points only for newly observed score gain/loss clues.\n"
            "- Score opportunities may repeat after episode restart; do not treat repeats as contradictions.\n"
            "- Avoid blocked/risky repeated actions unless there is clear new evidence.\n"
            "- Avoid proposing tasks too similar to failed/unprogressed archive entries unless you explicitly changed strategy.\n\n"
            f"Observation:\n{self._clip_text(obs, 1800)}\n\n"
            f"Current parsed updates:\n{json.dumps(updates, ensure_ascii=True)}\n\n"
            f"Backlog summary:\n{json.dumps(backlog_view, ensure_ascii=True)}\n\n"
            f"Reflection memory:\n{json.dumps(reflection_view, ensure_ascii=True)}\n\n"
            f"Score summary:\n{json.dumps(score_view, ensure_ascii=True)}\n\n"
            f"Memory snapshot:\n{json.dumps(snapshot, ensure_ascii=True)}\n\n"
            f"Compressed memory (recent):\n{json.dumps(compressed, ensure_ascii=True)}\n\n"
            f"Retrieved memory:\n{json.dumps(retrieved[:2], ensure_ascii=True)}\n\n"
            f"Recent turns:\n{json.dumps(recent_turns, ensure_ascii=True)}\n\n"
            f"Blocked repeated actions:\n{json.dumps(blocked_actions, ensure_ascii=True)}\n\n"
            f"Heuristic candidates:\n{json.dumps(candidate_slice, ensure_ascii=True)}\n\n"
            f"Admissible commands:\n{json.dumps(admissible_slice, ensure_ascii=True)}\n"
        )
        return self._clip_text(prompt, self.max_prompt_chars)

    def _build_json_repair_prompt(self, raw_output: str, admissible: List[str]) -> str:
        admissible_slice = admissible[:70]
        prompt = (
            "Rewrite the following model output into ONE valid JSON object only.\n"
            "Do not add markdown or explanations.\n"
            "Required top-level keys: thought, backlog_update, reflection_update, action, new_tasks.\n"
            "backlog_update keys: current_task_status, current_task_reason, tasks_to_add, next_current_task.\n"
            "reflection_update keys: notes, entries, score_points, avoid_actions.\n"
            "action must be exactly one command from admissible commands.\n"
            f"Admissible commands:\n{json.dumps(admissible_slice, ensure_ascii=True)}\n\n"
            "Original output:\n"
            f"{self._clip_text(raw_output, 2400)}\n"
        )
        return self._clip_text(prompt, self.max_prompt_chars)

    def _recent_turns_for_prompt(self, limit: int = 10) -> List[Dict[str, Any]]:
        turns: List[Dict[str, Any]] = []
        for item in self.history[-max(1, limit) :]:
            step = int(item.get("step", 0))
            turns.append(
                {
                    "step": step,
                    "observation": self._clip_text(str(item.get("observation", "")).strip(), 260),
                    "action": str(item.get("action", "")).strip(),
                    "observation_sig": self._observation_signature(str(item.get("observation", ""))),
                }
            )
        return turns

    def _build_memory_snapshot(self, obs: str) -> Dict[str, Any]:
        if self.scratchpad is None:
            return {}
        exported = self.scratchpad.export()
        importance = exported.get("importance", {})
        return {
            "facts": self._rank_memory_entries(
                exported.get("facts", []), importance, obs, self.max_context_items
            ),
            "rules": self._rank_memory_entries(
                exported.get("rules", []), importance, obs, self.max_context_items
            ),
            "preconditions": self._rank_memory_entries(
                exported.get("preconditions", []), importance, obs, min(8, self.max_context_items)
            ),
            "recent_failures": list(exported.get("failures", [])[-6:]),
            "recent_successes": list(exported.get("successes", [])[-6:]),
            "strategy_notes": self.reflection_notes[-6:],
            "reflection_archive": self.reflection_archive[-6:],
            "score_reflections": self.llm_score_points[-6:],
        }

    def _rank_memory_entries(
        self,
        items: List[str],
        importance: Dict[str, float],
        observation: str,
        limit: int,
    ) -> List[str]:
        if not items:
            return []
        obs_tokens = set(re.findall(r"[a-z0-9_]+", (observation or "").lower()))
        scored: List[Tuple[float, str]] = []
        size = float(max(len(items), 1))
        for idx, item in enumerate(items):
            text = str(item).strip()
            if not text:
                continue
            tokens = set(re.findall(r"[a-z0-9_]+", text.lower()))
            overlap = float(len(tokens & obs_tokens))
            recency_bonus = float(idx + 1) / size
            base_importance = float(importance.get(text, 0.0))
            base_importance += float(importance.get(f"need:{text}", 0.0))
            score = base_importance + (2.0 * overlap) + (0.35 * recency_bonus)
            scored.append((score, text))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return self._dedupe_text([text for _, text in scored])[:limit]

    @staticmethod
    def _clip_text(text: str, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        value = str(text or "")
        if len(value) <= max_chars:
            return value
        return value[: max_chars - 3] + "..."

    @staticmethod
    def _observation_signature(text: str) -> str:
        tokens = re.findall(r"[a-z0-9_]+", str(text or "").lower())
        return " ".join(tokens[:120])

    @staticmethod
    def _dedupe_text(values: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for value in values:
            clean = str(value).strip()
            key = clean.lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(clean)
        return out

    @staticmethod
    def _safe_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        text = str(value or "").strip().lower()
        return text in {"1", "true", "yes", "y"}

    @staticmethod
    def _normalize_task_status(raw_status: str) -> str:
        status = str(raw_status or "").strip().lower()
        if status in {"keep", "continue", "ongoing"}:
            return "in_progress"
        if status in {"complete", "completed", "done", "success"}:
            return "completed"
        if status in {"failed", "fail", "abandoned", "abandon"}:
            return "failed"
        if status in {"unprogressed", "stalled", "stagnant", "no_progress", "stuck", "unproductive"}:
            return "unprogressed"
        if status in {"none", ""}:
            return "none"
        if status in {"in_progress", "inprogress", "progressing"}:
            return "in_progress"
        return "none"

    def _coerce_text_list(self, raw_values: Any) -> List[str]:
        values = raw_values
        if values is None:
            return []
        if not isinstance(values, list):
            values = [values]
        out: List[str] = []
        for value in values:
            if isinstance(value, dict):
                text = str(
                    value.get("task")
                    or value.get("description")
                    or value.get("text")
                    or ""
                ).strip()
            else:
                text = str(value).strip()
            if not text:
                continue
            out.append(self._clip_text(" ".join(text.split()), 160))
        return self._dedupe_text(out)

    @staticmethod
    def _task_text_similarity(a: str, b: str) -> float:
        tokens_a = set(re.findall(r"[a-z0-9_]+", str(a or "").lower()))
        tokens_b = set(re.findall(r"[a-z0-9_]+", str(b or "").lower()))
        if not tokens_a or not tokens_b:
            return 0.0
        overlap = len(tokens_a & tokens_b)
        denom = float(min(len(tokens_a), len(tokens_b)))
        return overlap / denom if denom > 0 else 0.0

    def _filter_tasks_from_reflection(self, raw_tasks: List[str]) -> List[str]:
        tasks = self._coerce_text_list(raw_tasks)
        if not tasks:
            return []
        blocked_tasks = [
            str(entry.get("task", "")).strip()
            for entry in self.reflection_archive[-16:]
            if str(entry.get("status", "")).strip().lower() in {"failed", "unprogressed"}
        ]
        if not blocked_tasks:
            return self._dedupe_text(tasks)

        filtered: List[str] = []
        for task in tasks:
            lowered = task.lower()
            if "recover" in lowered or "alternative" in lowered:
                filtered.append(task)
                continue
            matched_bad = any(
                self._task_text_similarity(task, blocked) >= 0.8 for blocked in blocked_tasks
            )
            if matched_bad:
                continue
            filtered.append(task)
        return self._dedupe_text(filtered)

    def _avoid_blocked_action(
        self,
        action: str,
        admissible: List[str],
        candidates: List[str],
        blocked_actions: Set[str],
    ) -> Optional[str]:
        if not action:
            return None
        normalized = action.strip().lower()
        if normalized in blocked_actions:
            if admissible:
                return self._diversifying_admissible(admissible, blocked_actions, candidates)
            for candidate in candidates:
                if candidate.strip().lower() not in blocked_actions:
                    return candidate
            return None
        return action

    def _diversifying_admissible(
        self, admissible: List[str], blocked_actions: Set[str], candidates: List[str]
    ) -> str:
        admissible_set = set(admissible)
        for candidate in candidates:
            if candidate in admissible_set and candidate.strip().lower() not in blocked_actions:
                return candidate

        blocked_verbs = {value.split()[0] for value in blocked_actions if value.split()}
        for cmd in admissible:
            lowered = cmd.strip().lower()
            if lowered in blocked_actions:
                continue
            verb = lowered.split()[0] if lowered.split() else ""
            if blocked_verbs and verb and verb not in blocked_verbs:
                return cmd

        for cmd in admissible:
            if cmd.strip().lower() not in blocked_actions:
                return cmd

        return str(self.rng.choice(admissible))

    def _query_llm(
        self,
        prompt: str,
        api_key: str,
        strict_json: bool = False,
        max_tokens: int = 480,
    ) -> Dict[str, Any]:
        payload = {
            "model": self.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": int(max_tokens),
        }
        if strict_json:
            payload["response_format"] = {"type": "json_object"}
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
            lowered = body.lower()
            if strict_json and (
                "response_format" in lowered
                or "json_object" in lowered
                or "json schema" in lowered
                or "unsupported" in lowered
            ):
                return self._query_llm(
                    prompt,
                    api_key,
                    strict_json=False,
                    max_tokens=max_tokens,
                )
            raise RuntimeError(f"LLM HTTP {exc.code}: {body}") from exc

    @staticmethod
    def _extract_content(response_json: Dict[str, Any]) -> str:
        choices = response_json.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return str(message.get("content", "")).strip()

    @staticmethod
    def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
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
    def _sanitize_action(text: str) -> str:
        if not text:
            return ""
        for line in str(text).splitlines():
            clean = line.strip().strip("`").strip().strip('"').strip("'")
            if clean:
                return clean[:120]
        return ""

    def _align_to_admissible(self, action: str, admissible: List[str]) -> str:
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
        parts = [f"obs={obs[:200]}"]
        if updates:
            parts.append(f"updates={updates}")
        backlog = self.backlog.export()
        blocked = sorted(self._blocked_actions(obs))
        parts.append(f"current_task={backlog.get('current_task')}")
        parts.append(
            "backlog_counts="
            f"waiting:{len(backlog.get('waiting', []))},"
            f"failed:{len(backlog.get('failed', []))},"
            f"unprogressed:{len(backlog.get('unprogressed', []))}"
        )
        risky = backlog.get("risky_actions", [])
        if risky:
            parts.append(f"risky_actions={risky[:6]}")
        failure_events = backlog.get("failure_events", [])
        if failure_events:
            parts.append(f"last_failure={failure_events[-1]}")
        if self.replay_prefix:
            parts.append(
                f"replay_state=cursor:{self.replay_cursor}/{len(self.replay_prefix)},episode_scored:{self.episode_scored}"
            )
        if blocked:
            parts.append(f"blocked_actions={blocked}")
        if self.stagnation_task:
            parts.append(
                "stagnation_state="
                f"task:{self.stagnation_task},"
                f"no_progress_steps:{self.stagnation_no_progress_steps},"
                f"obs_window:{len(self.stagnation_obs_window)},"
                f"action_window:{len(self.stagnation_action_window)}"
            )
        if self.last_llm_thought:
            parts.append(f"thought={self.last_llm_thought}")
        if self.score_events:
            parts.append(f"last_score_event={self.score_events[-1]}")
        if self.llm_score_points:
            parts.append(f"score_points={self.llm_score_points[-4:]}")
        if self.reflection_archive:
            parts.append(f"reflection_archive={self.reflection_archive[-2:]}")
        return "\n".join(parts)


def build_argparser(parser=None):
    parser = parser or argparse.ArgumentParser()
    group = parser.add_argument_group("PlanningAgent settings")

    group.add_argument("--seed", type=int, default=20241001)
    group.add_argument("--memory-variant", choices=["structured", "rag", "option"], default="structured")
    group.add_argument("--compress-every", type=int, default=8)
    group.add_argument("--use-retrieval", action=argparse.BooleanOptionalAction, default=False)
    group.add_argument("--use-option-module", action=argparse.BooleanOptionalAction, default=False)
    group.add_argument("--max-context-items", type=int, default=12)
    group.add_argument("--max-prompt-chars", type=int, default=6500)
    group.add_argument(
        "--prompt-history-turns",
        type=int,
        default=10,
        help="Number of recent (step, observation, action) turns included in LLM planning prompt.",
    )
    group.add_argument("--max-repeat-action", type=int, default=2)
    group.add_argument(
        "--enable-score-replay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replay the shortest known score-increase prefix after restart.",
    )
    group.add_argument(
        "--max-replay-steps",
        type=int,
        default=8,
        help="Max length of replayed score prefix.",
    )
    group.add_argument("--bootstrap-task-limit", type=int, default=3)
    group.add_argument("--max-waiting-tasks", type=int, default=5, help="Max waiting backlog tasks.")
    group.add_argument("--task-failure-threshold", type=int, default=2, help="Steps before a task is failed.")
    group.add_argument(
        "--task-failed-cooldown-steps",
        type=int,
        default=6,
        help="Cooldown before failed tasks can be retried from the backlog.",
    )
    group.add_argument(
        "--enable-auto-unprogressed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-mark current task unprogressed when evidence shows persistent stagnation.",
    )
    group.add_argument(
        "--task-stagnation-steps",
        type=int,
        default=8,
        help="Consecutive no-progress steps before auto-marking task as unprogressed.",
    )
    group.add_argument(
        "--task-stagnation-max-unique-obs",
        type=int,
        default=3,
        help="Auto-stagnation triggers easier when recent observations stay within this uniqueness bound.",
    )
    group.add_argument(
        "--task-stagnation-max-unique-actions",
        type=int,
        default=3,
        help="Auto-stagnation triggers easier when recent actions stay within this uniqueness bound.",
    )
    group.add_argument("--use-llm-policy", action=argparse.BooleanOptionalAction, default=False)
    group.add_argument(
        "--use-llm-reflection",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use an extra LLM reflection pass to derive lessons, tasks, and risky actions.",
    )
    group.add_argument(
        "--reflect-every",
        type=int,
        default=2,
        help="Run reflection every N steps (and always when failures are detected).",
    )
    group.add_argument(
        "--max-reflection-notes",
        type=int,
        default=12,
        help="Maximum number of deduplicated reflection notes to keep.",
    )
    group.add_argument(
        "--max-reflection-records",
        type=int,
        default=32,
        help="Maximum number of structured reflection archive entries to keep.",
    )
    group.add_argument("--llm-api-url", default="https://tritonai-api.ucsd.edu/v1/chat/completions")
    group.add_argument("--llm-model", default="api-llama-4-scout")
    group.add_argument("--llm-api-key", default=None)
    group.add_argument("--llm-api-key-env", default="TRITON_API_KEY")
    group.add_argument("--llm-timeout", type=int, default=30)
    group.add_argument(
        "--llm-force-json-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ask the LLM API for strict JSON object responses when supported.",
    )
    group.add_argument(
        "--llm-json-retries",
        type=int,
        default=1,
        help="Number of JSON-repair retries when LLM output is not parseable.",
    )
    group.add_argument("--use-llm-parser", action=argparse.BooleanOptionalAction, default=False)
    group.add_argument("--use-llm-compressor", action=argparse.BooleanOptionalAction, default=False)

    return parser


register(
    name="planning-agent",
    desc="Agent powered by a LLM-driven task backlog state machine and scratchpad memory.",
    klass=PlanningAgent,
    add_arguments=build_argparser,
)
