import argparse
import json
import os
import re
from typing import Dict, List, Any, Optional
from pathlib import Path

import numpy as np

import tales
from tales.agent import register
from tales.token import get_token_counter


class CompressionAgent(tales.Agent):
    """Agent that compresses game state into structured JSON format for memory efficiency."""
    
    def __init__(self, **kwargs):
        self.seed = kwargs.get("seed", 1234)
        self.rng = np.random.RandomState(self.seed)
        self.token_counter = get_token_counter()
        
        # Compression settings
        self.compression_interval = kwargs.get("compression_interval", 10)
        self.compression_dir = kwargs.get("compression_dir", "./compressions")
        self.use_dynamic_compression = kwargs.get("use_dynamic_compression", False)
        
        # Initialize memory storage
        self.compressions: List[Dict[str, Any]] = []
        self.current_turn = 0
        self.last_compression_turn = 0
        
        # Track game state
        self.history = []  # Store (turn, obs, action, reward, info)
        self.current_location = None
        self.known_rooms = set()
        self.inventory = []
        self.objects_seen = {}
        self.attempts = []
        self.rules_learned = []
        self.current_score = 0
        self.current_goal = None
        
        # Create compression directory
        Path(self.compression_dir).mkdir(parents=True, exist_ok=True)
        
        # fmt:off
        self.actions = [
            "north", "south", "east", "west", "up", "down",
            "look", "inventory", "examine",
            "drop", "take", "take all",
            "open", "close", "use",
            "eat", "attack",
            "wait", "YES",
        ]
        # fmt:on

    @property
    def uid(self):
        return f"CompressionAgent_s{self.seed}_ci{self.compression_interval}"

    @property
    def params(self):
        return {
            "agent_type": "compression",
            "seed": self.seed,
            "compression_interval": self.compression_interval,
            "use_dynamic_compression": self.use_dynamic_compression,
        }

    def reset(self, obs, info, env):
        """Reset agent state for a new episode."""
        self.current_turn = 0
        self.last_compression_turn = 0
        self.history = []
        self.known_rooms = set()
        self.inventory = []
        self.objects_seen = {}
        self.attempts = []
        self.rules_learned = []
        self.current_score = 0
        self.current_location = None
        self.current_goal = self._extract_goal(obs, info)
        
        # Parse initial observation
        self._update_state(obs, info)

    def act(self, obs, reward, done, info):
        """Select an action and potentially compress state."""
        self.current_turn += 1
        
        # Update state from observation
        self._update_state(obs, info)
        
        # Check if we should compress
        if self._should_compress():
            compression = self._create_compression()
            self._save_compression(compression)
            self.last_compression_turn = self.current_turn
        
        # Select action (using random policy for now, but can be enhanced)
        if "admissible_commands" in info:
            action = self.rng.choice(info["admissible_commands"])
        else:
            action = self._select_action(obs, info)
        
        # Record attempt
        stats = {
            "prompt": None,
            "response": None,
            "nb_tokens": self.token_counter(text=obs),
        }
        
        # Store in history
        self.history.append({
            "turn": self.current_turn,
            "obs": obs,
            "action": action,
            "reward": reward,
            "info": info
        })
        
        return str(action), stats

    def _should_compress(self) -> bool:
        """Decide whether to create a compression at this turn."""
        if self.use_dynamic_compression:
            # Dynamic compression based on game signals
            return self._detect_subtask_completion()
        else:
            # Fixed interval compression
            turns_since_last = self.current_turn - self.last_compression_turn
            return turns_since_last >= self.compression_interval

    def _detect_subtask_completion(self) -> bool:
        """Detect if a subtask has been completed (heuristic-based)."""
        if len(self.history) < 2:
            return False
        
        recent_history = self.history[-5:]
        
        # Check for score increase
        if self.current_score > 0 and any(h.get("reward", 0) > 0 for h in recent_history):
            return True
        
        # Check for successful actions after failures
        recent_actions = [h["action"] for h in recent_history]
        if len(set(recent_actions)) == 1 and len(recent_actions) >= 3:
            # Same action repeated multiple times might indicate task completion
            return True
        
        return False

    def _create_compression(self) -> Dict[str, Any]:
        """Create a structured compression of current game state."""
        # TODO: use llm to compress
        turn_start = self.last_compression_turn + 1
        turn_end = self.current_turn
        
        # Extract location state
        location_state = {
            "current_room": self.current_location,
            "known_rooms": list(self.known_rooms),
            "exits": self._extract_exits()
        }
        
        # Extract goals
        goals = {
            "main_goal": self.current_goal,
            "current_subgoal": self._infer_subgoal()
        }
        
        # Get recent attempts
        recent_attempts = self._get_recent_attempts(turn_start, turn_end)
        
        # Calculate progress
        progress_signals = self._calculate_progress(turn_start, turn_end)
        
        # Generate summary
        summary_text = self._generate_summary()
        
        compression = {
            "turn_range": [turn_start, turn_end],
            "location_state": location_state,
            "inventory": self.inventory.copy(),
            "objects_seen": self.objects_seen.copy(),
            "goals": goals,
            "attempts": recent_attempts,
            "rules_learned": self.rules_learned.copy(),
            "progress_signals": progress_signals,
            "summary_text": summary_text
        }
        
        return compression

    def _save_compression(self, compression: Dict[str, Any]):
        """Save compression to memory and disk."""
        self.compressions.append(compression)
        
        # Save to disk
        filename = f"compression_turn_{compression['turn_range'][0]}-{compression['turn_range'][1]}.json"
        filepath = os.path.join(self.compression_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(compression, f, indent=2)

    def retrieve_similar_compressions(self, current_state: Dict[str, Any], top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve similar past compressions based on current state."""
        if not self.compressions:
            return []
        
        # Calculate similarity scores
        similarities = []
        for compression in self.compressions:
            score = self._calculate_similarity(current_state, compression)
            similarities.append((score, compression))
        
        # Sort by similarity and return top-k
        similarities.sort(reverse=True, key=lambda x: x[0])
        return [comp for _, comp in similarities[:top_k]]

    def _calculate_similarity(self, state1: Dict[str, Any], state2: Dict[str, Any]) -> float:
        """Calculate similarity between two states."""
        score = 0.0
        
        # Location similarity
        if state1.get("location_state", {}).get("current_room") == state2.get("location_state", {}).get("current_room"):
            score += 0.3
        
        # Inventory similarity
        inv1 = set(state1.get("inventory", []))
        inv2 = set(state2.get("inventory", []))
        if inv1 or inv2:
            score += 0.2 * len(inv1 & inv2) / max(len(inv1 | inv2), 1)
        
        # Goal similarity
        goal1 = state1.get("goals", {}).get("main_goal", "")
        goal2 = state2.get("goals", {}).get("main_goal", "")
        if goal1 and goal2 and goal1.lower() in goal2.lower():
            score += 0.3
        
        # Objects similarity
        obj1 = set(str(state1.get("objects_seen", {}).values()))
        obj2 = set(str(state2.get("objects_seen", {}).values()))
        if obj1 or obj2:
            score += 0.2 * len(obj1 & obj2) / max(len(obj1 | obj2), 1)
        
        return score

    def _update_state(self, obs: str, info: Dict[str, Any]):
        """Update internal state representation from observation."""
        # Extract current location
        location_match = re.search(r"(?:You are in|Location:)\s+(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", obs)
        if location_match:
            self.current_location = location_match.group(1)
            self.known_rooms.add(self.current_location)
        
        # Extract inventory
        if "inventory" in obs.lower() or "carrying" in obs.lower():
            inv_match = re.findall(r"(?:holding|carrying)(?:\s+a)?\s+([a-z]+)", obs.lower())
            if inv_match:
                self.inventory = inv_match
        
        # Extract objects in current room
        objects = re.findall(r"(?:there is|you see)\s+(?:a\s+)?([a-z]+(?:\s+[a-z]+)?)", obs.lower())
        if objects and self.current_location:
            if self.current_location not in self.objects_seen:
                self.objects_seen[self.current_location] = []
            self.objects_seen[self.current_location].extend(objects)
            # Remove duplicates
            self.objects_seen[self.current_location] = list(set(self.objects_seen[self.current_location]))
        
        # Update score
        if "score" in info:
            old_score = self.current_score
            self.current_score = info["score"]
            
            # Learn rules from score changes
            if self.current_score > old_score and len(self.history) > 0:
                last_action = self.history[-1].get("action", "")
                self.rules_learned.append(f"Action '{last_action}' increased score")

    def _extract_goal(self, obs: str, info: Dict[str, Any]) -> Optional[str]:
        """Extract the main goal from observation or info."""
        # Try to extract from objective/goal in info
        if "objective" in info:
            return info["objective"]
        
        # Try to extract from observation
        goal_match = re.search(r"(?:goal|objective|task):\s*(.+?)(?:\.|$)", obs.lower())
        if goal_match:
            return goal_match.group(1).strip()
        
        return "Unknown goal"

    def _extract_exits(self) -> Dict[str, List[str]]:
        """Extract known exits from visited rooms."""
        exits = {}
        if self.current_location:
            exits[self.current_location] = ["north", "south", "east", "west"]  # Simplified
        return exits

    def _infer_subgoal(self) -> str:
        """Infer current subgoal from recent actions."""
        if len(self.history) < 3:
            return "Exploring"
        
        recent_actions = [h["action"] for h in self.history[-5:]]
        
        # Pattern matching for common subgoals
        if any("take" in a for a in recent_actions):
            return "Collecting items"
        elif any("open" in a for a in recent_actions):
            return "Opening containers"
        elif any(d in a for d in ["north", "south", "east", "west"] for a in recent_actions):
            return "Navigating/Exploring"
        
        return "Unknown subgoal"

    def _get_recent_attempts(self, turn_start: int, turn_end: int) -> List[Dict[str, str]]:
        """Get attempts made in the turn range."""
        attempts = []
        for entry in self.history:
            if turn_start <= entry["turn"] <= turn_end:
                attempt = {
                    "action": entry["action"],
                    "result": "success" if entry.get("reward", 0) > 0 else "failed",
                    "reason": self._infer_failure_reason(entry["obs"])
                }
                attempts.append(attempt)
        return attempts[-10:]  # Keep last 10 attempts

    def _infer_failure_reason(self, obs: str) -> str:
        """Infer reason for action failure from observation."""
        if "can't" in obs.lower() or "cannot" in obs.lower():
            return "action not possible"
        elif "don't see" in obs.lower() or "not here" in obs.lower():
            return "object not found"
        elif "closed" in obs.lower():
            return "container closed"
        return "unknown"

    def _calculate_progress(self, turn_start: int, turn_end: int) -> Dict[str, Any]:
        """Calculate progress signals for the turn range."""
        score_delta = 0
        new_discoveries = []
        
        for entry in self.history:
            if turn_start <= entry["turn"] <= turn_end:
                if entry.get("reward", 0) > 0:
                    score_delta += entry["reward"]
                
                # Check for new room discoveries
                if self.current_location and self.current_location not in new_discoveries:
                    new_discoveries.append(f"visited {self.current_location}")
        
        return {
            "score_delta": score_delta,
            "new_discoveries": new_discoveries[-5:]  # Last 5 discoveries
        }

    def _generate_summary(self) -> str:
        """Generate a human-readable summary of recent activity."""
        parts = []
        
        if self.known_rooms:
            parts.append(f"Visited {', '.join(list(self.known_rooms)[-3:])}")
        
        if self.inventory:
            parts.append(f"Holding {', '.join(self.inventory)}")
        
        if self.current_location and self.current_location in self.objects_seen:
            objects = self.objects_seen[self.current_location][:3]
            if objects:
                parts.append(f"Found {', '.join(objects)}")
        
        if self.current_goal:
            parts.append(f"Goal: {self.current_goal}")
        
        return ". ".join(parts) + "."

    def _select_action(self, obs: str, info: Dict[str, Any]) -> str:
        """Select an action based on current state (random baseline)."""
        action = self.rng.choice(self.actions)
        
        if action in ["take", "drop", "eat", "attack", "open", "close", "examine"]:
            words = re.findall(r"\b[a-zA-Z]{4,}\b", obs)
            if len(words) > 0:
                action += " " + self.rng.choice(words)
        
        return action


def build_argparser(parser=None):
    parser = parser or argparse.ArgumentParser()
    group = parser.add_argument_group("CompressionAgent settings")
    group.add_argument(
        "--seed",
        type=int,
        default=20241001,
        help="Random generator seed. Default: %(default)s",
    )
    group.add_argument(
        "--compression-interval",
        type=int,
        default=10,
        help="Number of turns between compressions. Default: %(default)s",
    )
    group.add_argument(
        "--compression-dir",
        type=str,
        default="./compressions",
        help="Directory to save compressions. Default: %(default)s",
    )
    group.add_argument(
        "--use-dynamic-compression",
        action="store_true",
        help="Use dynamic compression based on subtask completion instead of fixed intervals.",
    )
    return parser


register(
    name="compression",
    desc=(
        "This agent compresses game state into structured JSON format at regular intervals "
        "or when subtasks are completed. Compressions are saved and can be retrieved for similar states."
    ),
    klass=CompressionAgent,
    add_arguments=build_argparser,
)
