"""LLM + VQ-VAE agent: VQ-VAE provides strategic action rankings, LLM decides with full context."""
import argparse
import json
import logging
import os
import re
import sys
import urllib.request
import urllib.error
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import tales
from tales.agent import register

_ROOT = Path(__file__).resolve().parent.parent
_agent_logger = None


def _get_agent_logger(debug_agent=False):
    global _agent_logger
    if not debug_agent:
        return None
    if _agent_logger is not None:
        return _agent_logger
    log_dir = _ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "agent_debug.log"
    logger = logging.getLogger("llm_vqvae_agent")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    _agent_logger = logger
    return _agent_logger

sys.path.insert(0, str(_ROOT / "latent-action"))
from vqvae import VQVAE, ActionVocab, PAD, UNK, normalize_text

SYSTEM_PROMPT = (
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
   "CRITICAL RULES:\n"
   "- Use the WORLD STATE to know where things are. Do NOT take something "
   "you are already holding. Do NOT put something where it already is.\n"
   "- NEVER undo a previous action. If you just put X somewhere, do NOT pick it up "
   "again unless you need it for a SPECIFIC next step (cooking, eating, etc).\n"
   "- Each action should move you forward in the game flow above.\n"
   "- If an action is marked [UNDO!] it would reverse a previous action -- avoid it.\n\n"
   "Output: the exact command, nothing else."
)


def _parse_action_effects(action):
    """Extract (verb, item, location) from an action string."""
    a = action.lower().strip()
    m = re.match(r"take (.+?) from (.+)", a)
    if m:
        return "take", m.group(1), m.group(2)
    m = re.match(r"put (.+?) on (.+)", a)
    if m:
        return "put", m.group(1), m.group(2)
    m = re.match(r"insert (.+?) into (.+)", a)
    if m:
        return "put", m.group(1), m.group(2)
    if a.startswith("take "):
        return "take", a[5:], "nearby"
    if a.startswith("drop "):
        return "drop", a[5:], "floor"
    if a.startswith("open "):
        return "open", a[5:], None
    if a.startswith("close "):
        return "close", a[6:], None
    m = re.match(r"cook (.+?) with (.+)", a)
    if m:
        return "cook", m.group(1), m.group(2)
    if a.startswith("eat "):
        return "eat", a[4:], None
    if a.startswith("examine "):
        return "examine", a[8:], None
    if a == "prepare meal":
        return "prepare", "meal", None
    if a == "look":
        return "look", None, None
    return None, None, None


class LLMVQVAEAgent(tales.Agent):
    def __init__(
        self,
        api_key,
        api_url,
        model,
        vqvae_checkpoint,
        window_size=None,
        vqvae_top_k=5,
        agent_step_budget=200,
        seed=20241001,
        debug_agent=False,
        **kwargs,
    ):
        self.api_key = api_key or os.environ.get("TRITONAI_API_KEY") or os.environ.get("TRITON_API_KEY")
        if not self.api_key:
            print("[llm-vqvae] WARNING: No API key (--api-key or TRITONAI_API_KEY). LLM calls will fail; using VQ-VAE only.")
        self.debug_agent = bool(debug_agent)
        self.vqvae_top_k = int(vqvae_top_k)
        self.nb_steps = int(agent_step_budget)
        self.api_url = api_url.rstrip("/")
        if not self.api_url.endswith("/v1/chat/completions"):
            self.api_url = self.api_url.rstrip("/") + "/v1/chat/completions"
        self.model_name = model
        self.rng = np.random.RandomState(seed)

        path = Path(vqvae_checkpoint)
        if not path.is_absolute():
            path = _ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"VQ-VAE checkpoint not found: {path}")

        ckpt = torch.load(path, map_location="cpu")
        args = ckpt.get("args", {})
        stoi = ckpt.get("action_vocab", {})
        if not stoi:
            raise ValueError(f"No action_vocab in checkpoint: {path}")

        action_vocab = ActionVocab(stoi)
        num_codes = args.get("num_codes", 128)
        latent_dim = args.get("latent_dim", 256)
        text_model = args.get("text_model", "all-MiniLM-L6-v2")
        traj_root = str(_ROOT / "latent-action" / "data" / "trajectories")

        self.vqvae = VQVAE(
            text_model_name=text_model,
            action_vocab=action_vocab,
            trajectories_root=traj_root,
            latent_dim=latent_dim,
            num_codes=num_codes,
            commitment_beta=args.get("commitment_beta", 1.0),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        sd = ckpt.get("model_state_dict", ckpt)
        if "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
        self.vqvae.load_state_dict(sd, strict=False)
        self.vqvae.eval()
        self.vqvae.to(self.vqvae.device)

        self.w = window_size if window_size is not None else args.get("window_size", 5)
        ckpt_epoch = ckpt.get("epoch", "?")
        print(f"[llm-vqvae] Loaded {path.name}: window_size={self.w}, num_codes={num_codes}, epoch={ckpt_epoch}")
        if self.debug_agent:
            print(f"[llm-vqvae] Debug logging enabled -> {_ROOT / 'logs' / 'agent_debug.log'}")

        self._reset_state()

    def _reset_state(self):
        self.obs_h = []
        self.act_h = []
        self.full_act_h = []
        self.action_counts = Counter()
        self.step_num = 0
        self.world_state = {}
        self.inventory = set()
        self.examined = set()
        self.recipe_known = False
        self.cooked = set()
        self.meal_prepared = False
        self.meal_eaten = False

    @property
    def uid(self):
        return f"LLMVQVAE_{self.model_name}_w{self.w}_topk{self.vqvae_top_k}"

    @property
    def params(self):
        return {
            "agent_type": "llm-vqvae",
            "model": self.model_name,
            "window_size": self.w,
            "vqvae_top_k": self.vqvae_top_k,
        }

    def reset(self, obs, info, env_name=None):
        self._reset_state()

    def _update_world_state(self, action, obs):
        verb, item, loc = _parse_action_effects(action)
        if not verb:
            return
        if verb == "take" and item:
            self.inventory.add(item)
            if loc and loc != "nearby":
                self.world_state.pop(item, None)
        elif verb == "put" and item and loc:
            self.inventory.discard(item)
            self.world_state[item] = f"on {loc}"
        elif verb == "drop" and item:
            self.inventory.discard(item)
            self.world_state[item] = "on floor"
        elif verb == "open" and item:
            self.world_state[item] = "open"
        elif verb == "close" and item:
            self.world_state[item] = "closed"
        elif verb == "cook" and item:
            self.inventory.discard(item)
            self.cooked.add(item)
            self.world_state[item] = f"cooked (with {loc})"
        elif verb == "eat" and item:
            self.inventory.discard(item)
            self.world_state.pop(item, None)
            self.meal_eaten = True
        elif verb == "examine" and item:
            self.examined.add(item)
            if item == "cookbook" or "cookbook" in item:
                self.recipe_known = True
        elif verb == "prepare":
            self.meal_prepared = True

    def _build_world_state_text(self):
        lines = []
        if self.inventory:
            lines.append(f"  Holding: {', '.join(sorted(self.inventory))}")
        else:
            lines.append("  Holding: nothing")
        for obj, state in sorted(self.world_state.items()):
            lines.append(f"  {obj}: {state}")
        return "WORLD STATE:\n" + "\n".join(lines)

    def _build_progress_text(self):
        milestones = []
        if self.recipe_known:
            milestones.append("[done] Read recipe")
        else:
            milestones.append("[TODO] Read recipe (examine cookbook)")
        if self.cooked:
            for item in self.cooked:
                milestones.append(f"[done] Cooked {item}")
        if self.meal_prepared:
            milestones.append("[done] Prepared meal")
        if self.meal_eaten:
            milestones.append("[done] Ate meal")
        return "PROGRESS:\n" + "\n".join(f"  {m}" for m in milestones)

    def _is_undo(self, cmd):
        """Check if cmd would reverse any recent action."""
        if not self.full_act_h:
            return False
        verb, item, loc = _parse_action_effects(cmd)
        if not verb or not item:
            return False
        for past in reversed(self.full_act_h[-6:]):
            pv, pi, pl = _parse_action_effects(past)
            if not pv or not pi:
                continue
            if pi != item:
                continue
            if verb == "take" and pv == "put":
                return True
            if verb == "take" and pv == "drop":
                return True
            if verb == "put" and pv == "take":
                return True
            if verb == "drop" and pv == "take":
                return True
            if verb == "open" and pv == "close":
                return True
            if verb == "close" and pv == "open":
                return True
        return False

    def _query_llm(self, prompt, log=None):
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 1024,
        }
        req = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            raw_body = resp.read().decode("utf-8")
        data = json.loads(raw_body)

        if "error" in data:
            if log:
                log.debug("  LLM API error: %s", str(data["error"])[:200])
            return ""

        choices = data.get("choices", [])
        if not choices:
            if log:
                log.debug("  LLM no choices: %s", raw_body[:300])
            return ""

        msg = choices[0].get("message") or choices[0]
        content = msg.get("content") or msg.get("text") or ""
        content = content.strip()

        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        if log and reasoning:
            log.debug("  LLM reasoning: '%s'", reasoning[:300])

        if "<think>" in content:
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        if "\n" in content:
            for line in content.split("\n"):
                line = line.strip().lstrip(">-•* ").strip()
                if line:
                    content = line
                    break

        if log:
            finish = choices[0].get("finish_reason", "?")
            log.debug("  LLM raw: '%s' (finish_reason=%s)", content[:120], finish)

        return content

    def _align_to_admissible(self, action, admissible):
        if not admissible:
            return ""
        action = (action or "").strip()
        if len(action) < 2:
            return ""
        if action in admissible:
            return action
        low = action.lower()
        for cmd in admissible:
            if cmd.lower() == low:
                return cmd
        matches = []
        for cmd in admissible:
            cl = cmd.lower()
            if cl in low or low in cl:
                matches.append(cmd)
        if matches:
            return max(matches, key=len)
        return ""

    def _build_history_text(self):
        if not self.full_act_h:
            return ""
        n = min(30, len(self.full_act_h))
        running_counts = Counter()
        for act in self.full_act_h:
            running_counts[act] += 1

        lines = []
        for i, act in enumerate(self.full_act_h[-n:]):
            step = len(self.full_act_h) - n + i + 1
            cnt = running_counts[act]
            tag = f" (x{cnt} total!)" if cnt > 2 else f" (x{cnt})" if cnt > 1 else ""
            lines.append(f"  step {step}: {act}{tag}")

        return "Action history:\n" + "\n".join(lines) + "\n"

    def _get_trajectory_prediction(self, log_probs, admissible):
        """Top VQ-VAE vocab actions filtered to be relevant to current game."""
        itos = self.vqvae.action_vocab.itos
        all_probs = log_probs.cpu().numpy()
        top_indices = np.argsort(-all_probs)[:30]

        game_words = set()
        for cmd in admissible:
            game_words.update(cmd.lower().split())
        for obj in list(self.world_state.keys()) + list(self.inventory) + list(self.examined):
            game_words.update(obj.lower().split())

        predictions = []
        for idx in top_indices:
            token = itos.get(idx, "")
            if not token or token in (PAD, UNK):
                continue
            words = set(token.lower().split())
            if words & game_words:
                predictions.append(token)
            if len(predictions) >= 5:
                break
        return predictions

    def _is_stuck(self, proposed_action):
        prop = normalize_text(proposed_action)
        h = [normalize_text(a) for a in self.full_act_h[-12:]] + [prop]
        n = len(h)

        for period in range(1, n // 2 + 1):
            tail = h[-period:]
            prev = h[-(2 * period):-period]
            if len(prev) == period and tail == prev:
                return True

        recent = h[-6:]
        verbs = {"take", "put", "drop", "insert"}
        objects = set()
        all_shuffle = True
        for act in recent:
            parts = act.split()
            if parts and parts[0] in verbs:
                obj = " ".join(p for p in parts[1:] if p not in ("from", "on", "into", "in"))
                objects.add(obj)
            else:
                all_shuffle = False
                break
        if all_shuffle and len(objects) <= 2 and len(recent) >= 4:
            return True

        return False

    def _compute_vq_scores(self, obs_str, admissible):
        """VQ-VAE forward + per-action log-probs. Returns (scores_dict, log_probs_tensor)."""
        if not admissible:
            return {}, None
        last_obs = self.obs_h[-(self.w - 1) :] + [obs_str]
        # Match training: encoder saw real actions at all positions. Use last known action
        # at final position instead of PAD so latent matches what model was trained on.
        last_known = self.act_h[-1] if self.act_h else PAD
        last_act = self.act_h[-(self.w - 1) :] + [last_known]
        obs_seq = ([""] * (self.w - len(last_obs)) + last_obs)[-self.w :]
        act_seq = ([PAD] * (self.w - len(last_act)) + last_act)[-self.w :]
        obs_seq = [normalize_text(o) for o in obs_seq]
        act_seq = [PAD if a == PAD else normalize_text(a) for a in act_seq]
        self.vqvae.to(self.vqvae.device)
        with torch.no_grad():
            logits, _, _, _ = self.vqvae([obs_seq], [act_seq])
            last_logits = logits[0, -1].float().clamp(-50.0, 50.0)
            log_probs = torch.log_softmax(last_logits, dim=-1)
        vq_scores = {}
        unk_idx = self.vqvae.action_vocab.stoi.get(UNK, 0)
        for cmd in admissible:
            try:
                idx = self.vqvae.action_vocab.encode(normalize_text(cmd))
                vq_scores[cmd] = float(log_probs[idx].item())
            except Exception:
                vq_scores[cmd] = float(log_probs[unk_idx].item())
        return vq_scores, log_probs

    def score_actions(self, obs, admissible_commands, info):
        """VQ-VAE log-probs per action. Returns dict[action, score] in [0, 1]."""
        admissible = list(admissible_commands or [])
        if not admissible:
            return {}
        obs_str = normalize_text(obs)
        vq_scores, _ = self._compute_vq_scores(obs_str, admissible)
        if not vq_scores:
            return {}
        mn, mx = min(vq_scores.values()), max(vq_scores.values())
        if mx > mn:
            return {a: (s - mn) / (mx - mn) for a, s in vq_scores.items()}
        return {a: 1.0 for a in vq_scores}

    def _pick_unstuck_action(self, admissible, sorted_by_vq, log):
        by_count = sorted(admissible, key=lambda c: self.action_counts.get(normalize_text(c), 0))
        least_count = self.action_counts.get(normalize_text(by_count[0]), 0)
        least_tried = [c for c in by_count if self.action_counts.get(normalize_text(c), 0) == least_count]
        vq_map = dict(sorted_by_vq)
        pick = max(least_tried, key=lambda c: vq_map.get(c, -999))
        if log:
            log.debug("  UNSTUCK: picking '%s' (done %d times)", pick, least_count)
        return pick

    def act(self, obs, reward, done, infos):
        log = _get_agent_logger(self.debug_agent)

        admissible = list(infos.get("admissible_commands") or [])
        if not admissible:
            return "wait", {"nb_tokens": 0}

        self.step_num += 1
        obs_str = normalize_text(obs)

        vq_scores, log_probs = self._compute_vq_scores(obs_str, admissible)
        traj_pred = self._get_trajectory_prediction(log_probs, admissible) if log_probs is not None else []

        sorted_by_vq = sorted(vq_scores.items(), key=lambda x: -x[1])
        top_k = sorted_by_vq[:self.vqvae_top_k]

        if log:
            log.debug("--- step %d | %d commands ---", self.step_num, len(admissible))
            log.debug("  Inventory: %s", sorted(self.inventory) if self.inventory else "empty")
            log.debug("  Traj prediction: %s", traj_pred[:5])
            for cmd, sc in sorted_by_vq[:8]:
                undo = " [UNDO]" if self._is_undo(cmd) else ""
                log.debug("  %s: %.3f (done %dx)%s", cmd, sc, self.action_counts.get(normalize_text(cmd), 0), undo)

        # Build prompt sections
        world_state_text = self._build_world_state_text()
        progress_text = self._build_progress_text()
        history_text = self._build_history_text()

        traj_lines = [f"  {i+1}. {a}" for i, a in enumerate(traj_pred)] if traj_pred else ["  (no relevant predictions)"]
        traj_hint = "VQ-VAE strategy (what winning games do at this stage):\n" + "\n".join(traj_lines)

        vq_hint_parts = []
        for i, (cmd, _) in enumerate(top_k):
            cnt = self.action_counts.get(normalize_text(cmd), 0)
            tags = []
            if cnt > 1:
                tags.append(f"done {cnt}x!")
            elif cnt > 0:
                tags.append(f"done {cnt}x")
            if self._is_undo(cmd):
                tags.append("UNDO!")
            tag = f" [{', '.join(tags)}]" if tags else ""
            vq_hint_parts.append(f"  {i+1}. {cmd}{tag}")
        vq_hint = "VQ-VAE ranked valid actions:\n" + "\n".join(vq_hint_parts)

        MAX_CMD = 30
        cmd_lines = []
        for c in admissible[:MAX_CMD]:
            undo_tag = " [UNDO!]" if self._is_undo(c) else ""
            cmd_lines.append(f"- {c}{undo_tag}")
        cmd_list = "\n".join(cmd_lines)
        if len(admissible) > MAX_CMD:
            cmd_list += f"\n... and {len(admissible) - MAX_CMD} more"

        prompt = (
            f"Step {self.step_num} of {self.nb_steps}\n\n"
            f"{world_state_text}\n\n"
            f"{progress_text}\n\n"
            f"{history_text}\n"
            f"Current observation:\n{obs_str}\n\n"
            f"{traj_hint}\n\n"
            f"{vq_hint}\n\n"
            f"Valid commands:\n{cmd_list}\n\n"
            f"Pick one command:"
        )

        # Query LLM
        raw_llm = ""
        vq_pick = top_k[0][0] if top_k else admissible[0]
        try:
            raw_llm = self._query_llm(prompt, log=log)
        except Exception as e:
            if log:
                log.debug("  LLM EXCEPTION: %s", e)

        action = ""
        source = "vq"
        if raw_llm:
            action = self._align_to_admissible(raw_llm, admissible)
            if action:
                source = "llm"
            elif log:
                log.debug("  LLM '%s' not in admissible", raw_llm[:80])

        if not action:
            action = vq_pick

        if self._is_stuck(action):
            action = self._pick_unstuck_action(admissible, sorted_by_vq, log)
            source = "unstuck"

        if log:
            log.debug("--- Final [%s]: %s (LLM: '%s', VQ #1: %s) ---",
                      source, action, raw_llm[:60] if raw_llm else "<empty>", vq_pick)
            log.debug("")

        # Update state
        self._update_world_state(action, obs_str)
        norm_action = normalize_text(action)
        self.action_counts[norm_action] += 1
        self.obs_h.append(obs_str)
        self.act_h.append(norm_action)
        self.full_act_h.append(norm_action)
        self.obs_h = self.obs_h[-(self.w + 35):]
        self.act_h = self.act_h[-self.w:]

        norm_scores = {}
        if vq_scores:
            mn, mx = min(vq_scores.values()), max(vq_scores.values())
            norm_scores = {a: (s - mn) / (mx - mn) if mx > mn else 1.0 for a, s in vq_scores.items()}
        stats = {"nb_tokens": 0, "action_scores": norm_scores}
        return action, stats


def build_argparser(parser=None):
    parser = parser or argparse.ArgumentParser()
    g = parser.add_argument_group("LLM+VQ-VAE agent")
    g.add_argument("--api-key", default=None, help="TritonAI API key (or TRITONAI_API_KEY / TRITON_API_KEY env)")
    g.add_argument("--api-url", default="https://tritonai-api.ucsd.edu")
    g.add_argument("--model", default="api-gpt-oss-120b")
    g.add_argument("--vqvae-checkpoint", default="latent-action/checkpoints/vqvae_checkpoint.pt")
    g.add_argument("--window-size", type=int, default=None)
    g.add_argument("--vqvae-top-k", type=int, default=5)
    g.add_argument("--agent-step-budget", type=int, default=200)
    g.add_argument("--seed", type=int, default=20241001, help="Random seed for reproducibility.")
    g.add_argument("--debug-agent", action="store_true")
    return parser


register(
    name="llm-vqvae",
    desc="LLM plays the game; VQ-VAE provides strategic action rankings from learned play patterns.",
    klass=LLMVQVAEAgent,
    add_arguments=build_argparser,
)
