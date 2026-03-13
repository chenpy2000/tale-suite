#!/usr/bin/env python3
"""
Filter collected trajectories before VQ-VAE training.

Drops episodes that are too short (--min-length), too low-reward (--min-reward),
or too repetitive (--max-repetition-rate). With --prune-passive: removes redundant
examine/look steps within kept episodes so the VQ-VAE trains on higher-quality
action sequences.

Input: data/trajectories_raw/ or --input-dir (episode_*.json)
Output: data/trajectories_cleaned/ or --output-dir
"""

import argparse
import json
from collections import Counter
from pathlib import Path

_PASSIVE_PREFIXES = ("examine ", "look", "inventory")


def _is_passive(action):
    a = action.strip().lower()
    return any(a.startswith(p) or a == p for p in _PASSIVE_PREFIXES)


def _prune_passive_steps(steps, max_passive_rate=0.25):
    """Remove duplicate passive actions (examine X repeated, look repeated) to
    keep passive actions at most max_passive_rate of the episode. Keeps the
    first occurrence of each unique passive action."""
    seen_passive = set()
    pruned = []
    passive_count = 0
    for s in steps:
        a = str(s.get("action", "")).strip()
        if _is_passive(a):
            key = a.lower()
            if key in seen_passive:
                continue  # drop repeated passive action
            seen_passive.add(key)
            passive_count += 1
        pruned.append(s)

    # If still too many passive actions, keep only first N passive ones
    limit = max(1, int(len(pruned) * max_passive_rate))
    if passive_count > limit:
        result = []
        pcount = 0
        for s in pruned:
            a = str(s.get("action", "")).strip()
            if _is_passive(a):
                pcount += 1
                if pcount > limit:
                    continue
            result.append(s)
        return result
    return pruned


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--min-length", type=int, default=3,
                   help="Minimum steps per episode (default 3)")
    p.add_argument("--min-reward", type=float, default=-20,
                   help="Minimum total reward (default -20)")
    p.add_argument("--max-repetition-rate", type=float, default=0.5,
                   help="Max fraction of steps with same action (default 0.5)")
    p.add_argument("--prune-passive", action="store_true",
                   help="Remove redundant examine/look steps to improve training data quality")
    p.add_argument("--max-passive-rate", type=float, default=0.25,
                   help="Max fraction of passive actions after pruning (default 0.25)")
    args = p.parse_args()

    root = Path(__file__).resolve().parent
    inp = args.input_dir or root / "data" / "trajectories"
    out = args.output_dir or root / "data" / "trajectories_cleaned"
    if not inp.exists():
        p.error(f"not found: {inp}")

    out.mkdir(parents=True, exist_ok=True)
    kept = 0
    dropped = 0
    steps_before = 0
    steps_after = 0
    for path in sorted(inp.rglob("episode_*.json")):
        try:
            traj = json.load(path.open(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            dropped += 1
            continue
        steps = traj.get("steps", [])
        if not isinstance(steps, list):
            dropped += 1
            continue
        n = len(steps)
        r = traj.get("total_reward")
        if r is None:
            r = sum(float(s.get("reward", 0)) for s in steps)
        if n < args.min_length or r < args.min_reward:
            dropped += 1
            continue
        actions = [str(s.get("action", "")) for s in steps]
        if not actions:
            dropped += 1
            continue
        rep = max(actions.count(a) for a in set(actions)) / len(actions)
        if rep > args.max_repetition_rate:
            dropped += 1
            continue

        steps_before += len(steps)

        if args.prune_passive:
            steps = _prune_passive_steps(steps, args.max_passive_rate)
            traj = dict(traj, steps=steps)
            if len(steps) < args.min_length:
                dropped += 1
                continue

        steps_after += len(steps)

        dst = out / path.relative_to(inp)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open("w", encoding="utf-8") as f:
            json.dump(traj, f, indent=2, ensure_ascii=True)
        kept += 1

    total = kept + dropped
    print(f"kept {kept} / {total} episodes -> {out}")
    if dropped:
        print(f"  dropped {dropped} (min_length={args.min_length}, min_reward={args.min_reward}, max_rep={args.max_repetition_rate})")
    if args.prune_passive:
        print(f"  passive pruning: {steps_before} steps -> {steps_after} steps ({steps_before - steps_after} removed)")


if __name__ == "__main__":
    main()
