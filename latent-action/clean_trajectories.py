#!/usr/bin/env python3
# Filter collected trajectories before VQ-VAE training.
# Drops episodes that are too short, too low-reward, or too repetitive (same action over and over).

import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    """Walk data/trajectories, keep episodes that pass min-length, min-reward, max-repetition-rate. Write to trajectories_cleaned."""
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--min-length", type=int, default=15)
    p.add_argument("--min-reward", type=float, default=-5)
    p.add_argument("--max-repetition-rate", type=float, default=0.4)
    args = p.parse_args()

    root = Path(__file__).resolve().parent
    inp = args.input_dir or root / "data" / "trajectories"
    out = args.output_dir or root / "data" / "trajectories_cleaned"
    if not inp.exists():
        p.error(f"not found: {inp}")

    out.mkdir(parents=True, exist_ok=True)
    kept = 0
    for path in sorted(inp.rglob("episode_*.json")):
        try:
            traj = json.load(path.open(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        steps = traj.get("steps", [])
        if not isinstance(steps, list):
            continue
        n = len(steps)
        r = traj.get("total_reward")
        if r is None:
            r = sum(float(s.get("reward", 0)) for s in steps)
        if n < args.min_length or r < args.min_reward:
            continue
        actions = [str(s.get("action", "")) for s in steps]
        if not actions:
            continue
        # Repetition rate = max count of any single action / total actions
        rep = max(actions.count(a) for a in set(actions)) / len(actions)
        if rep > args.max_repetition_rate:
            continue
        dst = out / path.relative_to(inp)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open("w", encoding="utf-8") as f:
            json.dump(traj, f, indent=2, ensure_ascii=True)
        kept += 1

    print(f"kept {kept} -> {out}")


if __name__ == "__main__":
    main()
