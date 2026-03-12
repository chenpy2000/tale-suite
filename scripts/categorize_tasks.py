#!/usr/bin/env python3
# Categorize TALES tasks by skill (spatial, deductive, inductive, grounded). Outputs task_categories.json and evaluation_subset.json.

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tales

SKILLS = ("spatial", "deductive", "inductive", "grounded")


def _difficulty(env_name: str) -> str:
    if env_name.startswith("TWCooking") and "Level" in env_name:
        try:
            level = int(env_name.replace("TWCookingLevel", "").split("Unseen")[0].split("Seen")[0])
        except (ValueError, IndexError):
            level = 5
        return "easy" if level <= 3 else "medium" if level <= 6 else "hard"
    if "SimonSays" in env_name:
        return "hard" if "100" in env_name else "medium" if "50" in env_name else "easy"
    return "medium"


def _twcooking(env_name: str) -> tuple[str, dict]:
    level = 5
    if env_name.startswith("TWCooking") and "Level" in env_name:
        try:
            level = int(env_name.replace("TWCookingLevel", "").split("Unseen")[0].split("Seen")[0])
        except (ValueError, IndexError):
            pass
    level = min(level, 10)
    w = [0.15 + 0.02 * level, 0.25 + 0.02 * level, 0.35 + 0.01 * level, 0.15 + 0.01 * level]
    s = sum(w)
    weights = {k: round(v / s, 2) for k, v in zip(SKILLS, w)}
    return max(weights, key=weights.get), weights


def _alfworld(env_name: str) -> tuple[str, dict]:
    weights = {"spatial": 0.2, "deductive": 0.3, "inductive": 0.2, "grounded": 0.3}
    primary = "grounded" if "Place" in env_name or "Pick" in env_name else "deductive"
    return primary, weights


def _twx(env_name: str) -> tuple[str, dict]:
    u = env_name.upper()
    if "COOKING" in u:
        return _twcooking("TWCookingLevel3")
    if "MAPREADER" in u:
        return "spatial", {"spatial": 0.5, "deductive": 0.2, "inductive": 0.2, "grounded": 0.1}
    if "SIMONSAYS" in u:
        m = 0.7 if "MEMORY" in u else 0.6
        return "inductive", {"spatial": 0.1, "deductive": 0.1, "inductive": m, "grounded": 0.3 if m == 0.6 else 0.2}
    if "COIN" in u or "COLLECTOR" in u:
        return "grounded", {"spatial": 0.3, "deductive": 0.2, "inductive": 0.2, "grounded": 0.3}
    if "COMMONSENSE" in u or "TWC" in u:
        return "grounded", {"spatial": 0.2, "deductive": 0.2, "inductive": 0.2, "grounded": 0.4}
    if "ARITHMETIC" in u or "SORTING" in u or "PECKING" in u:
        return "deductive", {"spatial": 0.1, "deductive": 0.5, "inductive": 0.3, "grounded": 0.1}
    return "inductive", {s: 0.25 for s in SKILLS}


def _jericho(env_name: str) -> tuple[str, dict]:
    return "spatial", {"spatial": 0.4, "deductive": 0.2, "inductive": 0.2, "grounded": 0.2}


def _scienceworld(env_name: str) -> tuple[str, dict]:
    return "deductive", {"spatial": 0.15, "deductive": 0.4, "inductive": 0.25, "grounded": 0.2}


def categorize(env_name: str, task_module: str) -> dict:
    t = task_module.lower()
    if "textworld" in t and "express" not in t:
        primary, weights = _twcooking(env_name)
    elif "alfworld" in t:
        primary, weights = _alfworld(env_name)
    elif "textworld_express" in t or env_name.startswith("TWX"):
        primary, weights = _twx(env_name)
    elif "jericho" in t:
        primary, weights = _jericho(env_name)
    elif "scienceworld" in t:
        primary, weights = _scienceworld(env_name)
    else:
        primary, weights = "inductive", {s: 0.25 for s in SKILLS}
    return {"primary_skill": primary, "skill_weights": weights, "difficulty": _difficulty(env_name)}


def _subset(categories: dict, per_skill: int = 5) -> dict:
    by_skill = {s: {"easy": [], "medium": [], "hard": []} for s in SKILLS}
    for env, data in categories.items():
        if data["primary_skill"] in by_skill and data["difficulty"] in by_skill[data["primary_skill"]]:
            by_skill[data["primary_skill"]][data["difficulty"]].append(env)
    chosen = {}
    for skill in SKILLS:
        pools = {d: list(lst) for d, lst in by_skill[skill].items()}
        order, out, i = ["easy", "medium", "hard"], [], 0
        while len(out) < per_skill:
            if not any(pools[d] for d in order):
                break
            d = order[i % 3]
            if pools[d]:
                out.append(pools[d].pop(0))
            i += 1
        chosen[skill] = out[:per_skill]
    return {"envs": [e for v in chosen.values() for e in v], "by_skill": chosen}


def _apply_overrides(categories: dict, args) -> dict:
    overrides = {s: getattr(args, f"{s.replace('-', '_')}_weight", None) for s in SKILLS}
    overrides = {k: v for k, v in overrides.items() if v is not None}
    if not overrides:
        return categories
    out = {}
    for env, data in categories.items():
        w = dict(data["skill_weights"])
        for s, v in overrides.items():
            w[s] = v
        s = sum(w.values())
        w = {k: round(v / s, 2) for k, v in w.items()}
        out[env] = {**data, "skill_weights": w, "primary_skill": max(w, key=w.get)}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-o", "--output", type=Path, default=Path("data/task_categories.json"))
    p.add_argument("--manual-review", action="store_true")
    for s in SKILLS:
        p.add_argument(f"--{s}-weight", type=float, default=None)
    p.add_argument("--subset-output", type=Path, default=Path("data/evaluation_subset.json"))
    p.add_argument("--per-skill", type=int, default=5)
    args = p.parse_args()

    envs, env2task = tales.envs, tales.env2task
    if not envs:
        print("No environments loaded.", file=sys.stderr)
        sys.exit(1)

    if args.manual_review:
        categories = {}
        for env in sorted(envs):
            d = categorize(env, env2task.get(env, ""))
            print(f"\n{env}: primary={d['primary_skill']} diff={d['difficulty']}")
            pi = input(f"  skill [{'/'.join(SKILLS)}]: ").strip().lower()
            di = input("  difficulty [easy/medium/hard]: ").strip().lower()
            categories[env] = {
                "primary_skill": pi if pi in SKILLS else d["primary_skill"],
                "skill_weights": d["skill_weights"],
                "difficulty": di if di in ("easy", "medium", "hard") else d["difficulty"],
            }
    else:
        categories = {env: categorize(env, env2task.get(env, "")) for env in envs}
        categories = _apply_overrides(categories, args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(categories, f, indent=2)
    print(f"Wrote {len(categories)} to {args.output}")

    subset = _subset(categories, args.per_skill)
    args.subset_output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.subset_output, "w") as f:
        json.dump(subset, f, indent=2)
    print(f"Wrote {len(subset['envs'])} to {args.subset_output}")


if __name__ == "__main__":
    main()
