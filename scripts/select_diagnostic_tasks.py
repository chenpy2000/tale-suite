#!/usr/bin/env python3
# Select diagnostic tasks for skill evaluation. Default: evaluation_subset.json. Override: --config or --from-config.

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tales

SKILLS = ("spatial", "deductive", "inductive", "grounded")
DIFF_BY_POS = ("easy", "medium", "medium-hard", "hard", "very-hard")


def _parse_entries(data: dict) -> dict:
    out = {}
    for skill in SKILLS:
        entries = []
        for item in data.get(skill, []):
            if isinstance(item, str):
                entries.append({"task": item, "difficulty": "medium", "episodes": 5 if skill == "inductive" else 1})
            elif isinstance(item, dict) and item.get("task"):
                entries.append({
                    "task": item["task"],
                    "difficulty": item.get("difficulty", "medium"),
                    "episodes": item.get("episodes", 5 if skill == "inductive" else 1),
                })
        out[skill] = entries
    return out


def _load_file(path: Path) -> dict:
    with open(path) as f:
        if path.suffix.lower() in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError:
                raise ImportError("pip install pyyaml")
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    return data


def load_default(project_root: Path) -> dict:
    subset_path = project_root / "data" / "evaluation_subset.json"
    cat_path = project_root / "data" / "task_categories.json"
    if not subset_path.exists():
        raise FileNotFoundError(f"Run categorize_tasks.py first. Missing: {subset_path}")
    with open(subset_path) as f:
        subset = json.load(f)
    categories = json.load(open(cat_path)) if cat_path.exists() else {}
    out = {}
    for skill in SKILLS:
        tasks = subset.get("by_skill", {}).get(skill, [])
        entries = []
        for i, task in enumerate(tasks):
            diff = categories.get(task, {}).get("difficulty", "medium")
            if len(tasks) == 5 and all(categories.get(t, {}).get("difficulty") == diff for t in tasks):
                diff = DIFF_BY_POS[min(i, 4)]
            entries.append({"task": task, "difficulty": diff, "episodes": 5 if skill == "inductive" else 1})
        out[skill] = entries
    return out


def main():
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", type=Path, help="Flat config with skill keys at top level")
    p.add_argument("--from-config", type=Path, help="Evaluation config with nested diagnostic_tasks")
    p.add_argument("-o", "--output", type=Path, default=Path("data/diagnostic_tasks.json"))
    p.add_argument("--subset", type=Path, default=root / "data" / "evaluation_subset.json")
    args = p.parse_args()

    if args.from_config:
        path = args.from_config if args.from_config.is_absolute() else root / args.from_config
        if not path.exists():
            print(f"Not found: {path}", file=sys.stderr)
            sys.exit(1)
        data = _load_file(path)
        tasks = _parse_entries(data.get("diagnostic_tasks", data))
    elif args.config:
        path = args.config if args.config.is_absolute() else root / args.config
        if not path.exists():
            print(f"Not found: {path}", file=sys.stderr)
            sys.exit(1)
        tasks = _parse_entries(_load_file(path))
    else:
        tasks = load_default(root)

    valid = set(tales.envs)
    invalid = [e["task"] for s in SKILLS for e in tasks.get(s, []) if e.get("task") and e["task"] not in valid]
    if invalid:
        print(f"Invalid tasks: {invalid}", file=sys.stderr)
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(tasks, f, indent=2)
    print(f"Wrote {sum(len(tasks[s]) for s in SKILLS)} tasks to {args.output}")


if __name__ == "__main__":
    main()
