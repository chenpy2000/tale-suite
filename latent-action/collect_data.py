#!/usr/bin/env python3
# Runs benchmark.py per env, converts output to episode JSONs.

import argparse
import json
import subprocess
import sys
from numbers import Number
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import tales


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=["textworld", "textworld_express"],
                   choices=sorted(tales.envs_per_task.keys()))
    p.add_argument("--envs", nargs="+")
    p.add_argument("--episodes-per-game", type=int, default=1)
    p.add_argument("--nb-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=20241001)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--work-dir", type=Path, default=None)
    p.add_argument("--agent", default="agents/collectors.py")
    p.add_argument("--agent-name", default="diverse-collector")
    p.add_argument("--min-episode-length", type=int, default=None)
    p.add_argument("--min-reward", type=float, default=None)
    p.add_argument("--max-repetition-rate", type=float, default=None)
    p.add_argument("--noise-rate", type=float, default=None)
    p.add_argument("--no-admissible-commands", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    root = Path(__file__).resolve().parent
    out_dir = args.output_dir or root / "data" / "trajectories"
    work_dir = args.work_dir or root / "data" / "trajectory_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    envs = args.envs or []
    if not envs:
        for t in args.tasks:
            envs.extend(tales.envs_per_task[t])
    envs = list(dict.fromkeys(envs))

    if args.dry_run:
        for e in envs:
            print(e)
        return

    failures = []
    for env in envs:
        safe = env.replace("/", "-")
        traj_path = work_dir / f"{safe}.json"
        qtable_path = work_dir / f"{safe}_qtable.json"

        cmd = [sys.executable, "benchmark.py", "--agent", args.agent, args.agent_name,
               "--envs", env, "--nb-steps", str(args.nb_steps), "--seed", str(args.seed),
               "--trajectory-path", str(traj_path), "--autosave", "--save-every", "10", "-ff"]
        if args.agent_name == "trajectory-collector":
            cmd += ["--alpha", "0.2", "--gamma", "0.95", "--epsilon", "0.3", "--epsilon-min", "0.05",
                    "--epsilon-decay", "0.999", "--qtable-path", str(qtable_path)]
        if args.agent_name == "diverse-collector":
            if args.min_episode_length is not None: cmd += ["--min-episode-length", str(args.min_episode_length)]
            if args.min_reward is not None: cmd += ["--min-reward", str(args.min_reward)]
            if args.max_repetition_rate is not None: cmd += ["--max-repetition-rate", str(args.max_repetition_rate)]
        if args.agent_name == "noisy-walkthrough":
            if args.noise_rate is not None: cmd += ["--noise-rate", str(args.noise_rate)]
            if args.min_episode_length is not None: cmd += ["--min-episode-length", str(args.min_episode_length)]
        if not args.no_admissible_commands:
            cmd.append("--admissible-commands")

        try:
            subprocess.run(cmd, cwd=_ROOT, check=True)
        except subprocess.CalledProcessError:
            failures.append(env)
            continue

        if not traj_path.exists():
            failures.append(env)
            continue

        with traj_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        episodes = list(payload.get("episodes", []))[:args.episodes_per_game]
        if len(episodes) < args.episodes_per_game:
            prog = payload.get("in_progress_episode", {}) or {}
            if prog.get("trajectory"):
                episodes.append(prog)
        episodes = episodes[:args.episodes_per_game]

        if not episodes:
            failures.append(env)
            continue

        game_dir = out_dir / env
        game_dir.mkdir(parents=True, exist_ok=True)
        for i, ep in enumerate(episodes):
            steps = []
            total_r = 0.0
            for s in ep.get("trajectory", []):
                r = s.get("reward", 0.0)
                total_r += float(r) if isinstance(r, Number) else 0.0
                steps.append({"obs": s.get("observation", ""), "action": s.get("action", ""),
                             "reward": float(r) if isinstance(r, Number) else 0.0,
                             "done": bool(s.get("done", False)), "infos": s.get("infos", {}) or {}})
            rec = {"game": env, "steps": steps, "total_reward": total_r}
            pth = game_dir / f"episode_{i:05d}.json"
            with pth.open("w", encoding="utf-8") as f:
                json.dump(rec, f, indent=2, ensure_ascii=True)

    (out_dir / "manifest.json").write_text(json.dumps({"envs": envs, "failures": failures}, indent=2))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
