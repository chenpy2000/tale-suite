#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate memory-agent benchmark summaries into side-by-side CSVs."
    )
    parser.add_argument(
        "--logs-root",
        default="logs",
        help="Root directory to scan for benchmark summary JSON files. Default: %(default)s",
    )
    parser.add_argument(
        "--output-dir",
        default="logs/summary",
        help="Directory where CSV outputs are written. Default: %(default)s",
    )
    parser.add_argument(
        "--only-memory-agent",
        action="store_true",
        help="Keep only runs with folder names starting with tales_MemoryAgent_.",
    )
    parser.add_argument(
        "--only-exp-dirs",
        action="store_true",
        help="Keep only runs under experiment directories (e.g., exp_*).",
    )
    parser.add_argument(
        "--exp-prefix",
        default="exp_",
        help="Prefix used to detect experiment directories. Default: %(default)s",
    )
    return parser.parse_args()


def looks_like_summary(payload: Dict) -> bool:
    required = {"env_name", "norm_score", "status"}
    return required.issubset(payload.keys())


def extract_run_label(run_dir_name: str) -> str:
    prefix = "tales_MemoryAgent_"
    if run_dir_name.startswith(prefix):
        return run_dir_name[len(prefix) :]
    return run_dir_name


def find_summary_files(logs_root: Path) -> List[Path]:
    files = []
    for path in logs_root.rglob("*.json"):
        # Keep only env summary files. Skip obvious non-summary files early.
        if path.name.startswith("a") and "_steps" in path.name:
            files.append(path)
    return files


def load_records(
    files: List[Path], only_memory_agent: bool, only_exp_dirs: bool, exp_prefix: str
) -> pd.DataFrame:
    rows: List[Dict] = []

    for path in files:
        # Expected shape: <logs_root>/<run_dir>/<env>/<summary.json>
        # or <logs_root>/<subdir>/<run_dir>/<env>/<summary.json>
        parents = path.parents
        if len(parents) < 3:
            continue

        env_name = path.parent.name
        run_dir = path.parent.parent.name
        exp_group = path.parent.parent.parent.name if len(parents) >= 4 else ""

        if only_memory_agent and not run_dir.startswith("tales_MemoryAgent_"):
            continue
        if only_exp_dirs and not exp_group.startswith(exp_prefix):
            continue

        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue

        if not looks_like_summary(payload):
            continue

        rows.append(
            {
                "run_dir": run_dir,
                "exp_group": exp_group,
                "run_label": f"{exp_group}/{extract_run_label(run_dir)}"
                if exp_group
                else extract_run_label(run_dir),
                "env_name": payload.get("env_name", env_name),
                "env_params": payload.get("env_params"),
                "status": payload.get("status"),
                "norm_score": payload.get("norm_score"),
                "highscore": payload.get("highscore"),
                "max_score": payload.get("max_score"),
                "nb_steps": payload.get("nb_steps"),
                "nb_invalid_actions": payload.get("nb_invalid_actions"),
                "duration": payload.get("duration"),
                "token_efficiency": payload.get("token_efficiency"),
                "doom_loop_count": payload.get("doom_loop_count"),
                "summary_file": str(path),
                "mtime": path.stat().st_mtime,
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # If same run/env/env_params appears multiple times, keep latest summary file.
    df = df.sort_values("mtime").drop_duplicates(
        subset=["exp_group", "run_dir", "env_name", "env_params"], keep="last"
    )

    # Stable sorting for readability.
    df = df.sort_values(
        ["exp_group", "run_label", "env_name", "env_params"]
    ).reset_index(drop=True)
    return df


def build_outputs(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pivot = df.pivot_table(
        index=["env_name", "env_params"],
        columns="run_label",
        values="norm_score",
        aggfunc="first",
    ).reset_index()

    means = (
        df.groupby("run_label", as_index=False)
        .agg(
            mean_norm_score=("norm_score", "mean"),
            mean_highscore=("highscore", "mean"),
            mean_steps=("nb_steps", "mean"),
            mean_invalid_actions=("nb_invalid_actions", "mean"),
            mean_duration_sec=("duration", "mean"),
            mean_token_efficiency=("token_efficiency", "mean"),
            mean_doom_loop_count=("doom_loop_count", "mean"),
            num_envs=("env_name", "count"),
        )
        .sort_values("mean_norm_score", ascending=False)
    )

    return pivot, means


def with_seed_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["seed"] = out["env_params"].map(extract_seed_from_env_params)
    return out


def extract_seed_from_env_params(env_params: str) -> str:
    # Example formats: a1_s42_steps200, a1_sNone_steps100
    text = str(env_params or "")
    match = re.search(r"_s([^_]+)_steps", text)
    if not match:
        return "unknown"
    return match.group(1)


def summarize_with_ci(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    grouped = (
        df.groupby(group_cols, as_index=False)
        .agg(
            n=("norm_score", "count"),
            mean_norm_score=("norm_score", "mean"),
            std_norm_score=("norm_score", "std"),
            min_norm_score=("norm_score", "min"),
            max_norm_score=("norm_score", "max"),
        )
        .fillna({"std_norm_score": 0.0})
    )
    grouped["sem_norm_score"] = grouped["std_norm_score"] / grouped["n"].pow(0.5)
    grouped["ci95_low"] = grouped["mean_norm_score"] - 1.96 * grouped["sem_norm_score"]
    grouped["ci95_high"] = grouped["mean_norm_score"] + 1.96 * grouped["sem_norm_score"]
    return grouped


def main():
    args = parse_args()
    logs_root = Path(args.logs_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = find_summary_files(logs_root)
    df = load_records(
        files,
        only_memory_agent=args.only_memory_agent,
        only_exp_dirs=args.only_exp_dirs,
        exp_prefix=args.exp_prefix,
    )

    if df.empty:
        print(f"No summary files found under: {logs_root}")
        return 1

    pivot, means = build_outputs(df)
    df_seeded = with_seed_column(df)

    long_csv = output_dir / "memory_variants_long.csv"
    pivot_csv = output_dir / "memory_variants_side_by_side.csv"
    means_csv = output_dir / "memory_variants_means.csv"
    per_seed_env_csv = output_dir / "memory_variants_per_seed_env.csv"
    per_seed_means_csv = output_dir / "memory_variants_per_seed_means.csv"
    per_env_means_csv = output_dir / "memory_variants_per_env_means.csv"
    ci_overall_csv = output_dir / "memory_variants_ci_overall.csv"
    ci_per_env_csv = output_dir / "memory_variants_ci_per_env.csv"

    df.to_csv(long_csv, index=False)
    pivot.to_csv(pivot_csv, index=False)
    means.to_csv(means_csv, index=False)
    df_seeded.to_csv(per_seed_env_csv, index=False)

    per_seed_means = (
        df_seeded.groupby(["run_label", "seed"], as_index=False)
        .agg(
            mean_norm_score=("norm_score", "mean"),
            mean_highscore=("highscore", "mean"),
            mean_steps=("nb_steps", "mean"),
            num_envs=("env_name", "count"),
        )
        .sort_values(["run_label", "seed"])
    )
    per_seed_means.to_csv(per_seed_means_csv, index=False)

    per_env_means = (
        df_seeded.groupby(["run_label", "env_name"], as_index=False)
        .agg(
            mean_norm_score=("norm_score", "mean"),
            std_norm_score=("norm_score", "std"),
            mean_steps=("nb_steps", "mean"),
            num_seeds=("seed", "nunique"),
        )
        .sort_values(["run_label", "env_name"])
    )
    per_env_means.to_csv(per_env_means_csv, index=False)

    ci_overall = summarize_with_ci(df_seeded, ["run_label"])
    ci_overall.to_csv(ci_overall_csv, index=False)

    ci_per_env = summarize_with_ci(df_seeded, ["run_label", "env_name"])
    ci_per_env.to_csv(ci_per_env_csv, index=False)

    print(f"Wrote {long_csv}")
    print(f"Wrote {pivot_csv}")
    print(f"Wrote {means_csv}")
    print(f"Wrote {per_seed_env_csv}")
    print(f"Wrote {per_seed_means_csv}")
    print(f"Wrote {per_env_means_csv}")
    print(f"Wrote {ci_overall_csv}")
    print(f"Wrote {ci_per_env_csv}")
    print(f"Rows: {len(df)}, Runs: {df['run_label'].nunique()}, Envs: {df['env_name'].nunique()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
