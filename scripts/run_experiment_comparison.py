#!/usr/bin/env python3
"""
Script to run baseline and compression with LLM experiments on 10 tasks,
calculate average scores, and generate a comparison report.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


def build_baseline_command(args, game_seed):
    """Build command for baseline agent (memory agent without compression)."""
    cmd = [
        sys.executable,
        "benchmark.py",
        "--agent",
        "agents/memory_agent.py",
        "memory-agent",
        "--memory-variant",
        "baseline",
        "--nb-steps",
        str(args.nb_steps),
        "--log-dir",
        f"{args.log_dir}/baseline",
        "--game-seed",
        str(game_seed),
        "--use-llm-policy",
        "--llm-model",
        args.baseline_model,
        "--llm-api-url",
        args.llm_api_url,
        "--llm-api-key-env",
        args.llm_api_key_env,
    ]

    if args.envs:
        cmd.extend(["--envs", *args.envs])

    if args.admissible_commands:
        cmd.append("--admissible-commands")

    if args.force:
        cmd.append("-ff")

    return cmd


def build_compression_llm_command(args, game_seed):
    """Build command for compression with LLM agent."""
    cmd = [
        sys.executable,
        "benchmark.py",
        "--agent",
        "agents/memory_agent.py",
        "memory-agent",
        "--memory-variant",
        "compressed",
        "--compress-every",
        str(args.compress_every),
        "--nb-steps",
        str(args.nb_steps),
        "--log-dir",
        f"{args.log_dir}/compression_llm",
        "--game-seed",
        str(game_seed),
        "--use-llm-policy",
        "--use-llm-compressor",
    ]

    if args.envs:
        cmd.extend(["--envs", *args.envs])

    if args.admissible_commands:
        cmd.append("--admissible-commands")

    if args.compressor_model:
        cmd.extend(["--compressor-model", args.compressor_model])

    if args.llm_api_url:
        cmd.extend(["--llm-api-url", args.llm_api_url])

    if args.llm_api_key_env:
        cmd.extend(["--llm-api-key-env", args.llm_api_key_env])

    if args.policy_model:
        cmd.extend(["--llm-model", args.policy_model])

    if args.force:
        cmd.append("-ff")

    return cmd


def run_experiment(variant_name, cmd, dry_run=False):
    """Run a single experiment."""
    print(f"\n{'='*80}")
    print(f"Running {variant_name}")
    print(f"{'='*80}")
    print("$", " ".join(shlex.quote(c) for c in cmd))

    if dry_run:
        return 0

    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        print(f"⚠️  {variant_name} failed with exit code {completed.returncode}")
        return completed.returncode

    print(f"✓ {variant_name} completed successfully")
    return 0


def collect_results(log_dir, variant_name):
    """Collect results from log files."""
    results = []
    variant_dir = Path(log_dir) / variant_name

    if not variant_dir.exists():
        print(f"Warning: {variant_dir} does not exist")
        return pd.DataFrame()

    # Find all .jsonl files
    jsonl_files = list(variant_dir.glob("**/*.jsonl"))

    for jsonl_file in jsonl_files:
        try:
            # Parse file path to extract metadata
            path_parts = str(jsonl_file.relative_to(variant_dir)).split(os.sep)
            if len(path_parts) < 3:
                continue

            agent_name = path_parts[0]
            env_name = path_parts[1]
            params_file = path_parts[2]

            # Extract params from filename (e.g., "a0_sNone_steps100.jsonl")
            params_str = params_file.replace(".jsonl", "")
            parts = params_str.split("_")

            # Read JSONL data
            data = pd.read_json(jsonl_file, lines=True)

            if len(data) == 0:
                continue

            result = {
                "variant": variant_name,
                "agent": agent_name,
                "env_name": env_name,
                "game_seed": parts[1][1:] if len(parts) > 1 else "unknown",
                "nb_steps": data["Step"].max() if "Step" in data.columns else 0,
                "score": data["Score"].iloc[-1] if "Score" in data.columns else 0,
                "max_score": data["Max Score"].iloc[-1] if "Max Score" in data.columns else 0,
                "normalized_score": data["Normalized Score"].iloc[-1] if "Normalized Score" in data.columns else 0,
                "total_tokens": data["Token Usage"].sum() if "Token Usage" in data.columns else 0,
                "avg_tokens_per_step": data["Token Usage"].mean() if "Token Usage" in data.columns else 0,
            }
            results.append(result)

        except Exception as e:
            print(f"Error processing {jsonl_file}: {e}")
            continue

    return pd.DataFrame.from_records(results)


def generate_report(all_results, output_file):
    """Generate a comparison report."""
    if len(all_results) == 0:
        print("No results to report")
        return

    print(f"\n{'='*80}")
    print("EXPERIMENT COMPARISON REPORT")
    print(f"{'='*80}\n")

    # Overall statistics by variant
    print("=" * 80)
    print("OVERALL RESULTS BY VARIANT")
    print("=" * 80)
    variant_stats = all_results.groupby("variant").agg({
        "normalized_score": ["mean", "std", "count"],
        "score": ["mean", "std"],
        "total_tokens": "mean",
        "avg_tokens_per_step": "mean",
    })
    print(variant_stats)
    print()

    # Results by environment
    print("=" * 80)
    print("RESULTS BY ENVIRONMENT")
    print("=" * 80)
    env_stats = all_results.groupby(["variant", "env_name"]).agg({
        "normalized_score": ["mean", "std"],
        "score": "mean",
        "total_tokens": "mean",
    })
    print(env_stats)
    print()

    # Detailed results table
    print("=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(all_results.to_string(index=False))
    print()

    # Save to CSV
    all_results.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")

    # Create summary comparison
    summary = all_results.groupby("variant").agg({
        "normalized_score": ["mean", "std"],
        "score": "mean",
        "total_tokens": "mean",
        "avg_tokens_per_step": "mean",
    })

    summary_file = output_file.replace(".csv", "_summary.csv")
    summary.to_csv(summary_file)
    print(f"✓ Summary saved to: {summary_file}")

    # Print winner
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    mean_scores = all_results.groupby("variant")["normalized_score"].mean()
    best_variant = mean_scores.idxmax()
    best_score = mean_scores.max()

    print(f"\n🏆 Best performing variant: {best_variant}")
    print(f"   Average normalized score: {best_score:.4f}")
    print()

    for variant in mean_scores.index:
        score = mean_scores[variant]
        std = all_results.groupby("variant")["normalized_score"].std()[variant]
        print(f"   {variant}: {score:.4f} ± {std:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline and compression with LLM experiments and compare results."
    )
    parser.add_argument(
        "--envs",
        nargs="+",
        default=["TWCookingLevel1"],
        help="Environments to test on.",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=10,
        help="Number of tasks (different seeds) to run for each variant.",
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=1,
        help="Starting seed for game experiments.",
    )
    parser.add_argument(
        "--nb-steps",
        type=int,
        default=100,
        help="Maximum number of steps per game.",
    )
    parser.add_argument(
        "--compress-every",
        type=int,
        default=8,
        help="Compress history every N steps.",
    )
    parser.add_argument(
        "--log-dir",
        default="logs/experiment_comparison",
        help="Base directory for logs.",
    )
    parser.add_argument(
        "--output-file",
        default="experiment_results.csv",
        help="Output CSV file for results.",
    )
    parser.add_argument(
        "--admissible-commands",
        action="store_true",
        help="Use admissible commands.",
    )
    parser.add_argument(
        "--baseline-model",
        default="api-gpt-oss-120b",
        help="Model to use for baseline agent.",
    )
    parser.add_argument(
        "--compressor-model",
        default="api-gpt-oss-120b",
        help="Model to use for LLM compression.",
    )
    parser.add_argument(
        "--policy-model",
        default="api-gpt-oss-120b",
        help="Model to use for LLM policy in compression agent.",
    )
    parser.add_argument(
        "--llm-api-url",
        default="https://tritonai-api.ucsd.edu/v1/chat/completions",
        help="LLM API URL.",
    )
    parser.add_argument(
        "--llm-api-key-env",
        default="TRITON_API_KEY",
        help="Environment variable name for LLM API key.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip running baseline experiments.",
    )
    parser.add_argument(
        "--skip-compression",
        action="store_true",
        help="Skip running compression experiments.",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze existing results without running experiments.",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-run experiments even if results already exist.",
    )

    args = parser.parse_args()

    # Create log directory
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    # Prepare output file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output_file
    if not output_file.endswith(".csv"):
        output_file += ".csv"
    output_path = Path(args.log_dir) / f"{timestamp}_{output_file}"

    print(f"\n{'='*80}")
    print("EXPERIMENT CONFIGURATION")
    print(f"{'='*80}")
    print(f"Environments: {args.envs}")
    print(f"Number of tasks per variant: {args.num_tasks}")
    print(f"Seed range: {args.start_seed} to {args.start_seed + args.num_tasks - 1}")
    print(f"Steps per game: {args.nb_steps}")
    print(f"Compression frequency: every {args.compress_every} steps")
    print(f"Log directory: {args.log_dir}")
    print(f"Output file: {output_path}")
    print(f"Dry run: {args.dry_run}")
    print(f"{'='*80}\n")

    if not args.analyze_only:
        # Run experiments
        all_failed = []

        for seed in range(args.start_seed, args.start_seed + args.num_tasks):
            print(f"\n{'#'*80}")
            print(f"Task {seed - args.start_seed + 1}/{args.num_tasks} (seed={seed})")
            print(f"{'#'*80}")

            # Run baseline
            if not args.skip_baseline:
                cmd = build_baseline_command(args, seed)
                ret = run_experiment(f"Baseline (seed={seed})", cmd, args.dry_run)
                if ret != 0:
                    all_failed.append(f"baseline_seed{seed}")

            # Run compression with LLM
            if not args.skip_compression:
                cmd = build_compression_llm_command(args, seed)
                ret = run_experiment(f"Compression+LLM (seed={seed})", cmd, args.dry_run)
                if ret != 0:
                    all_failed.append(f"compression_llm_seed{seed}")

        if all_failed:
            print(f"\n⚠️  Warning: {len(all_failed)} experiments failed:")
            for failed in all_failed:
                print(f"   - {failed}")

        if args.dry_run:
            print("\nDry run complete. No experiments were executed.")
            return 0

    # Collect and analyze results
    print(f"\n{'='*80}")
    print("COLLECTING RESULTS")
    print(f"{'='*80}\n")

    all_results = []

    if not args.skip_baseline or args.analyze_only:
        print("Collecting baseline results...")
        baseline_results = collect_results(args.log_dir, "baseline")
        all_results.append(baseline_results)
        print(f"  Found {len(baseline_results)} baseline results")

    if not args.skip_compression or args.analyze_only:
        print("Collecting compression+LLM results...")
        compression_results = collect_results(args.log_dir, "compression_llm")
        all_results.append(compression_results)
        print(f"  Found {len(compression_results)} compression+LLM results")

    # Combine all results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)

        if len(combined_results) > 0:
            generate_report(combined_results, str(output_path))
        else:
            print("\n⚠️  No results found to analyze.")
            return 1
    else:
        print("\n⚠️  No results collected.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
