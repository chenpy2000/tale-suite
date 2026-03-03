#!/usr/bin/env python3
import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def build_command(args, variant):
    cmd = [
        sys.executable,
        "benchmark.py",
        "--agent",
        "agents/memory_agent.py",
        "memory-agent",
        "--memory-variant",
        variant,
        "--compress-every",
        str(args.compress_every),
        "--nb-steps",
        str(args.nb_steps),
        "--log-dir",
        args.log_dir,
    ]

    if args.envs:
        cmd.extend(["--envs", *args.envs])

    if args.admissible_commands:
        cmd.append("--admissible-commands")

    if variant == "rag":
        cmd.append("--use-retrieval")

    if variant == "option":
        cmd.extend(["--use-option-module", "--use-retrieval"])

    if args.use_llm_policy:
        cmd.extend(
            [
                "--use-llm-policy",
                "--llm-model",
                args.llm_model,
                "--llm-api-url",
                args.llm_api_url,
                "--llm-api-key-env",
                args.llm_api_key_env,
                "--llm-timeout",
                str(args.llm_timeout),
            ]
        )
    if args.use_llm_parser:
        cmd.append("--use-llm-parser")
    if args.use_llm_compressor:
        cmd.append("--use-llm-compressor")
    if args.parser_model:
        cmd.extend(["--parser-model", args.parser_model])
    if args.compressor_model:
        cmd.extend(["--compressor-model", args.compressor_model])

    if args.prompt_variant:
        cmd.extend(["--prompt-variant", args.prompt_variant])

    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmark.py with MemoryAgent variants."
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["baseline", "compressed", "structured", "rag", "option"],
        choices=["baseline", "compressed", "structured", "rag", "option"],
        help="Variants to benchmark.",
    )
    parser.add_argument(
        "--envs",
        nargs="+",
        default=["TWCookingLevel1"],
        help="Environments to benchmark.",
    )
    parser.add_argument("--nb-steps", type=int, default=100)
    parser.add_argument("--compress-every", type=int, default=8)
    parser.add_argument("--log-dir", default="logs/memory_variants")
    parser.add_argument("--admissible-commands", action="store_true")
    parser.add_argument("--use-llm-policy", action="store_true")
    parser.add_argument("--llm-model", default="api-llama-4-scout")
    parser.add_argument(
        "--llm-api-url",
        default="https://tritonai-api.ucsd.edu/v1/chat/completions",
    )
    parser.add_argument("--llm-api-key-env", default="TRITON_API_KEY")
    parser.add_argument("--llm-timeout", type=int, default=30)
    parser.add_argument("--use-llm-parser", action="store_true")
    parser.add_argument("--use-llm-compressor", action="store_true")
    parser.add_argument("--parser-model", default=None)
    parser.add_argument("--compressor-model", default=None)
    parser.add_argument("--prompt-variant", default=None,
                        help="Compression prompt variant (default, v1, v2, v3)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    args = parser.parse_args()

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    for variant in args.variants:
        cmd = build_command(args, variant)
        print(f"\n[variant={variant}]")
        print("$", " ".join(shlex.quote(c) for c in cmd))

        if args.dry_run:
            continue

        completed = subprocess.run(cmd, check=False)
        if completed.returncode != 0:
            print(f"Variant {variant} failed with exit code {completed.returncode}")
            return completed.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
