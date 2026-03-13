#!/usr/bin/env bash
set -euo pipefail

# Run memory-agent ablations across seeds:
# 1) policy-only
# 2) full-memory (policy + parser + compressor)
# 3) rag (full-memory + retrieval)
#
# Usage:
#   bash scripts/run_memory_ablation.sh
#   SEEDS="1 7 42" STEPS=200 MODEL=api-gpt-oss-120b bash scripts/run_memory_ablation.sh
#   ENVS="textworld" LOG_ROOT="logs/ablation" bash scripts/run_memory_ablation.sh

SEEDS=${SEEDS:-"1 7 42 123 2024"}
STEPS=${STEPS:-200}
MODEL=${MODEL:-api-gpt-oss-120b}
ENVS=${ENVS:-"textworld"}
COMPRESS_EVERY=${COMPRESS_EVERY:-8}
LOG_ROOT=${LOG_ROOT:-"logs"}
API_KEY_ENV=${API_KEY_ENV:-TRITON_API_KEY}

if [[ -z "${!API_KEY_ENV:-}" ]]; then
  echo "Error: API key env '$API_KEY_ENV' is not set."
  echo "Example: export $API_KEY_ENV=..."
  exit 1
fi

run_one() {
  local seed="$1"
  local mode="$2"

  local common=(
    python benchmark.py
    --agent agents/memory_agent.py
    memory-agent
    --envs "$ENVS"
    --nb-steps "$STEPS"
    --compress-every "$COMPRESS_EVERY"
    --admissible-commands
    --use-llm-policy
    --llm-model "$MODEL"
    --llm-api-key-env "$API_KEY_ENV"
    --game-seed "$seed"
    -ff
  )

  case "$mode" in
    policy_only)
      "${common[@]}" \
        --memory-variant structured \
        --log-dir "$LOG_ROOT/exp_policy_only"
      ;;

    full_memory)
      "${common[@]}" \
        --memory-variant structured \
        --use-llm-parser \
        --use-llm-compressor \
        --log-dir "$LOG_ROOT/exp_full_memory"
      ;;

    rag)
      "${common[@]}" \
        --memory-variant rag \
        --use-retrieval \
        --use-llm-parser \
        --use-llm-compressor \
        --log-dir "$LOG_ROOT/exp_rag"
      ;;

    *)
      echo "Unknown mode: $mode"
      exit 2
      ;;
  esac
}

echo "Running ablation with:"
echo "  SEEDS=$SEEDS"
echo "  STEPS=$STEPS"
echo "  MODEL=$MODEL"
echo "  ENVS=$ENVS"
echo "  COMPRESS_EVERY=$COMPRESS_EVERY"
echo "  LOG_ROOT=$LOG_ROOT"
echo "  API_KEY_ENV=$API_KEY_ENV"

total=0
for s in $SEEDS; do
  echo
  echo "=== Seed: $s | policy_only ==="
  run_one "$s" policy_only
  total=$((total + 1))

  echo
  echo "=== Seed: $s | full_memory ==="
  run_one "$s" full_memory
  total=$((total + 1))

  echo
  echo "=== Seed: $s | rag ==="
  run_one "$s" rag
  total=$((total + 1))
done

echo

echo "Completed $total runs."
echo "You can aggregate results with:"
echo "  python scripts/summarize_memory_variants.py --logs-root $LOG_ROOT --output-dir $LOG_ROOT/summary --only-memory-agent"
