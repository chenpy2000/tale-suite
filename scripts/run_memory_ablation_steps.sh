#!/usr/bin/env bash
set -euo pipefail

# Run memory-agent across step budgets for a single seed.
# Usage:
#   bash scripts/run_memory_ablation_steps.sh
#   SEED=123 STEPS_LIST="20 50 100 200" COMPRESS_EVERY=10 ENVS="textworld" bash scripts/run_memory_ablation_steps.sh

STEPS_LIST=${STEPS_LIST:-"20 50 100 200"}
COMPRESS_EVERY=${COMPRESS_EVERY:-10}
ENVS=${ENVS:-"textworld"}
MODEL=${MODEL:-api-gpt-oss-120b}
API_KEY_ENV=${API_KEY_ENV:-TRITON_API_KEY}
RUN_ID=${RUN_ID:-"$(date +%Y%m%d_%H%M%S)"}
LOG_ROOT=${LOG_ROOT:-"logs/steps_ce${COMPRESS_EVERY}_${RUN_ID}"}

if [[ -z "${!API_KEY_ENV:-}" ]]; then
  echo "Error: API key env '$API_KEY_ENV' is not set."
  echo "Example: export $API_KEY_ENV=..."
  exit 1
fi

mkdir -p "$LOG_ROOT"

echo "Running step ablation with:"
echo "  STEPS_LIST=$STEPS_LIST"
echo "  COMPRESS_EVERY=$COMPRESS_EVERY"
echo "  ENVS=$ENVS"
echo "  MODEL=$MODEL"
echo "  LOG_ROOT=$LOG_ROOT"

for steps in $STEPS_LIST; do
  echo
  echo "=== steps=$steps ==="
  python benchmark.py \
    --agent agents/memory_agent.py \
    memory-agent \
    --envs "$ENVS" \
    --memory-variant structured \
    --compress-every "$COMPRESS_EVERY" \
    --nb-steps "$steps" \
    --admissible-commands \
    --use-llm-policy \
    --use-llm-parser \
    --use-llm-compressor \
    --llm-model "$MODEL" \
    --llm-api-key-env "$API_KEY_ENV" \
    --log-dir "$LOG_ROOT/steps_${steps}" \
    -ff

done

echo
echo "Completed step ablation."
echo "Logs are under: $LOG_ROOT"
