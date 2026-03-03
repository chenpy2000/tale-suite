#!/bin/bash
# Convenience script to run baseline vs compression+LLM comparison on 10 tasks

set -e

echo "Running Baseline vs Compression+LLM Comparison"
echo "=============================================="
echo ""

# Check if API key is set (adjust based on your needs)
if [ -z "$TRITON_API_KEY" ]; then
    echo "⚠️  Warning: No API key found in TRITON_API_KEY"
    echo "   Set the environment variable before running:"
    echo "   export TRITON_API_KEY=your_key_here"
    echo ""
fi

# Default values
NUM_TASKS=10
NB_STEPS=100
ENVS="TWCookingLevel1"
LOG_DIR="logs/baseline_vs_compression"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-tasks)
            NUM_TASKS="$2"
            shift 2
            ;;
        --nb-steps)
            NB_STEPS="$2"
            shift 2
            ;;
        --envs)
            ENVS="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --analyze-only)
            ANALYZE_ONLY="--analyze-only"
            shift
            ;;
        --force|-f)
            FORCE="--force"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--num-tasks N] [--nb-steps N] [--envs ENV] [--log-dir DIR] [--dry-run] [--analyze-only] [--force]"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Number of tasks: $NUM_TASKS"
echo "  Steps per task: $NB_STEPS"
echo "  Environments: $ENVS"
echo "  Log directory: $LOG_DIR"
echo ""

# Run the Python script
python3 scripts/run_experiment_comparison.py \
    --num-tasks $NUM_TASKS \
    --nb-steps $NB_STEPS \
    --envs $ENVS \
    --log-dir $LOG_DIR \
    --compress-every 8 \
    --compressor-model api-gpt-oss-120b \
    --llm-api-url https://tritonai-api.ucsd.edu/v1/chat/completions \
    --llm-api-key-env TRITON_API_KEY \
    $DRY_RUN \
    $ANALYZE_ONLY \
    $FORCE

echo ""
echo "=============================================="
echo "✓ Experiment comparison complete!"
echo "   Check $LOG_DIR for detailed results"
