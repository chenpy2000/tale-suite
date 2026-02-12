#!/bin/bash
# Script to run baseline agent and verify metrics

set -e

# Load environment variables from .env if present
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

# Determine script directory to support execution from anywhere
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Define variables
AGENT="agents/baseline.py"
AGENT_NAME="baseline-cot"
# Use available tasks for testing
ENVS=("TWCookingLevel1" "ScienceWorldBoil") 

# Number of seeds for consistency check
SEEDS=(1) 

echo "Running Baseline Agent ($AGENT_NAME) on envs: ${ENVS[*]}"
echo "Using seeds: ${SEEDS[*]}"

# Create a temporary log file for aggregation
RESULTS_FILE="baseline_results.txt"
> $RESULTS_FILE

for seed in "${SEEDS[@]}"; do
    echo "Running seed $seed..."
    OUTPUT_FILE="logs/run_s${seed}.log"
    
    # Run benchmark with -ff to force overwrite logs
    python3 benchmark.py \
        --agent $AGENT \
        $AGENT_NAME \
        --envs "${ENVS[@]}" \
        --seed $seed \
        --nb-steps 10 \
        -ff \
        --logging-level INFO | tee $OUTPUT_FILE
    
    # Extract Score (Success Metric)
    SCORE=$(grep "Mean score" $OUTPUT_FILE | awk '{print $7}' | tr -d '%')
    if [ -z "$SCORE" ]; then SCORE="0.0"; fi
    
    # Extract Invalid Actions (Error Metric)
    INVALID=$(grep "Total .* invalid actions" $OUTPUT_FILE | awk '{print $2}')
    if [ -z "$INVALID" ]; then INVALID="0"; fi

    echo "$seed,$SCORE,$INVALID" >> $RESULTS_FILE
done

echo "=== Results ==="
cat $RESULTS_FILE

python3 -c "
import numpy as np
import sys
data = [line.strip().split(',') for line in sys.stdin if line.strip()]
if data:
    scores = [float(row[1]) for row in data]
    invalids = [float(row[2]) for row in data]
    print(f'Average Score (Success): {np.mean(scores):.2f}%')
    print(f'Score Consistency (Std Dev): {np.std(scores):.2f}')
    print(f'Average Invalid Actions (Error): {np.mean(invalids):.2f}')
else:
    print('No data collected.')
" < $RESULTS_FILE
