# Experiment Comparison Scripts

This directory contains scripts to run and compare baseline and compression with LLM experiments on multiple tasks.

## Quick Start

### Simple Usage (Bash Script)

Run 10 tasks comparing baseline vs compression+LLM:

```bash
./scripts/run_comparison.sh
```

### Custom Configuration

```bash
./scripts/run_comparison.sh \
  --num-tasks 10 \
  --nb-steps 100 \
  --envs TWCookingLevel1 \
  --log-dir logs/my_experiment
```

### Python Script (More Options)

```bash
python3 scripts/run_experiment_comparison.py \
  --num-tasks 10 \
  --nb-steps 100 \
  --envs TWCookingLevel1 TWCookingLevel2 \
  --log-dir logs/baseline_vs_compression \
  --compressor-model gpt-4o-mini \
  --llm-api-key-env OPENAI_API_KEY
```

## Options

### Common Options

- `--num-tasks N`: Number of tasks (different seeds) to run for each variant (default: 10)
- `--nb-steps N`: Maximum number of steps per game (default: 100)
- `--envs ENV1 ENV2 ...`: List of environments to test (default: TWCookingLevel1)
- `--log-dir DIR`: Base directory for logs (default: logs/experiment_comparison)
- `--dry-run`: Print commands without executing them

### Python Script Additional Options

- `--start-seed N`: Starting seed for experiments (default: 1)
- `--compress-every N`: Compress history every N steps (default: 8)
- `--output-file FILE`: Output CSV filename (default: experiment_results.csv)
- `--compressor-model MODEL`: Model for LLM compression (default: gpt-4o-mini)
- `--llm-api-url URL`: Custom LLM API URL
- `--llm-api-key-env VAR`: Environment variable name for API key (default: OPENAI_API_KEY)
- `--skip-baseline`: Skip running baseline experiments
- `--skip-compression`: Skip running compression experiments
- `--analyze-only`: Only analyze existing results without running new experiments

## Examples

### 1. Run Full Experiment (10 tasks on multiple environments)

```bash
python3 scripts/run_experiment_comparison.py \
  --num-tasks 10 \
  --nb-steps 100 \
  --envs TWCookingLevel1 TWCookingLevel2 TWCookingLevel3 \
  --log-dir logs/multi_env_experiment
```

### 2. Quick Test (3 tasks, 50 steps)

```bash
./scripts/run_comparison.sh \
  --num-tasks 3 \
  --nb-steps 50 \
  --log-dir logs/quick_test
```

### 3. Dry Run (See Commands Without Executing)

```bash
python3 scripts/run_experiment_comparison.py \
  --num-tasks 10 \
  --dry-run
```

### 4. Analyze Existing Results Only

```bash
python3 scripts/run_experiment_comparison.py \
  --log-dir logs/baseline_vs_compression \
  --analyze-only
```

### 5. Run Only Compression (Skip Baseline)

```bash
python3 scripts/run_experiment_comparison.py \
  --num-tasks 10 \
  --skip-baseline \
  --log-dir logs/compression_only
```

### 6. Use Custom LLM Model

```bash
python3 scripts/run_experiment_comparison.py \
  --num-tasks 10 \
  --compressor-model gpt-4 \
  --llm-api-key-env OPENAI_API_KEY
```

## Output

The script generates several outputs:

1. **Log files**: Detailed JSONL logs for each experiment in `{log-dir}/{variant}/{agent}/{env}/`
2. **CSV results**: Combined results in `{log-dir}/{timestamp}_experiment_results.csv`
3. **Summary CSV**: Aggregated statistics in `{log-dir}/{timestamp}_experiment_results_summary.csv`
4. **Console report**: Printed comparison showing:
   - Overall results by variant (mean, std, count)
   - Results by environment
   - Detailed results table
   - Winner announcement

### Sample Output

```
================================================================================
COMPARISON SUMMARY
================================================================================

🏆 Best performing variant: compression_llm
   Average normalized score: 0.7234

   baseline: 0.6891 ± 0.1234
   compression_llm: 0.7234 ± 0.1089
```

## Environment Setup

Before running experiments, ensure you have:

1. **Python dependencies installed**:
   ```bash
   pip install -r requirements.txt
   ```

2. **API key configured** (for LLM compression):
   ```bash
   export OPENAI_API_KEY="your_key_here"
   # or for custom API
   export TRITON_API_KEY="your_key_here"
   ```

3. **Tale-suite environments downloaded**:
   The environments will be automatically downloaded on first use.

## Troubleshooting

### Issue: "No API key found"
**Solution**: Set the appropriate environment variable:
```bash
export OPENAI_API_KEY="sk-..."
```

### Issue: "No results found to analyze"
**Solution**: Check that experiments completed successfully and log files exist in the log directory.

### Issue: Experiments taking too long
**Solution**: Reduce `--num-tasks` or `--nb-steps`:
```bash
./scripts/run_comparison.sh --num-tasks 3 --nb-steps 50
```

## File Structure

```
scripts/
├── run_experiment_comparison.py  # Main Python script
├── run_comparison.sh             # Convenience bash wrapper
└── EXPERIMENT_COMPARISON_README.md  # This file

logs/
└── baseline_vs_compression/      # Default log directory
    ├── baseline/                 # Baseline results
    │   └── tales_MemoryAgent_*/
    ├── compression_llm/          # Compression+LLM results
    │   └── tales_MemoryAgent_*/
    └── {timestamp}_experiment_results.csv  # Combined results
```

## Advanced Usage

### Running with Multiple Environments

Test across different difficulty levels:

```bash
python3 scripts/run_experiment_comparison.py \
  --num-tasks 10 \
  --envs TWCookingLevel1 TWCookingLevel2 TWCookingLevel3 \
  --nb-steps 100
```

### Custom Compression Frequency

Experiment with different compression intervals:

```bash
python3 scripts/run_experiment_comparison.py \
  --num-tasks 10 \
  --compress-every 4  # Compress every 4 steps instead of 8
```

### Using Different API Endpoints

For custom LLM API endpoints:

```bash
python3 scripts/run_experiment_comparison.py \
  --num-tasks 10 \
  --llm-api-url "https://api.your-llm-service.com/v1/chat/completions" \
  --llm-api-key-env YOUR_API_KEY_VAR \
  --compressor-model "your-model-name"
```

## Notes

- Each task runs with a different random seed to ensure statistical validity
- Results are automatically aggregated and averaged across all tasks
- The script handles failed experiments gracefully and reports them at the end
- Use `--dry-run` to preview commands before running expensive experiments
- Use `--analyze-only` to regenerate reports from existing log files
