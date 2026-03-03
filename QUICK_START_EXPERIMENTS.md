# Quick Start Guide: Running Baseline vs Compression+LLM Experiments

## What I Created for You

I've created a comprehensive experiment framework that:
1. Runs **baseline** and **compression with LLM** agents on multiple tasks
2. Automatically runs 10 different game instances (using different seeds)
3. Collects and analyzes all results
4. Calculates average scores and generates comparison reports

## Files Created

1. **`scripts/run_experiment_comparison.py`** - Main Python script
2. **`scripts/run_comparison.sh`** - Simple bash wrapper
3. **`scripts/EXPERIMENT_COMPARISON_README.md`** - Detailed documentation

## Quickest Way to Run

### Option 1: Using the Bash Script (Simplest)

```bash
cd /root/tale-suite
./scripts/run_comparison.sh
```

This will:
- Run 10 tasks (seeds 1-10) for both baseline and compression+LLM
- Test on TWCookingLevel1 environment
- Use 100 steps per game
- Save results to `logs/baseline_vs_compression/`
- Generate a CSV report with averages and comparison

### Option 2: Using Python Script Directly (More Control)

```bash
cd /root/tale-suite
python3 scripts/run_experiment_comparison.py \
  --num-tasks 10 \
  --nb-steps 100 \
  --envs TWCookingLevel1 \
  --log-dir logs/baseline_vs_compression
```

## Before Running: Set Up API Key

For the compression+LLM variant to work, you need an OpenAI API key:

```bash
export OPENAI_API_KEY="sk-your-key-here"
```

Or if using a custom API:
```bash
export TRITON_API_KEY="your-key-here"
```

Then modify the script call to use the appropriate key environment variable.

## Example: Quick Test Run (3 tasks, 50 steps)

To test the system without waiting too long:

```bash
cd /root/tale-suite
python3 scripts/run_experiment_comparison.py \
  --num-tasks 3 \
  --nb-steps 50 \
  --envs TWCookingLevel1
```

This runs much faster and helps verify everything works.

## Example: Full Experiment (10 tasks, multiple environments)

```bash
cd /root/tale-suite
python3 scripts/run_experiment_comparison.py \
  --num-tasks 10 \
  --nb-steps 100 \
  --envs TWCookingLevel1 TWCookingLevel2 TWCookingLevel3 \
  --log-dir logs/full_experiment
```

## What the Output Looks Like

After running, you'll see:

```
================================================================================
OVERALL RESULTS BY VARIANT
================================================================================
                    normalized_score                   score             ...
                                mean       std count   mean       std    ...
variant                                                                   ...
baseline                      0.6891  0.1234    10  45.2  12.3         ...
compression_llm                0.7234  0.1089    10  48.7  10.8         ...

================================================================================
COMPARISON SUMMARY
================================================================================

🏆 Best performing variant: compression_llm
   Average normalized score: 0.7234

   baseline: 0.6891 ± 0.1234
   compression_llm: 0.7234 ± 0.1089
```

## Output Files

All results are saved to your log directory:

- `{timestamp}_experiment_results.csv` - Full detailed results
- `{timestamp}_experiment_results_summary.csv` - Aggregated statistics
- `baseline/` - Individual baseline experiment logs
- `compression_llm/` - Individual compression+LLM experiment logs

## Testing Without Running (Dry Run)

To see what commands would be executed without actually running:

```bash
python3 scripts/run_experiment_comparison.py \
  --num-tasks 10 \
  --dry-run
```

## Analyzing Existing Results

If you already have results and just want to regenerate the report:

```bash
python3 scripts/run_experiment_comparison.py \
  --log-dir logs/baseline_vs_compression \
  --analyze-only
```

## Common Issues & Solutions

### "No API key found"
Set the OpenAI API key:
```bash
export OPENAI_API_KEY="sk-..."
```

### Want to run only baseline or only compression?
```bash
# Only baseline
python3 scripts/run_experiment_comparison.py --skip-compression

# Only compression
python3 scripts/run_experiment_comparison.py --skip-baseline
```

### Experiments take too long?
Reduce tasks or steps:
```bash
python3 scripts/run_experiment_comparison.py --num-tasks 3 --nb-steps 50
```

## Next Steps

1. **Set your API key** (if using OpenAI models)
2. **Run a quick test** with 3 tasks to verify everything works
3. **Run the full experiment** with 10 tasks
4. **Check the results** in the generated CSV files

For more details, see `scripts/EXPERIMENT_COMPARISON_README.md`.
