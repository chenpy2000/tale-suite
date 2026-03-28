# Diagnostic Evaluation Plotting System

Complete toolkit for parsing and visualizing diagnostic evaluation results from TALES benchmark.

## Quick Start

```bash
# 1. Parse terminal output
python parse_diagnostic_results.py \
  --input vqvae_diagnostic.txt \
  --output vqvae_results.json \
  --skills diagnostic_tasks.json

# 2. Generate plots
python plot_diagnostic.py radar \
  --data vqvae_results.json graph_results.json react_results.json \
  --output skill_radar.png
```

---

## File Overview

- **parse_diagnostic_results.py**: Parse terminal output → JSON
- **plot_diagnostic.py**: Generate 4 types of plots
- **diagnostic_tasks.json**: Skill category mappings

---

## Step 1: Parse Terminal Output

### Input Format
Terminal output from benchmark runs (see `vqvae_diagnostic.txt` for example).

**Handles interrupted/restarted runs**: If you had to restart the benchmark and have multiple terminal output sections, the parser will automatically combine them. It keeps the most recent result for each environment.

### Command
```bash
python parse_diagnostic_results.py \
  --input <terminal_output.txt> \
  --output <results.json> \
  --skills diagnostic_tasks.json
```

**For interrupted runs**: Simply paste all terminal sections into one file. The parser will:
- Extract all environment results
- Handle duplicates (keeps last occurrence)
- Calculate statistics from parsed data (doesn't rely on summary lines)

### Output JSON Structure
```json
{
  "agent": "LLMVQVAE",
  "results": [
    {
      "env": "JerichoEnv905",
      "score": 0,
      "max_score": 1,
      "normalized_score": 0.0,
      "time_seconds": 219,
      "doom_loops": 0
    },
    ...
  ],
  "mean_score": 0.0796,
  "total_time_minutes": 52.75,
  "total_doom_loops": 0,
  "skill_scores": {
    "spatial": {"mean": 0.1029, "count": 1},
    "deductive": {"mean": 0.0, "count": 0},
    "inductive": {"mean": 0.4467, "count": 3},
    "grounded": {"mean": 0.0, "count": 0}
  }
}
```

---

## Step 2: Generate Plots

### Plot 1: Radar Chart (Skill Profiles)

Shows each agent's performance across 4 reasoning skills.

```bash
python plot_diagnostic.py radar \
  --data vqvae_results.json graph_results.json react_results.json \
  --output plots/skill_radar.png
```

**Requirements:**
- Input: Parsed JSON files with `skill_scores`
- One file per agent
- All agents plotted on same radar chart

---

### Plot 2: Line Plot (Step Budget Comparison)

Compares agent performance across different episode lengths.

```bash
python plot_diagnostic.py step-budget \
  --data step_budget_data.json \
  --output plots/step_budget.png
```

**Input JSON Format:**
```json
{
  "nb_steps": [20, 50, 100, 200],
  "agents": {
    "Graph": [35.71, 68.67, 76.00, 80.73],
    "VQ-VAE": [22.41, 35.73, 38.32, 44.45],
    "Graph+VQ-VAE": [28.50, 52.20, 57.16, 62.59]
  }
}
```

**How to create this file:**
- Run benchmark with different `--nb-steps` values
- Manually compile mean scores into JSON
- See `step_budget_example.json` for template

---

### Plot 3: Bar Chart (Computation Efficiency)

Shows wall time for each agent at different step budgets.

```bash
python plot_diagnostic.py efficiency \
  --data efficiency_data.json \
  --output plots/efficiency.png
```

**Input JSON Format:**
```json
{
  "nb_steps": [20, 50, 100, 200],
  "agents": {
    "Graph": [25.3, 54.2, 109.3, 197.3],
    "VQ-VAE": [22.5, 73.7, 141.4, 215.9]
  }
}
```

**Values are in minutes** (total wall time for all environments).

See `efficiency_example.json` for template.

---

### Plot 4: Bar Chart (Per-Skill Comparison)

Compares agents across the 4 reasoning skills side-by-side.

```bash
python plot_diagnostic.py skill-comparison \
  --data vqvae_results.json graph_results.json react_results.json \
  --output plots/skill_comparison.png
```

**Requirements:**
- Input: Parsed JSON files with `skill_scores`
- Shows bars for each agent grouped by skill
- Useful for seeing which agent excels at which skill

---

## Complete Workflow Example

### Scenario: Compare 3 agents (Graph, VQ-VAE, ReAct)

**Step 1: Run evaluations**
```bash
# Graph agent
python benchmark.py --agent agents/graph_agent.py graph \
  --envs JerichoEnv905 ... ALFWorldPickTwoObjAndPlaceSeen \
  --nb-steps 50 > graph_diagnostic.txt

# VQ-VAE agent  
python benchmark.py --agent agents/llm_vqvae_agent.py llm-vqvae \
  --envs JerichoEnv905 ... ALFWorldPickTwoObjAndPlaceSeen \
  --nb-steps 50 > vqvae_diagnostic.txt

# ReAct agent
python benchmark.py --agent agents/react_agent.py react \
  --envs JerichoEnv905 ... ALFWorldPickTwoObjAndPlaceSeen \
  --nb-steps 50 > react_diagnostic.txt
```

**Step 2: Parse results**
```bash
python parse_diagnostic_results.py --input graph_diagnostic.txt --output graph_results.json
python parse_diagnostic_results.py --input vqvae_diagnostic.txt --output vqvae_results.json
python parse_diagnostic_results.py --input react_diagnostic.txt --output react_results.json
```

**Step 3: Generate radar chart**
```bash
python plot_diagnostic.py radar \
  --data graph_results.json vqvae_results.json react_results.json \
  --output plots/skill_profiles.png
```

**Step 4: Generate skill comparison**
```bash
python plot_diagnostic.py skill-comparison \
  --data graph_results.json vqvae_results.json react_results.json \
  --output plots/skill_bars.png
```

**Step 5: Create step budget data (manual)**

Create `step_budget.json`:
```json
{
  "nb_steps": [20, 50, 100, 200],
  "agents": {
    "Graph": [30.2, 65.1, 73.8, 79.5],
    "VQ-VAE": [20.1, 33.2, 36.7, 42.3],
    "ReAct": [5.2, 18.3, 17.9, 19.1]
  }
}
```

Then plot:
```bash
python plot_diagnostic.py step-budget --data step_budget.json --output plots/step_budget.png
```

---

## Tips

### Adding New Agents

1. Run benchmark, save terminal output
2. Parse with `parse_diagnostic_results.py`
3. Add to existing plot commands

### Customizing Colors

Edit `AGENT_COLORS` dict in `plot_diagnostic.py`:
```python
AGENT_COLORS = {
    'my-agent': '#FF5733',  # Add your agent
    ...
}
```

### Debugging Parser

If parsing fails:
```bash
# Check what the parser extracts
python parse_diagnostic_results.py --input test.txt --output /dev/null
```

Look for pattern mismatches in terminal output format.

---

## Data Format Reference

### diagnostic_tasks.json
Maps environments to reasoning skills. **Don't modify unless changing test suite.**

### Parsed Results JSON
Generated by parser. Contains per-environment results + skill aggregations.

### Step Budget JSON
Manually created. Compile mean scores from multiple runs.

### Efficiency JSON  
Manually created. Wall times in minutes from benchmark logs.

---

## Troubleshooting

**"Warning: X has no skill_scores"**
- Parser couldn't map environments to skills
- Check that `diagnostic_tasks.json` includes all environments in your test

**Parser finds 0 environments**
- Terminal output format doesn't match regex
- Verify output has lines like: `EnvName   Steps:   X/  Y  Time: H:MM:SS ...`

**Plot is empty**
- Check JSON data format matches expected structure
- Ensure agent names in JSON match `AGENT_COLORS` keys (case-insensitive partial match)

**Agent bars missing in skill comparison**
- JSON file doesn't have `skill_scores` field
- Re-run parser with `--skills` flag

**Interrupted runs show duplicate environments**
- Parser keeps the LAST occurrence of each environment
- This is correct behavior - final run overwrites earlier attempts
- To verify: Check `num_environments` in output JSON

**Mean score doesn't match benchmark summary**
- Parser calculates from parsed data, not summary line
- This is more accurate for interrupted runs
- Summary line may be from partial run

---

## Files Included

```
parse_diagnostic_results.py  # Parser
plot_diagnostic.py          # Plotting
diagnostic_tasks.json       # Skill mappings
step_budget_example.json    # Example data
efficiency_example.json     # Example data
vqvae_diagnostic.txt        # Example terminal output
README_PLOTTING.md          # This file
```

---

## Next Steps

1. Parse your agent results
2. Generate radar chart to see skill profiles
3. Create step budget data for performance curves
4. Generate all 4 plots for presentation

Good luck!
