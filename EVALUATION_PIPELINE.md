# Evaluation Pipeline

Evaluates text-game agents across **4 skills** (spatial, deductive, inductive, grounded) on a fixed **evaluation subset** of 20 environments (5 per skill). Produces diagnostic scores, full benchmarks, transfer analysis, and comparison plots.

---

## Overview: What This Pipeline Does

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Categorize     → Assigns each env to a primary skill + difficulty        │
│ 2. Select diag    → Picks 20 tasks (5 per skill) for diagnostic evaluation  │
│ 3. Diagnostic     → Runs graph, llm-vqvae, memory-agent on diagnostic tasks │
│ 4. Full benchmark → Same agents on full 20-env subset                        │
│ 5. Transfer       → Predicts full-task scores from diagnostic skill profile │
│ 6. Hybrid agents  → graph-vqvae, memory-react, full-hybrid (if API_KEY set)  │
│ 7. Plots          → diagnostic_comparison, skill_profiles, hybrid_comparison  │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key files:**
- `data/task_categories.json` — skill/difficulty per env (from step 1)
- `data/evaluation_subset.json` — 20 envs used for evaluation (from step 1)
- `data/diagnostic_tasks.json` — tasks per skill for diagnostic runs (from step 2)
- `logs/*_diagnostic.json` — per-agent diagnostic metrics (from step 3)
- `plots/` — PNG comparison charts (from step 7)

---

## Quick Start (Full Pipeline)

**Standard** (nb-steps 100 full / 50 diagnostic):

```bash
./scripts/run_full_evaluation.sh --api-key "your-tritonai-key"
```

**Higher quality** (nb-steps 150 full / 75 diagnostic):

```bash
./scripts/run_full_evaluation.sh --api-key "your-tritonai-key" --nb-steps 150 --nb-steps-diagn 75
```

**TWCooking-only:**

```bash
./scripts/run_full_evaluation.sh --api-key "your-tritonai-key" --diagnostic-config config/evaluation_config.yaml
```

---

## Complete Start-to-Finish Guide

All commands use the **evaluation subset** (20 envs: 5 spatial, 5 deductive, 5 inductive, 5 grounded). Parameters below use explicit arguments.

### 0. Prerequisites

```bash
cd /root/tale-suite   # or your repo path

pip install -r requirements.txt
```

**Note:** `collect_data.py` (step 2) uses rule-based collectors—no API key needed. Steps 4+ use LLM agents and **require** a TritonAI API key.

---

### 1. Create Evaluation Subset (if missing)

**Script:** `scripts/categorize_tasks.py`

**What it does:** Scans all TALES environments (Jericho, ALFWorld, ScienceWorld, TextWorld, etc.) and assigns each to a primary skill (spatial, deductive, inductive, grounded) and difficulty. Uses heuristics per env family (e.g. TWCooking levels → difficulty).

**Outputs:**
- `data/task_categories.json` — `{env_name: {primary_skill, difficulty, skill_weights}}`
- `data/evaluation_subset.json` — 20 envs (5 per skill) for evaluation

```bash
python scripts/categorize_tasks.py -o data/task_categories.json
```

---

### 2. VQ-VAE Training (required for llm-vqvae and graph-vqvae hybrid)

**Location:** `latent-action/` (run from repo root or `cd latent-action`)

**What it does:** Trains a VQ-VAE on (observation, action) windows from trajectories. The model learns patterns like "after examining the cookbook, taking it is often good." At inference, it scores admissible actions to guide the LLM.

**Substeps:**
- **2a. Collect** — `collect_data.py` runs rule-based collectors (diverse, noisy-walkthrough) that play games without LLMs. Outputs `episode_*.json` per env.
- **2b. Clean** — `clean_trajectories.py` drops episodes that are too short, low-reward, or repetitive.
- **2c. Train** — `train_vqvae.py` trains on sliding windows; `--window-size` must match inference (e.g. 5).

```bash
cd latent-action

python collect_data.py --agent agents/collectors.py --agent-name diverse-collector \
  --envs JerichoEnv905 JerichoEnvAcorncourt JerichoEnvAdvent JerichoEnvAdventureland JerichoEnvAfflicted \
  ALFWorldLookAtObjInLightSeen TWCookingLevel10 ALFWorldLookAtObjInLightUnseen ScienceWorldBoil ScienceWorldMelt \
  TWXSimonSays10 TWXCookingWorld TWXSimonSays100 TWXSimonSaysWithMemory10 TWXSimonSays50 \
  ALFWorldPickAndPlaceSimpleSeen ALFWorldPickCleanThenPlaceInRecepSeen ALFWorldPickHeatThenPlaceInRecepSeen \
  ALFWorldPickCoolThenPlaceInRecepSeen ALFWorldPickTwoObjAndPlaceSeen \
  --episodes-per-game 10 --runs-per-env 3 \
  --output-dir data/trajectories/diverse \
  --nb-steps 100

python collect_data.py --agent agents/collectors.py --agent-name noisy-walkthrough \
  --envs JerichoEnv905 JerichoEnvAcorncourt JerichoEnvAdvent JerichoEnvAdventureland JerichoEnvAfflicted \
  ALFWorldLookAtObjInLightSeen TWCookingLevel10 ALFWorldLookAtObjInLightUnseen ScienceWorldBoil ScienceWorldMelt \
  TWXSimonSays10 TWXCookingWorld TWXSimonSays100 TWXSimonSaysWithMemory10 TWXSimonSays50 \
  ALFWorldPickAndPlaceSimpleSeen ALFWorldPickCleanThenPlaceInRecepSeen ALFWorldPickHeatThenPlaceInRecepSeen \
  ALFWorldPickCoolThenPlaceInRecepSeen ALFWorldPickTwoObjAndPlaceSeen \
  --episodes-per-game 10 --runs-per-env 3 \
  --output-dir data/trajectories/noisy_walkthrough \
  --noise-rate 0.15 --nb-steps 100

python clean_trajectories.py --min-length 15 --min-reward -5 --max-repetition-rate 0.4

python train_vqvae.py --epochs 30 --num-codes 64 --window-size 5 --balanced --commitment-schedule

cd ..
```

**Skip** if you already have `latent-action/checkpoints/vqvae_checkpoint.pt`. If you trained with other window sizes, e.g. `--window-size 10`, add `--window-size 10` when running llm-vqvae (the script uses default 5).

### 3. Select Diagnostic Tasks

**Script:** `scripts/select_diagnostic_tasks.py`

**What it does:** Picks one task per env from the evaluation subset so that diagnostic runs cover all 4 skills evenly. Produces `data/diagnostic_tasks.json` used by the benchmark for quick skill profiling.

```bash
python scripts/select_diagnostic_tasks.py -o data/diagnostic_tasks.json
```

TWCooking-only: `--from-config config/evaluation_config.yaml`

---

### 4. Run Full Evaluation Pipeline

**Script:** `scripts/run_full_evaluation.sh`

**What it does:** Orchestrates steps 1–7 in one run: categorize, select diagnostic tasks, run diagnostic benchmarks (graph, llm-vqvae, memory-agent), full benchmark on 20 envs, transfer analysis, hybrid agents (if API key set), and plots.

**Standard** (nb-steps 100 full / 50 diagnostic):

```bash
./scripts/run_full_evaluation.sh --api-key "your-tritonai-key"
```

**Higher quality** (nb-steps 150 full / 75 diagnostic):

```bash
./scripts/run_full_evaluation.sh --api-key "your-tritonai-key" --nb-steps 150 --nb-steps-diagn 75
```

**Quick test** (5 steps):

```bash
./scripts/run_full_evaluation.sh --api-key "your-tritonai-key" --nb-steps 5 --nb-steps-diagn 5
```

Pipeline steps (all agents use `--admissible-commands --seed 20241001`):
1. Categorize tasks
2. Select diagnostic tasks
3. Diagnostic tests (graph, llm-vqvae, memory-agent)
4. Full benchmark on 20 envs
5. Transfer analysis
6. Hybrid agents (graph-vqvae 0.6/0.4, memory-react 0.5/0.5, full-hybrid 0.3/0.3/0.2/0.2)
7. Plots

### 5. Optional: Hybrid Weight Sensitivity

**Script:** `scripts/run_hybrid_evaluation.sh`

**What it does:** Sweeps over hybrid weight combinations (e.g. graph 0.4–0.8, vqvae 0.2–0.6) to find optimal blends. Useful for tuning before reporting final numbers.

```bash
./scripts/run_hybrid_evaluation.sh
```

---

### 6. Verify Outputs

**What to check:** `logs/*_diagnostic.json` contain per-agent, per-skill scores. `plots/` contains PNG charts (diagnostic comparison, skill profiles, hybrid comparison).

```bash
ls logs/*_diagnostic.json
ls plots/
```

---

## Individual Steps (Manual)

Use explicit `--api-key "your-key"` in all commands. Replace with your TritonAI key.

### 1. Categorize tasks

Assigns skill/difficulty to envs; writes evaluation subset. See step 1 above.

```bash
python scripts/categorize_tasks.py -o data/task_categories.json
```
Outputs: `task_categories.json`, `evaluation_subset.json` (20 tasks, 5 per skill).

### 2. Select diagnostic tasks

Picks one task per env for diagnostic runs. See step 3 above.

```bash
python scripts/select_diagnostic_tasks.py -o data/diagnostic_tasks.json
```
TWCooking-only: `--from-config config/evaluation_config.yaml`

### 3. Diagnostic benchmark (standard: nb-steps 50)

Runs graph, llm-vqvae, and memory-agent on diagnostic tasks only. Writes `logs/*_diagnostic.json` with per-skill scores. Shorter than full benchmark (50 vs 100 steps).
```bash
python benchmark.py --agent agents/graph_agent.py graph \
  --diagnostic-tests data/diagnostic_tasks.json \
  --admissible-commands --nb-steps 50 --seed 20241001 --conversation \
  --key "your-tritonai-key" --llm api-gpt-oss-120b \
  --output-metrics logs/graph_diagnostic.json

python benchmark.py --agent agents/llm_vqvae_agent.py llm-vqvae \
  --diagnostic-tests data/diagnostic_tasks.json \
  --admissible-commands --nb-steps 50 --seed 20241001 \
  --api-key "your-tritonai-key" --vqvae-checkpoint latent-action/checkpoints/vqvae_checkpoint.pt \
  --vqvae-top-k 5 --output-metrics logs/llm-vqvae_diagnostic.json

python benchmark.py --agent agents/memory_agent.py memory-agent \
  --diagnostic-tests data/diagnostic_tasks.json \
  --admissible-commands --nb-steps 50 --seed 20241001 \
  --llm-api-key "your-tritonai-key" --output-metrics logs/memory-agent_diagnostic.json
```

### 4. Full benchmark (standard: nb-steps 100)

Runs the same three agents on all 20 envs in the evaluation subset. Longer runs (100 steps) for final scores. Outputs go to stdout and benchmark logs.

```bash
ENVS=$(python -c "import json; print(' '.join(json.load(open('data/evaluation_subset.json'))['envs']))")

python benchmark.py --agent agents/graph_agent.py graph --envs $ENVS \
  --admissible-commands --nb-steps 100 --seed 20241001 \
  --conversation --key "your-tritonai-key" --llm api-gpt-oss-120b

python benchmark.py --agent agents/llm_vqvae_agent.py llm-vqvae --envs $ENVS \
  --admissible-commands --nb-steps 100 --seed 20241001 \
  --api-key "your-tritonai-key" --vqvae-checkpoint latent-action/checkpoints/vqvae_checkpoint.pt --vqvae-top-k 5

python benchmark.py --agent agents/memory_agent.py memory-agent --envs $ENVS \
  --admissible-commands --nb-steps 100 --seed 20241001 \
  --llm-api-key "your-tritonai-key"
```

### 5. Hybrid agents (standard weights)

Runs graph-vqvae (graph + VQ-VAE), memory-react (memory + ReAct), and full-hybrid (all four components). Requires VQ-VAE checkpoint and API key.

```bash
VQVAE="latent-action/checkpoints/vqvae_checkpoint.pt"
COMMON="--admissible-commands --seed 20241001"

python benchmark.py --agent agents/hybrid_agents.py graph-vqvae --envs $ENVS \
  --api-key "your-tritonai-key" --vqvae-checkpoint $VQVAE \
  --graph-weight 0.6 --vqvae-weight 0.4 $COMMON --nb-steps 100

python benchmark.py --agent agents/hybrid_agents.py memory-react --envs $ENVS \
  --api-key "your-tritonai-key" --memory-weight 0.5 --react-weight 0.5 \
  --conversation $COMMON --nb-steps 100

python benchmark.py --agent agents/hybrid_agents.py full-hybrid --envs $ENVS \
  --api-key "your-tritonai-key" --vqvae-checkpoint $VQVAE \
  --graph-weight 0.3 --vqvae-weight 0.3 --memory-weight 0.2 --react-weight 0.2 \
  --conversation $COMMON --nb-steps 100
```

### 6. Skill transfer analysis

**Script:** `scripts/analyze_skill_transfer.py`

Predicts full-task scores from diagnostic skill profiles using a linear model. Writes `data/*_transfer.json` (e.g. `graph_transfer.json`). Useful to extrapolate from quick diagnostic runs.

```bash
python scripts/analyze_skill_transfer.py -d logs/graph_diagnostic.json \
  -f logs -c data/task_categories.json -o data/graph_transfer.json
```

### 7. Plots

**Script:** `scripts/plot_results.py`

Generates PNG charts from diagnostic logs: `diagnostic_comparison` (agent vs agent), `skill_profiles` (per-skill breakdown), `hybrid_comparison` (hybrid vs base agents).

```bash
python scripts/plot_results.py diagnostic-comparison -m logs/*_diagnostic.json -o plots/
python scripts/plot_results.py skill-profiles -m logs/*_diagnostic.json -o plots/
python scripts/plot_results.py hybrid-comparison -m logs/*_diagnostic.json -o plots/
```

---

## Parameter Reference

| Preset | Full benchmark nb-steps | Diagnostic nb-steps | Use case |
|--------|--------------------------|---------------------|----------|
| Quick test | 5 | 5 | Smoke test |
| Standard | 100 | 50 | Normal evaluation |
| Higher quality | 150 | 75 | More thorough |

**Hybrid weights (default):**
- graph-vqvae: 0.6 / 0.4
- memory-react: 0.5 / 0.5
- full-hybrid: graph 0.3, vqvae 0.3, memory 0.2, react 0.2
