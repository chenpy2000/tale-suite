# Evaluation Pipeline

4 skills: spatial, deductive, inductive, grounded. Uses `data/evaluation_subset.json` (20 envs, 5 per skill).

## Quick Start (Full Pipeline)

```bash
export API_KEY="your-key"   # for LLM agents (TritonAI)
./scripts/run_full_evaluation.sh
```

TWCooking-only: `DIAGNOSTIC_CONFIG=config/evaluation_config.yaml ./scripts/run_full_evaluation.sh`

---

## Complete Start-to-Finish Guide

All commands use the **evaluation subset** (20 envs: 5 spatial, 5 deductive, 5 inductive, 5 grounded). Agents run with **equalized args**: `--admissible-commands`, `--seed 20241001`, `--nb-steps 100` (full) / `50` (diagnostic).

### 0. Prerequisites

```bash
cd /root/tale-suite   # or your repo path

pip install -r requirements.txt

export API_KEY="your-tritonai-key"
export TRITON_API_KEY="$API_KEY"
```

**Note:** `collect_data.py` (steps 2a–2b) uses rule-based collectors (diverse, noisy-walkthrough) that do **not** make LLM calls—no API key is needed. The evaluation pipeline (steps 4+) uses LLM agents (graph, llm-vqvae, memory-agent) and **requires** a TritonAI API key.

### 1. Create Evaluation Subset (if missing)

```bash
python scripts/categorize_tasks.py -o data/task_categories.json
# Outputs: data/task_categories.json, data/evaluation_subset.json (20 envs, 5 per skill)
```

### 2. VQ-VAE Training (required for llm-vqvae and graph-vqvae hybrid)

Collect trajectories on the **exact 20 envs** from the evaluation subset, then train. Run from repo root:

```bash
cd latent-action

# 2a: Collect trajectories on evaluation subset (use tmux/screen for long runs)
python collect_data.py --agent agents/collectors.py --agent-name diverse-collector \
  --envs JerichoEnv905 JerichoEnvAcorncourt JerichoEnvAdvent JerichoEnvAdventureland JerichoEnvAfflicted \
  ALFWorldLookAtObjInLightSeen TWCookingLevel10 ALFWorldLookAtObjInLightUnseen ScienceWorldBoil ScienceWorldMelt \
  TWXSimonSays10 TWXCookingWorld TWXSimonSays100 TWXSimonSaysWithMemory10 TWXSimonSays50 \
  ALFWorldPickAndPlaceSimpleSeen ALFWorldPickCleanThenPlaceInRecepSeen ALFWorldPickHeatThenPlaceInRecepSeen \
  ALFWorldPickCoolThenPlaceInRecepSeen ALFWorldPickTwoObjAndPlaceSeen \
  --episodes-per-game 10 --runs-per-env 3 \
  --output-dir data/trajectories/diverse \
  --nb-steps 200

python collect_data.py --agent agents/collectors.py --agent-name noisy-walkthrough \
  --envs JerichoEnv905 JerichoEnvAcorncourt JerichoEnvAdvent JerichoEnvAdventureland JerichoEnvAfflicted \
  ALFWorldLookAtObjInLightSeen TWCookingLevel10 ALFWorldLookAtObjInLightUnseen ScienceWorldBoil ScienceWorldMelt \
  TWXSimonSays10 TWXCookingWorld TWXSimonSays100 TWXSimonSaysWithMemory10 TWXSimonSays50 \
  ALFWorldPickAndPlaceSimpleSeen ALFWorldPickCleanThenPlaceInRecepSeen ALFWorldPickHeatThenPlaceInRecepSeen \
  ALFWorldPickCoolThenPlaceInRecepSeen ALFWorldPickTwoObjAndPlaceSeen \
  --episodes-per-game 10 --runs-per-env 3 \
  --output-dir data/trajectories/noisy_walkthrough \
  --noise-rate 0.15 --nb-steps 200

# 2b: Clean trajectories
python clean_trajectories.py

# 2c: Train VQ-VAE (window-size 5 matches inference default)
python train_vqvae.py --epochs 10 --num-codes 64 --window-size 5 --balanced

cd ..
# Checkpoint: latent-action/checkpoints/vqvae_checkpoint.pt
```

**Skip** if you already have `latent-action/checkpoints/vqvae_checkpoint.pt`.

### 3. Select Diagnostic Tasks

```bash
python scripts/select_diagnostic_tasks.py -o data/diagnostic_tasks.json
# Or TWCooking-only: --from-config config/evaluation_config.yaml
```

### 4. Run Full Evaluation Pipeline

```bash
./scripts/run_full_evaluation.sh
```

Quick test (5 steps for diagnostics and full benchmark):

```bash
NB_STEPS=5 NB_STEPS_DIAG=5 ./scripts/run_full_evaluation.sh
```

Runs (all agents use `--admissible-commands --seed 20241001`):
1. Categorize tasks
2. Select diagnostic tasks
3. Diagnostic tests (graph, llm-vqvae, memory-agent) — `--nb-steps 50`
4. Full benchmark on 20 envs — `--nb-steps 100`
5. Transfer analysis
6. Hybrid agents (graph-vqvae, memory-react, full-hybrid) if API_KEY set
7. Plots

### 5. Optional: Hybrid Weight Sensitivity

```bash
./scripts/run_hybrid_evaluation.sh
```

### 6. Verify Outputs

```bash
ls logs/*_diagnostic.json
ls plots/
```

---

## Individual Steps (Manual)

### 1. Categorize tasks
```bash
python scripts/categorize_tasks.py -o data/task_categories.json
```
Outputs: `task_categories.json`, `evaluation_subset.json` (20 tasks, 5 per skill)
Flags: `--manual-review`, `--per-skill N`, `--spatial-weight` etc.

### 2. Select diagnostic tasks
```bash
python scripts/select_diagnostic_tasks.py -o data/diagnostic_tasks.json
```
Default: from evaluation_subset. Override: `--config path` or `--from-config config/evaluation_config.yaml`

### 3. Diagnostic benchmark (equalized: admissible-commands, seed, nb-steps)
```bash
python benchmark.py --agent agents/graph_agent.py graph \
  --diagnostic-tests data/diagnostic_tasks.json \
  --admissible-commands --nb-steps 50 --seed 20241001 --conversation \
  --api-key $API_KEY --output-metrics logs/graph_diagnostic.json

python benchmark.py --agent agents/llm_vqvae_agent.py llm-vqvae \
  --diagnostic-tests data/diagnostic_tasks.json \
  --admissible-commands --nb-steps 50 --seed 20241001 \
  --api-key $API_KEY --vqvae-checkpoint latent-action/checkpoints/vqvae_checkpoint.pt \
  --output-metrics logs/llm-vqvae_diagnostic.json

python benchmark.py --agent agents/memory_agent.py memory-agent \
  --diagnostic-tests data/diagnostic_tasks.json \
  --admissible-commands --nb-steps 50 --seed 20241001 \
  --output-metrics logs/memory-agent_diagnostic.json
```

### 4. Full benchmark (evaluation subset, equalized args)
```bash
python benchmark.py --agent agents/graph_agent.py graph \
  --envs JerichoEnv905 JerichoEnvAcorncourt JerichoEnvAdvent JerichoEnvAdventureland JerichoEnvAfflicted \
  ALFWorldLookAtObjInLightSeen TWCookingLevel10 ALFWorldLookAtObjInLightUnseen ScienceWorldBoil ScienceWorldMelt \
  TWXSimonSays10 TWXCookingWorld TWXSimonSays100 TWXSimonSaysWithMemory10 TWXSimonSays50 \
  ALFWorldPickAndPlaceSimpleSeen ALFWorldPickCleanThenPlaceInRecepSeen ALFWorldPickHeatThenPlaceInRecepSeen \
  ALFWorldPickCoolThenPlaceInRecepSeen ALFWorldPickTwoObjAndPlaceSeen \
  --admissible-commands --nb-steps 100 --seed 20241001 \
  --conversation --api-key $API_KEY
```

### 5. Skill transfer analysis
```bash
python scripts/analyze_skill_transfer.py -d logs/graph_diagnostic.json \
  -f logs -c data/task_categories.json -o data/graph_transfer.json
```
`-f` can be a directory (crawls for env_name + norm_score) or JSON file.

### 6. Plots
```bash
python scripts/plot_results.py diagnostic-comparison -m logs/*_diagnostic.json -o plots/
python scripts/plot_results.py skill-profiles -m logs/*_diagnostic.json -o plots/
```
