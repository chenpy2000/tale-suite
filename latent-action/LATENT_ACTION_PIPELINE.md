# Latent-Action Option Discovery Pipeline

VQ-VAE pipeline for discovering discrete options from text-adventure trajectories. Encodes sliding windows of (observation, action) pairs into discrete codes; the decoder reconstructs action sequences. Learned options can be used for hierarchical RL, as a policy prior, or directly as an agent.

**Goal:** Learn a small set of reusable "options" (action subsequences) from trajectories, with good codebook utilization (target >90% of codes used).

---

## How It Works

1. **Collect** — Run agents (diverse-collector, noisy-walkthrough, or trajectory-collector) in games. Each step records (obs, action, reward). Episodes are saved as JSON.
2. **Clean** — Filter out short, low-reward, or repetitive episodes.
3. **Train** — Build sliding windows over (obs, action) sequences. Encode with a transformer, quantize to discrete codes, decode to reconstruct actions. The VQ-VAE learns to cluster similar behavior into options.
4. **Analyze** — Map trajectories to options, report utilization and per-option statistics.
5. **Run** — Use the trained model as an agent: encode current (obs, action) history, predict next action from the decoder.

---

## Problems Addressed

- **Codebook collapse:** VQ-VAEs often use only 2–10% of codes. Fixes: higher `commitment_beta` (default 1.0), `--commitment-schedule` warmup (0.25→2.0 over 10 epochs), `CodebookReset` for dead/rare codes every 500 steps, smaller `num_codes` (64), `--balanced` dataset (full windows only, no padding).
- **Low-quality data:** Filter by `min_length`, `min_reward`, `max_repetition_rate` in `clean_trajectories.py`.
- **Repetitive trajectories:** Use diverse-collector (rotates strategies) and noisy-walkthrough (walkthrough + controlled noise) for varied data.

---

## Pipeline Steps

### Step 1: Collect

**Script:** `latent-action/collect_data.py`

Runs `benchmark.py` from the repo root for each environment. The agent plays and saves trajectories; `collect_data` converts them to normalized episode JSONs under `latent-action/data/trajectories/`.

**Collectors** (in `agents/collectors.py`):
- **diverse-collector** — Rotates between goal-oriented, exploratory, object-focused, navigation, and random strategies. Skips episodes that are too short, too low-reward, or too repetitive. Good for varied exploration.
- **noisy-walkthrough** — Follows `info["extra.walkthrough"]` when available, with probability `--noise-rate` of deviating. Best for TextWorld/TextWorld-Express games that provide walkthroughs.

**Recommended: mixed collection** (noisy for high-quality demos, diverse for exploration):

```bash
cd latent-action
rm -rf data/trajectories/*

# Noisy walkthrough (games with walkthroughs)
python collect_data.py --agent agents/collectors.py --agent-name noisy-walkthrough \
  --envs TWCookingLevel1 TWCookingLevel2 TWCookingLevel3 TWCookingLevel4 TWCookingLevel5 \
         TWCookingLevel6 TWCookingLevel7 TWCookingLevel8 TWCookingLevel9 TWCookingLevel10 \
  --episodes-per-game 10 --output-dir data/trajectories/noisy_walkthrough --noise-rate 0.2

# Diverse exploration
python collect_data.py --agent agents/collectors.py --agent-name diverse-collector \
  --envs TWCookingLevel1 TWCookingLevel2 TWCookingLevel3 TWCookingLevel4 TWCookingLevel5 \
         TWCookingLevel6 TWCookingLevel7 TWCookingLevel8 TWCookingLevel9 TWCookingLevel10 \
  --episodes-per-game 10 --output-dir data/trajectories/diverse
```

**collect_data parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--tasks` | textworld, textworld_express | Task names; expands to envs |
| `--envs` | (from tasks) | Override with explicit env names |
| `--episodes-per-game` | 1 | Episodes to save per env |
| `--nb-steps` | 200 | Max steps per game run |
| `--seed` | 20241001 | Random seed |
| `--output-dir` | data/trajectories | Where to write episode JSONs |
| `--work-dir` | data/trajectory_runs | Temp benchmark output |
| `--agent` | agents/collectors.py | Agent module path |
| `--agent-name` | diverse-collector | Which registered agent (diverse-collector, noisy-walkthrough, trajectory-collector) |
| `--min-episode-length` | (none) | Pass to agent: skip shorter episodes |
| `--min-reward` | (none) | Pass to agent: skip lower-reward episodes |
| `--max-repetition-rate` | (none) | Pass to agent: skip if most common action exceeds this fraction |
| `--noise-rate` | (none) | Pass to noisy-walkthrough: deviation probability |
| `--no-admissible-commands` | false | Disable admissible commands |
| `--dry-run` | false | List envs only, no collection |

---

### Step 2: Clean

**Script:** `latent-action/clean_trajectories.py`

Filters episodes by min length, min reward, and max repetition rate. Writes to `data/trajectories_cleaned/`.

```bash
cd latent-action
python clean_trajectories.py
```

**Stricter filters:**

```bash
python clean_trajectories.py --min-length 15 --min-reward -5 --max-repetition-rate 0.4
```

**clean_trajectories parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input-dir` | data/trajectories | Source episodes |
| `--output-dir` | data/trajectories_cleaned | Filtered output |
| `--min-length` | 15 | Minimum steps per episode |
| `--min-reward` | -5 | Minimum total reward |
| `--max-repetition-rate` | 0.4 | Max fraction of most common action |

---

### Step 3: Train

**Script:** `latent-action/train_vqvae.py`

Uses `data/trajectories_cleaned/` by default; falls back to `data/trajectories/` if cleaned does not exist. Saves checkpoints to `checkpoints/`.

**Recommended training:**

```bash
cd latent-action
python train_vqvae.py --epochs 30 --num-codes 64 --window-size 10 --balanced --commitment-schedule
```

**train_vqvae parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--trajectories-dir` | data/trajectories_cleaned | Trajectory root |
| `--window-size` | 5 | Sliding window size (5, 10, or 15) |
| `--batch-size` | 32 | Training batch size |
| `--epochs` | 10 | Training epochs |
| `--lr` | 1e-4 | Learning rate |
| `--num-codes` | 64 | Codebook size (32, 64, 128, 256) |
| `--commitment-beta` | 1.0 | Commitment loss weight |
| `--text-model` | all-MiniLM-L6-v2 | Sentence-transformers model |
| `--latent-dim` | 256 | Latent/embedding dimension |
| `--checkpoint-path` | checkpoints/vqvae_checkpoint.pt | Output filename |
| `--seed` | 20241001 | Random seed |
| `--augment` | false | Randomly drop one step per window |
| `--balanced` | false | Full windows only, no padding (recommended) |
| `--commitment-schedule` | false | Warmup commitment from 0.25 to 2.0 over 10 epochs (recommended) |

**Dataset modes:**
- **Default (unbalanced):** Sliding windows; short episodes padded with `<PAD>`. Requires at least 3 non-pad actions per window.
- **`--balanced`:** Only full windows (no padding). Reduces dominant padded patterns.
- **`--augment`:** Randomly drops one step per window (with probability 0.3) for data augmentation.

**CodebookReset:** Runs every 500 steps: resets dead codes and codes used <10 times by copying from active codes with small noise.

---

### Step 4: Analyze

**Script:** `latent-action/analyze_options.py`

Encodes all trajectory windows with the trained model, maps them to options, and writes a report and histogram to `analysis/`.

```bash
cd latent-action
python analyze_options.py
```

**analyze_options parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--checkpoint-path` | checkpoints/vqvae_checkpoint.pt | Trained model |
| `--trajectories-dir` | data/trajectories_cleaned | Trajectory root |
| `--window-size` | (from checkpoint) | Override window size |
| `--batch-size` | 32 | Encoding batch size |
| `--max-windows` | (none) | Cap windows for quick checks |
| `--analysis-dir` | analysis | Output directory |

**Outputs:**
- `option_analysis.txt` — Codebook stats, per-option counts, avg reward, top action sequences
- `option_usage_histogram.png` — Bar chart of code usage

---

### Step 5: Run VQ-VAE Agent

After training, run the VQ-VAE as an agent via `benchmark.py` from the repo root:

```bash
python benchmark.py --agent agents/vqvae_agent.py vqvae \
  --checkpoint-path latent-action/checkpoints/vqvae_checkpoint.pt \
  --envs TWCookingLevel1 TWCookingLevel2 --nb-steps 200
```

**vqvae agent parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--checkpoint-path` | latent-action/checkpoints/vqvae_checkpoint.pt | Trained model |
| `--window-size` | (from checkpoint) | Override window size |
| `--seed` | 20241001 | Random seed for fallback |

**Behavior:** Maintains (obs, action) history. Encodes current window, quantizes, decodes to predict next action. If predicted action is not in admissible commands, picks randomly from admissible.

---

## Quick Reference: Full Pipeline

```bash
cd latent-action

# 1. Collect
python collect_data.py --agent agents/collectors.py --agent-name noisy-walkthrough \
  --envs TWCookingLevel1 TWCookingLevel2 TWCookingLevel3 TWCookingLevel4 TWCookingLevel5 \
         TWCookingLevel6 TWCookingLevel7 TWCookingLevel8 TWCookingLevel9 TWCookingLevel10 \
  --episodes-per-game 10 --output-dir data/trajectories/noisy_walkthrough --noise-rate 0.2

python collect_data.py --agent agents/collectors.py --agent-name diverse-collector \
  --envs TWCookingLevel1 TWCookingLevel2 TWCookingLevel3 TWCookingLevel4 TWCookingLevel5 \
         TWCookingLevel6 TWCookingLevel7 TWCookingLevel8 TWCookingLevel9 TWCookingLevel10 \
  --episodes-per-game 10 --output-dir data/trajectories/diverse

# 2. Clean
python clean_trajectories.py

# 3. Train
python train_vqvae.py --epochs 30 --num-codes 64 --window-size 10 --balanced --commitment-schedule

# 4. Analyze
python analyze_options.py

# 5. Run agent (from repo root)
python benchmark.py --agent agents/vqvae_agent.py vqvae \
  --checkpoint-path latent-action/checkpoints/vqvae_checkpoint.pt \
  --envs TWCookingLevel1 TWCookingLevel2 --nb-steps 200
```

---

## File Layout

```
tale-suite/
├── agents/
│   ├── collectors.py        # diverse-collector + noisy-walkthrough
│   ├── trajectory_collector.py
│   └── vqvae_agent.py
├── latent-action/
│   ├── LATENT_ACTION_PIPELINE.md
│   ├── vqvae.py             # Model (encoder, quantizer, decoder, ActionVocab)
│   ├── collect_data.py
│   ├── clean_trajectories.py
│   ├── train_vqvae.py       # Train + dataset classes
│   ├── analyze_options.py
│   ├── data/
│   │   ├── trajectories/
│   │   ├── trajectories_cleaned/
│   │   └── trajectory_runs/
│   ├── checkpoints/
│   └── analysis/
```

---

## Dependencies

- torch
- sentence-transformers
- matplotlib (for analyze_options plots)
