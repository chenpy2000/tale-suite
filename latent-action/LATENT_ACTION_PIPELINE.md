# Latent-Action Pipeline

This pipeline improves how an AI plays text-based games (like Zork or cooking games) by combining a large language model (LLM) with a small model trained on human-like play. The LLM reasons about the game; the small model helps pick the best action.

**Pipeline:** Collect → Clean → Train → Benchmark

Steps 1–3 don't use any API. Step 4 (benchmark) calls the LLM API—about $35 for the full run.

---

## How the whole pipeline works

### The problem

Text games give you a description of the world and a list of valid commands (e.g. "take cookbook", "go north"). These are called *admissible commands*—only these will work right now. You type a command and get a new description. LLMs can read and write well, but they often pick bad or irrelevant actions. They don't reliably learn from the game state which action will lead to progress.

### The idea

We collect many playthroughs (trajectories) of games—sequences of (what you saw, what you did). We train a small neural network called a VQ-VAE on these trajectories. It learns patterns like "after examining the cookbook, taking it from the table is often good." At runtime, we combine this with an LLM: the LLM suggests an action, and the VQ-VAE scores each valid action. We pick the one the VQ-VAE likes most, with a boost if it matches the LLM.

### What each step does

**1. Collect** — We run automated "collector" agents that play the games. Two types:
- **Noisy walkthrough:** Follows a game's built-in solution (walkthrough) when available, but randomly deviates 20% of the time so we get variety.
- **Diverse collector:** Doesn't need walkthroughs. Cycles through strategies (goal-directed, exploration, object interaction, navigation, random) so we get varied behavior on any game.

Output: `data/trajectories/` with one folder per game, each containing `episode_*.json` files. Each file is one playthrough: a list of (observation, action, reward) steps.

**2. Clean** — We filter out bad episodes: too short, too low reward, or too repetitive (e.g. "look" over and over). These wouldn't help the model learn useful patterns.

Output: `data/trajectories_cleaned/` with the same structure, fewer files.

**3. Train** — We train a VQ-VAE on sliding windows of (observation, action) from the cleaned trajectories. For each window (e.g. the last 10 steps), the model learns to predict the next action. The "VQ" part compresses the continuous representation into discrete codes—like learning a small vocabulary of "situations" that recur across games.

Output: `checkpoints/vqvae_checkpoint.pt` — the trained model plus the action vocabulary.

**4. Benchmark** — We run the LLM+VQ-VAE agent on the games. For each step:
- The game gives an observation and a list of valid commands.
- We build a short history of recent (obs, action) pairs.
- The VQ-VAE scores each valid action (how likely it would predict that action given the history).
- We show the LLM the observation, valid commands, and the top VQ-VAE suggestions as a hint.
- **The LLM decides** which action to take (it may follow or ignore the VQ-VAE suggestions).
- We send the LLM's choice to the game and repeat.

This step uses the LLM API (TritonAI), so it costs money. Steps 1–3 are free.

### What is a VQ-VAE?

A VQ-VAE (Vector Quantized Variational Autoencoder) has three parts:

1. **Encoder** — Takes a sequence of (observation, action) and produces a continuous vector summarizing it.
2. **Quantizer** — Maps that vector to the nearest entry in a fixed "codebook" (e.g. 64 or 128 vectors). This forces the model to use a discrete set of representations.
3. **Decoder** — Takes the quantized vector and predicts the next action (or reconstructs the sequence).

By training it to reconstruct actions, the model learns which action sequences tend to go together. At runtime, we use the decoder's prediction scores to suggest top actions to the LLM as an advisory hint.

### Summary flow

```
Collectors play games → trajectories (raw)
       ↓
Clean (filter bad episodes) → trajectories_cleaned
       ↓
Train VQ-VAE on windows → checkpoint (model + vocab)
       ↓
Benchmark: LLM + VQ-VAE play games, pick actions together
```

---

## Full Pipeline (~$35 API credits)

All 122 environments. Expect hours to days depending on hardware.

### Step 1: Collect trajectories

From the repo root. Collection runs for hours—use `tmux` or `screen` so it survives a dropped connection:

```bash
tmux new -s collect   # or: screen -S collect
cd latent-action

# Noisy walkthrough — uses game walkthroughs when available, adds 20% noise
python collect_data.py --agent agents/collectors.py --agent-name noisy-walkthrough \
  --tasks alfworld jericho scienceworld textworld textworld_express \
  --episodes-per-game 10 --runs-per-env 3 \
  --output-dir data/trajectories/noisy_walkthrough \
  --noise-rate 0.2 \
  --nb-steps 200

# Diverse exploration — multiple strategies, works for all games
python collect_data.py --agent agents/collectors.py --agent-name diverse-collector \
  --tasks alfworld jericho scienceworld textworld textworld_express \
  --episodes-per-game 10 --runs-per-env 3 \
  --output-dir data/trajectories/diverse \
  --nb-steps 200
```

Reattach later with `tmux attach -t collect` (or `screen -r collect`). Without tmux/screen: `nohup python collect_data.py ... > collect.log 2>&1 &` then `tail -f collect.log`.

### Step 2: Clean trajectories

```bash
python clean_trajectories.py
```

Defaults (min-length 3, min-reward -20) work for short nb-steps runs. For longer runs, use stricter filters:

```bash
python clean_trajectories.py --min-length 15 --min-reward -5 --max-repetition-rate 0.4
```

### Step 3: Train VQ-VAE

```bash
python train_vqvae.py --epochs 30 --num-codes 64 --window-size 10 --balanced --commitment-schedule
```

Checkpoint is saved to `latent-action/checkpoints/vqvae_checkpoint.pt`.

To speed up training, run this step on Google Colab with a T4 GPU—see "Training on Colab" below.

**Optional: Analyze checkpoint**

```bash
python analyze_options.py
```

Produces `option_analysis.txt` (per-option action sequences, rewards, codebook utilization) plus t-SNE and histogram plots in the current dir. Use `--output-dir analysis` to put outputs in `analysis/`, or `--max-windows 500` for a quicker run.

### Step 4: Benchmark

From the repo root. You need an API key:

```bash
cd ..
python benchmark.py --agent agents/llm_vqvae_agent.py llm-vqvae \
  --api-key sk-jtESe5Ch5ymQNgwUfR3afQ \
  --api-url https://tritonai-api.ucsd.edu \
  --model api-gpt-oss-120b \
  --vqvae-checkpoint latent-action/checkpoints/vqvae_checkpoint.pt \
  --admissible-commands \
  --nb-steps 10
```

122 envs × 280 steps ≈ 34,160 API calls ≈ $35 at typical TritonAI pricing. Check the [model hub](https://tritonai-api.ucsd.edu/ui/model_hub_table/) for current rates.

To reduce cost: `--nb-steps 100` (~$12) or `--nb-steps 150` (~$18).

---

## Short Pipeline (local testing)

Quick end-to-end test. ~30–60 minutes. Uses only textworld and textworld_express (26 envs), fewer episodes, fewer steps.

### Step 1: Collect

```bash
cd latent-action

python collect_data.py --agent agents/collectors.py --agent-name diverse-collector \
  --tasks textworld \
  --episodes-per-game 20 \
  --runs-per-env 5 \
  --output-dir data/trajectories/diverse \
  --nb-steps 300

python collect_data.py --agent agents/collectors.py --agent-name noisy-walkthrough \
  --tasks textworld \
  --episodes-per-game 30 \
  --runs-per-env 5 \
  --output-dir data/trajectories/noisy_walkthrough \
  --noise-rate 0.15 \
  --nb-steps 300
```

### Step 2: Clean

```bash
python clean_trajectories.py
```

### Step 3: Train

```bash
python train_vqvae.py --epochs 10 --num-codes 64 --window-size 5 --balanced
```

### Step 4: Benchmark

```bash
cd ..
python benchmark.py --agent agents/llm_vqvae_agent.py llm-vqvae \
  --api-key sk-jtESe5Ch5ymQNgwUfR3afQ \
  --api-url https://tritonai-api.ucsd.edu \
  --model api-gpt-oss-120b \
  --vqvae-checkpoint latent-action/checkpoints/vqvae_checkpoint.pt \
  --admissible-commands \
  --envs textworld \
  --nb-steps 10
```

---

## Textworld-only pipeline (after collect + clean)

After collecting and cleaning trajectories for textworld, run:

### Step 3: Train (textworld)

```bash
cd latent-action

# Standard training (CPU, ~30–60 min)
python train_vqvae.py --epochs 30 --num-codes 64 --window-size 10 --balanced --commitment-schedule

# Or faster on GPU (Colab): see "Training on Colab" below
```

Ensure `--trajectories-dir` points to cleaned data (default: `data/trajectories_cleaned`). Use `--window-size` matching your data (10 or 15 recommended for textworld).

### Step 4: Benchmark (textworld)

From the repo root:

```bash
cd ..
python benchmark.py --agent agents/llm_vqvae_agent.py llm-vqvae \
  --envs textworld \
  --vqvae-checkpoint latent-action/checkpoints/vqvae_checkpoint.pt \
  --api-key sk-jtESe5Ch5ymQNgwUfR3afQ  \
  --api-url https://tritonai-api.ucsd.edu \
  --model api-gpt-oss-120b \
  --admissible-commands \
  --nb-steps 200
```

Use `--window-size 10` (or 15) if your checkpoint was trained with that value. Reduce `--nb-steps` (e.g. 50) for cheaper runs.

---

## Training on Colab (GPU)

Colab doesn't run Docker. To speed up Step 3 (train) with a free GPU:

1. Run Steps 1–2 in Docker or locally.
2. Zip the entire `latent-action` folder (with `data/trajectories_cleaned`, `vqvae.py`, `train_vqvae.py`).
3. Upload the zip to Colab.
4. Runtime → Change runtime type → T4 GPU.
5. Run:

```python
!unzip -q latent-action.zip
%cd latent-action
!pip install -q torch sentence-transformers
!python train_vqvae.py --epochs 30 --num-codes 64 --window-size 10 --balanced --commitment-schedule
```

6. Download `checkpoints/vqvae_checkpoint.pt` and copy it into your local `latent-action/checkpoints/`.
7. Run Step 4 (benchmark) in Docker or locally as usual.

---

## Environment reference

| Task              | Envs | Description                          |
|-------------------|------|--------------------------------------|
| textworld         | 10   | TWCookingLevel1–10                   |
| textworld_express | 16   | CookingWorld, CoinCollector, etc.    |
| alfworld          | 12   | Pick/place, look, clean, heat, cool   |
| jericho           | 54   | Classic text adventures              |
| scienceworld      | 30   | Science tasks                        |

**Total:** 122 environments.

List all envs:

```bash
cd latent-action
python collect_data.py --tasks alfworld jericho scienceworld textworld textworld_express --dry-run
```

---

## Agent options

- `--api-key` — Required. Your TritonAI API key.
- `--admissible-commands` — Required for llm-vqvae. The agent uses the game's valid command list. The benchmark auto-enables this when you use llm-vqvae.
- `--api-url` — Default: https://tritonai-api.ucsd.edu
- `--model` — Default: api-llama-4-scout. See https://tritonai-api.ucsd.edu/ui/model_hub_table/
- `--vqvae-top-k` — Number of top VQ-VAE suggestions to show the LLM as a hint (default: 5). The LLM is in control; VQ-VAE advises.

---

## File layout

```
tale-suite/
├── agents/
│   ├── collectors.py
│   └── llm_vqvae_agent.py
├── latent-action/
│   ├── LATENT_ACTION_PIPELINE.md
│   ├── vqvae.py
│   ├── collect_data.py
│   ├── clean_trajectories.py
│   ├── train_vqvae.py
│   ├── data/
│   │   ├── trajectories/
│   │   ├── trajectories_cleaned/
│   │   └── trajectory_runs/
│   └── checkpoints/
```

---

## Dependencies

- torch
- sentence-transformers
- requests
