# Latent-Action Pipeline

Train a VQ-VAE on game trajectories, then pair it with an LLM. The VQ-VAE learns action patterns from demos; the LLM reasons about the game. Together they pick better actions than either alone.

**Pipeline:** Collect → Clean → Train → Benchmark

Steps 1–3 don't use the API. Step 4 (benchmark) does. Full run is ~$35 at 280 steps per env.

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
  --episodes-per-game 10 \
  --output-dir data/trajectories/noisy_walkthrough \
  --noise-rate 0.2 \
  --nb-steps 200

# Diverse exploration — multiple strategies, works for all games
python collect_data.py --agent agents/collectors.py --agent-name diverse-collector \
  --tasks alfworld jericho scienceworld textworld textworld_express \
  --episodes-per-game 10 \
  --output-dir data/trajectories/diverse \
  --nb-steps 200
```

Reattach later with `tmux attach -t collect` (or `screen -r collect`). Without tmux/screen: `nohup python collect_data.py ... > collect.log 2>&1 &` then `tail -f collect.log`.

### Step 2: Clean trajectories

```bash
python clean_trajectories.py
```

Optional stricter filters:

```bash
python clean_trajectories.py --min-length 15 --min-reward -5 --max-repetition-rate 0.4
```

### Step 3: Train VQ-VAE

```bash
python train_vqvae.py --epochs 30 --num-codes 64 --window-size 10 --balanced --commitment-schedule
```

Checkpoint is saved to `latent-action/checkpoints/vqvae_checkpoint.pt`.

To speed up training, run this step on Google Colab with a T4 GPU—see "Training on Colab" below.

### Step 4: Benchmark

From the repo root. You need an API key:

```bash
cd ..
python benchmark.py --agent agents/llm_vqvae_agent.py llm-vqvae \
  --api-key YOUR_API_KEY \
  --api-url https://tritonai-api.ucsd.edu \
  --model api-llama-4-scout \
  --vqvae-checkpoint latent-action/checkpoints/vqvae_checkpoint.pt \
  --admissible-commands \
  --nb-steps 280
```

122 envs × 280 steps ≈ 34,160 API calls ≈ $35 at typical TritonAI pricing. Check the [model hub](https://tritonai-api.ucsd.edu/ui/model_hub_table/) for current rates.

To reduce cost: `--nb-steps 100` (~$12) or `--nb-steps 150` (~$18).

---

## Short Pipeline (local testing)

Quick end-to-end test. ~30–60 minutes. Uses only textworld and textworld_express (26 envs), fewer episodes, fewer steps.

### Step 1: Collect

```bash
cd latent-action

python collect_data.py --agent agents/collectors.py --agent-name noisy-walkthrough \
  --tasks textworld textworld_express \
  --episodes-per-game 3 \
  --output-dir data/trajectories/noisy_walkthrough \
  --noise-rate 0.2 \
  --nb-steps 100

python collect_data.py --agent agents/collectors.py --agent-name diverse-collector \
  --tasks textworld textworld_express \
  --episodes-per-game 3 \
  --output-dir data/trajectories/diverse \
  --nb-steps 100
```

### Step 2: Clean

```bash
python clean_trajectories.py --min-length 5 --min-reward -20 --max-repetition-rate 0.5
```

### Step 3: Train

```bash
python train_vqvae.py --epochs 10 --num-codes 64 --window-size 5 --balanced
```

### Step 4: Benchmark

```bash
cd ..
python benchmark.py --agent agents/llm_vqvae_agent.py llm-vqvae \
  --api-key YOUR_API_KEY \
  --api-url https://tritonai-api.ucsd.edu \
  --model api-llama-4-scout \
  --vqvae-checkpoint latent-action/checkpoints/vqvae_checkpoint.pt \
  --admissible-commands \
  --envs TWCookingLevel1 TWCookingLevel2 \
  --nb-steps 50
```

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
- `--admissible-commands` — Required for llm-vqvae. The agent scores actions from the game's valid command list. The benchmark auto-enables this when you use llm-vqvae.
- `--api-url` — Default: https://tritonai-api.ucsd.edu
- `--model` — Default: api-llama-4-scout. See https://tritonai-api.ucsd.edu/ui/model_hub_table/
- `--llm-weight` — Weight for LLM suggestion when ranking actions (default: 0.3)

---

## How it works

- **VQ-VAE:** Encodes (obs, action) windows into discrete codes, decodes to reconstruct actions. Learns which action sequences belong together.
- **LLM+VQ-VAE agent:** LLM reasons about the game state and suggests an action. VQ-VAE scores each admissible action from (obs, action) history. Picks the action with highest VQ-VAE score, boosted if it matches the LLM suggestion.

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
