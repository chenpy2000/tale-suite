#!/bin/bash
# Hybrid evaluation on evaluation subset. Equalized: --admissible-commands --seed 20241001.

set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f .env ] && export $(cat .env | xargs)

API_KEY="${API_KEY:-}"
NB_STEPS=50
VQVAE_CHECKPOINT="latent-action/checkpoints/vqvae_checkpoint.pt"
COMMON="--admissible-commands --seed 20241001 --diagnostic-tests data/diagnostic_tasks.json --nb-steps $NB_STEPS"

mkdir -p logs/hybrids
[[ -z "$API_KEY" ]] && { echo "Set API_KEY"; exit 1; }
[[ ! -f data/diagnostic_tasks.json ]] && { echo "Run scripts/run_full_evaluation.sh first (or select_diagnostic_tasks.py)"; exit 1; }

echo "=== Evaluating Hybrid Agents (evaluation subset) ==="

# Graph + VQ-VAE (vary weights)
python benchmark.py --agent agents/hybrid_agents.py graph-vqvae \
  --api-key "$API_KEY" --vqvae-checkpoint "$VQVAE_CHECKPOINT" \
  --graph-weight 0.6 --vqvae-weight 0.4 $COMMON \
  --output-metrics logs/hybrids/graph-vqvae-60-40.json

python benchmark.py --agent agents/hybrid_agents.py graph-vqvae \
  --api-key "$API_KEY" --vqvae-checkpoint "$VQVAE_CHECKPOINT" \
  --graph-weight 0.8 --vqvae-weight 0.2 $COMMON \
  --output-metrics logs/hybrids/graph-vqvae-80-20.json

python benchmark.py --agent agents/hybrid_agents.py graph-vqvae \
  --api-key "$API_KEY" --vqvae-checkpoint "$VQVAE_CHECKPOINT" \
  --graph-weight 0.4 --vqvae-weight 0.6 $COMMON \
  --output-metrics logs/hybrids/graph-vqvae-40-60.json

# Memory + ReAct
python benchmark.py --agent agents/hybrid_agents.py memory-react \
  --api-key "$API_KEY" --memory-weight 0.5 --react-weight 0.5 \
  --conversation $COMMON \
  --output-metrics logs/hybrids/memory-react-50-50.json

# Full Hybrid
python benchmark.py --agent agents/hybrid_agents.py full-hybrid \
  --api-key "$API_KEY" --vqvae-checkpoint "$VQVAE_CHECKPOINT" \
  --graph-weight 0.25 --vqvae-weight 0.25 --memory-weight 0.25 --react-weight 0.25 \
  --conversation $COMMON \
  --output-metrics logs/hybrids/full-hybrid-balanced.json

python benchmark.py --agent agents/hybrid_agents.py full-hybrid \
  --api-key "$API_KEY" --vqvae-checkpoint "$VQVAE_CHECKPOINT" \
  --graph-weight 0.5 --vqvae-weight 0.2 --memory-weight 0.2 --react-weight 0.1 \
  --conversation $COMMON \
  --output-metrics logs/hybrids/full-hybrid-spatial.json

python benchmark.py --agent agents/hybrid_agents.py full-hybrid \
  --api-key "$API_KEY" --vqvae-checkpoint "$VQVAE_CHECKPOINT" \
  --graph-weight 0.2 --vqvae-weight 0.5 --memory-weight 0.2 --react-weight 0.1 \
  --conversation $COMMON \
  --output-metrics logs/hybrids/full-hybrid-inductive.json

echo "Hybrid evaluation complete. Results in logs/hybrids/"

# Generate comparison plots
echo "Generating hybrid comparison plots..."
mkdir -p plots
python scripts/plot_results.py hybrid-weights \
  --metrics logs/hybrids/graph-vqvae-*.json \
  --output plots/
python scripts/plot_results.py hybrid-comparison \
  --metrics logs/hybrids/*.json \
  --output plots/
