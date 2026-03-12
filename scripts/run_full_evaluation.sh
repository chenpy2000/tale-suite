#!/bin/bash
# Full evaluation: categorize -> diagnostic tasks -> diagnostic tests -> full benchmark -> transfer analysis -> plots.

set -e

[ -f .env ] && export $(cat .env | xargs)
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

AGENTS=(graph llm-vqvae memory-agent)
API_KEY="${API_KEY:-}"
DIAGNOSTIC_CONFIG="${DIAGNOSTIC_CONFIG:-}"

get_agent_file() {
    case "$1" in
        graph) echo "agents/graph_agent.py" ;;
        llm-vqvae) echo "agents/llm_vqvae_agent.py" ;;
        memory-agent) echo "agents/memory_agent.py" ;;
        *) echo "agents/${1//-/_}_agent.py" ;;
    esac
}

# 1. Categorize tasks
echo "Step 1: Categorizing tasks..."
python scripts/categorize_tasks.py -o data/task_categories.json

# 2. Select diagnostic tasks
echo "Step 2: Selecting diagnostic tasks..."
if [[ -n "$DIAGNOSTIC_CONFIG" ]]; then
    python scripts/select_diagnostic_tasks.py --from-config "$DIAGNOSTIC_CONFIG" -o data/diagnostic_tasks.json
else
    python scripts/select_diagnostic_tasks.py -o data/diagnostic_tasks.json
fi

# 3. Run diagnostic tests
echo "Step 3: Running diagnostic tests..."
for agent in "${AGENTS[@]}"; do
    f=$(get_agent_file "$agent")
    [[ -f "$f" ]] || { echo "Skip $agent: $f not found"; continue; }
    extra=""
    [[ "$agent" == "graph" ]] && extra="--conversation"
    python benchmark.py --agent "$f" "$agent" \
        --diagnostic-tests data/diagnostic_tasks.json \
        --admissible-commands --nb-steps 50 --seed 20241001 \
        ${API_KEY:+--api-key "$API_KEY"} \
        $extra \
        --output-metrics "logs/${agent}_diagnostic.json"
done

# 4. Full benchmark
echo "Step 4: Full benchmark..."
ENVS=$(python -c "import json; print(' '.join(json.load(open('data/evaluation_subset.json'))['envs']))")
for agent in "${AGENTS[@]}"; do
    f=$(get_agent_file "$agent")
    [[ -f "$f" ]] || continue
    extra=""
    [[ "$agent" == "graph" ]] && extra="--conversation"
    python benchmark.py --agent "$f" "$agent" --envs $ENVS \
        --admissible-commands --nb-steps 100 --seed 20241001 \
        ${API_KEY:+--api-key "$API_KEY"} $extra
done

# 5. Transfer analysis
echo "Step 5: Transfer analysis..."
for agent in "${AGENTS[@]}"; do
    [[ -f "logs/${agent}_diagnostic.json" ]] || continue
    dir=$(find logs -maxdepth 1 -type d -name "tales_*${agent//-/_}*" 2>/dev/null | head -1)
    dir=${dir:-logs}
    python scripts/analyze_skill_transfer.py -d "logs/${agent}_diagnostic.json" \
        -f "$dir" -c data/task_categories.json -o "data/${agent}_transfer.json" 2>/dev/null || true
done

# 6. Run hybrid agents
echo "Step 6: Running hybrid agents..."
VQVAE_CKPT="latent-action/checkpoints/vqvae_checkpoint.pt"
NB_STEPS=100
if [[ -z "$API_KEY" ]]; then
    echo "Skipping hybrids: API_KEY not set"
else

# Common args for equalized comparison
COMMON="--admissible-commands --seed 20241001"

# Graph + VQ-VAE (full benchmark)
python benchmark.py --agent agents/hybrid_agents.py graph-vqvae \
  --api-key "$API_KEY" --vqvae-checkpoint "$VQVAE_CKPT" \
  --graph-weight 0.6 --vqvae-weight 0.4 \
  $COMMON --envs $ENVS --nb-steps $NB_STEPS 2>/dev/null || true

# Graph + VQ-VAE (diagnostic)
python benchmark.py --agent agents/hybrid_agents.py graph-vqvae \
  --api-key "$API_KEY" --vqvae-checkpoint "$VQVAE_CKPT" \
  --graph-weight 0.6 --vqvae-weight 0.4 \
  $COMMON --diagnostic-tests data/diagnostic_tasks.json --nb-steps 50 \
  --output-metrics logs/hybrid_gv_diagnostic.json 2>/dev/null || true

# Memory + ReAct (full + diagnostic)
python benchmark.py --agent agents/hybrid_agents.py memory-react \
  --api-key "$API_KEY" --memory-weight 0.5 --react-weight 0.5 \
  --conversation $COMMON --envs $ENVS --nb-steps $NB_STEPS 2>/dev/null || true

python benchmark.py --agent agents/hybrid_agents.py memory-react \
  --api-key "$API_KEY" --memory-weight 0.5 --react-weight 0.5 \
  --conversation $COMMON --diagnostic-tests data/diagnostic_tasks.json --nb-steps 50 \
  --output-metrics logs/hybrid_mr_diagnostic.json 2>/dev/null || true

# Full Hybrid (full + diagnostic)
python benchmark.py --agent agents/hybrid_agents.py full-hybrid \
  --api-key "$API_KEY" --vqvae-checkpoint "$VQVAE_CKPT" \
  --graph-weight 0.3 --vqvae-weight 0.3 --memory-weight 0.2 --react-weight 0.2 \
  --conversation $COMMON --envs $ENVS --nb-steps $NB_STEPS 2>/dev/null || true

python benchmark.py --agent agents/hybrid_agents.py full-hybrid \
  --api-key "$API_KEY" --vqvae-checkpoint "$VQVAE_CKPT" \
  --graph-weight 0.3 --vqvae-weight 0.3 --memory-weight 0.2 --react-weight 0.2 \
  --conversation $COMMON --diagnostic-tests data/diagnostic_tasks.json --nb-steps 50 \
  --output-metrics logs/hybrid_full_diagnostic.json 2>/dev/null || true
fi

# 7. Generate plots including hybrids
echo "Step 7: Generating plots with hybrid results..."
mkdir -p plots
files=()
for agent in "${AGENTS[@]}"; do
    [[ -f "logs/${agent}_diagnostic.json" ]] && files+=("logs/${agent}_diagnostic.json")
done
[[ -f "logs/hybrid_gv_diagnostic.json" ]] && files+=("logs/hybrid_gv_diagnostic.json")
[[ ${#files[@]} -gt 0 ]] && python scripts/plot_results.py diagnostic-comparison -m "${files[@]}" -o plots/

# Skill profiles (single + hybrid diagnostics)
profile_files=("${files[@]}")
[[ ${#profile_files[@]} -gt 0 ]] && python scripts/plot_results.py skill-profiles -m "${profile_files[@]}" -o plots/

# Hybrid comparison (single agents + hybrid diagnostics)
comp_files=("${files[@]}")
[[ -f "logs/hybrid_mr_diagnostic.json" ]] && comp_files+=("logs/hybrid_mr_diagnostic.json")
[[ -f "logs/hybrid_full_diagnostic.json" ]] && comp_files+=("logs/hybrid_full_diagnostic.json")
[[ ${#comp_files[@]} -gt 0 ]] && python scripts/plot_results.py hybrid-comparison -m "${comp_files[@]}" -o plots/

echo "Done."
