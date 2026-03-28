#!/bin/bash
# Full evaluation pipeline: categorize -> select diagnostic tasks -> diagnostic tests ->
# full benchmark -> transfer analysis -> hybrid agents (optional) -> plots.
#
# Usage: ./run_full_evaluation.sh [--api-key KEY] [--nb-steps N] [--nb-steps-diagn N] [--diagnostic-config PATH]
# Env vars (API_KEY, NB_STEPS, etc.) override defaults; CLI args override env vars.

set -e

# --- CLI argument parsing ---
# --api-key: TritonAI key for LLM agents (graph, llm-vqvae, memory-agent, hybrids)
# --nb-steps: Max steps per episode for full benchmark (default 100)
# --nb-steps-diagn: Max steps for diagnostic runs (default 50)
# --diagnostic-config: YAML config for TWCooking-only diagnostic subset
while [[ $# -gt 0 ]]; do
    case "$1" in
        --api-key) API_KEY="$2"; shift 2 ;;
        --api-key=*) API_KEY="${1#*=}"; shift ;;
        --nb-steps) NB_STEPS="$2"; shift 2 ;;
        --nb-steps=*) NB_STEPS="${1#*=}"; shift ;;
        --nb-steps-diagn) NB_STEPS_DIAG="$2"; shift 2 ;;
        --nb-steps-diagn=*) NB_STEPS_DIAG="${1#*=}"; shift ;;
        --diagnostic-config) DIAGNOSTIC_CONFIG="$2"; shift 2 ;;
        --diagnostic-config=*) DIAGNOSTIC_CONFIG="${1#*=}"; shift ;;
        -h|--help) echo "Usage: $0 [--api-key KEY] [--nb-steps N] [--nb-steps-diagn N] [--diagnostic-config PATH]"; exit 0 ;;
        *) echo "Unknown arg: $1 (use --help)"; exit 1 ;;
    esac
done

# Load .env if present; cd to project root
[ -f .env ] && export $(cat .env | xargs)
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(pwd)"

# Suppress HuggingFace tokenizers fork warning (benchmark spawns envs/subprocesses after tokenizer load)
export TOKENIZERS_PARALLELISM=false

# Use project's .llm/ for extra-openai-models.yaml (api-gpt-oss-120b, api-llama-4-scout, etc.)
export LLM_USER_PATH="${LLM_USER_PATH:-$PROJECT_ROOT/.llm}"

# Ensure Triton keys are set when API_KEY is provided (graph, memory-agent, llm-vqvae use them)
[[ -n "$API_KEY" ]] && export TRITON_API_KEY="${TRITON_API_KEY:-$API_KEY}" && export TRITONAI_API_KEY="${TRITONAI_API_KEY:-$API_KEY}"

# Base agents (no hybrids): graph uses LLM+conversation; llm-vqvae uses VQ-VAE; memory-agent uses memory
AGENTS=(graph llm-vqvae memory-agent)
API_KEY="${API_KEY:-}"
DIAGNOSTIC_CONFIG="${DIAGNOSTIC_CONFIG:-}"
NB_STEPS="${NB_STEPS:-100}"
NB_STEPS_DIAG="${NB_STEPS_DIAG:-50}"

get_agent_file() {
    case "$1" in
        graph) echo "agents/graph_agent.py" ;;
        llm-vqvae) echo "agents/llm_vqvae_agent.py" ;;
        memory-agent) echo "agents/memory_agent.py" ;;
        *) echo "agents/${1//-/_}_agent.py" ;;
    esac
}

# --- Step 1: Categorize tasks ---
# Assigns skill/difficulty to envs; writes data/task_categories.json and data/evaluation_subset.json (20 envs, 5 per skill)
echo "Step 1: Categorizing tasks..."
python scripts/categorize_tasks.py -o data/task_categories.json

# --- Step 2: Select diagnostic tasks ---
# Picks one task per env for quick skill profiling; writes data/diagnostic_tasks.json
echo "Step 2: Selecting diagnostic tasks..."
if [[ -n "$DIAGNOSTIC_CONFIG" ]]; then
    python scripts/select_diagnostic_tasks.py --from-config "$DIAGNOSTIC_CONFIG" -o data/diagnostic_tasks.json
else
    python scripts/select_diagnostic_tasks.py -o data/diagnostic_tasks.json
fi

# --- Step 3: Run diagnostic tests ---
# Runs graph, llm-vqvae, memory-agent on diagnostic tasks only; writes logs/*_diagnostic.json
echo "Step 3: Running diagnostic tests..."
for agent in "${AGENTS[@]}"; do
    f=$(get_agent_file "$agent")
    [[ -f "$f" ]] || { echo "Skip $agent: $f not found"; continue; }
    extra=""
    [[ "$agent" == "graph" ]] && extra="--conversation --llm api-gpt-oss-120b"
    api_key_arg=""
    if [[ -n "$API_KEY" ]]; then
        [[ "$agent" == "llm-vqvae" ]] && api_key_arg="--api-key $API_KEY"
        [[ "$agent" == "graph" ]] && api_key_arg="--key $API_KEY"
        [[ "$agent" == "memory-agent" ]] && api_key_arg="--llm-api-key $API_KEY"
    fi
    python benchmark.py --agent "$f" "$agent" \
        --diagnostic-tests data/diagnostic_tasks.json \
        --admissible-commands --nb-steps ${NB_STEPS_DIAG} --seed 20241001 \
        $api_key_arg $extra \
        --output-metrics "logs/${agent}_diagnostic.json"
done

# --- Step 4: Full benchmark ---
# Same agents on all 20 envs in evaluation_subset.json (longer runs for final scores)
echo "Step 4: Full benchmark..."
ENVS=$(python -c "import json; print(' '.join(json.load(open('data/evaluation_subset.json'))['envs']))")
for agent in "${AGENTS[@]}"; do
    f=$(get_agent_file "$agent")
    [[ -f "$f" ]] || continue
    extra=""
    [[ "$agent" == "graph" ]] && extra="--conversation --llm api-gpt-oss-120b"
    api_key_arg=""
    if [[ -n "$API_KEY" ]]; then
        [[ "$agent" == "llm-vqvae" ]] && api_key_arg="--api-key $API_KEY"
        [[ "$agent" == "graph" ]] && api_key_arg="--key $API_KEY"
        [[ "$agent" == "memory-agent" ]] && api_key_arg="--llm-api-key $API_KEY"
    fi
    python benchmark.py --agent "$f" "$agent" --envs $ENVS \
        --admissible-commands --nb-steps ${NB_STEPS} --seed 20241001 \
        $api_key_arg $extra
done

# --- Step 5: Transfer analysis ---
# Predicts full-task scores from diagnostic skill profiles; writes data/*_transfer.json
echo "Step 5: Transfer analysis..."
for agent in "${AGENTS[@]}"; do
    [[ -f "logs/${agent}_diagnostic.json" ]] || continue
    dir=$(find logs -maxdepth 1 -type d -name "tales_*${agent//-/_}*" 2>/dev/null | head -1)
    dir=${dir:-logs}
    python scripts/analyze_skill_transfer.py -d "logs/${agent}_diagnostic.json" \
        -f "$dir" -c data/task_categories.json -o "data/${agent}_transfer.json" 2>/dev/null || true
done

# --- Step 6: Run hybrid agents ---
# graph-vqvae, memory-react, full-hybrid (skipped if API_KEY not set; requires vqvae_checkpoint.pt)
echo "Step 6: Running hybrid agents..."
VQVAE_CKPT="latent-action/checkpoints/vqvae_checkpoint.pt"
if [[ -z "$API_KEY" ]]; then
    echo "Skipping hybrids: API_KEY not set"
else

# Common args for equalized comparison (--conversation and --llm required by graph/react)
COMMON="--admissible-commands --seed 20241001 --conversation --llm api-gpt-oss-120b"

# Graph + VQ-VAE (full benchmark)
python benchmark.py --agent agents/hybrid_agents.py graph-vqvae \
  --api-key "$API_KEY" --vqvae-checkpoint "$VQVAE_CKPT" \
  --graph-weight 0.6 --vqvae-weight 0.4 \
  $COMMON --envs $ENVS --nb-steps $NB_STEPS 2>/dev/null || true

# Graph + VQ-VAE (diagnostic)
python benchmark.py --agent agents/hybrid_agents.py graph-vqvae \
  --api-key "$API_KEY" --vqvae-checkpoint "$VQVAE_CKPT" \
  --graph-weight 0.6 --vqvae-weight 0.4 \
  $COMMON --diagnostic-tests data/diagnostic_tasks.json --nb-steps ${NB_STEPS_DIAG} \
  --output-metrics logs/hybrid_gv_diagnostic.json 2>/dev/null || true

# Memory + ReAct (full + diagnostic)
python benchmark.py --agent agents/hybrid_agents.py memory-react \
  --api-key "$API_KEY" --memory-weight 0.5 --react-weight 0.5 \
  $COMMON --envs $ENVS --nb-steps $NB_STEPS 2>/dev/null || true

python benchmark.py --agent agents/hybrid_agents.py memory-react \
  --api-key "$API_KEY" --memory-weight 0.5 --react-weight 0.5 \
  $COMMON --diagnostic-tests data/diagnostic_tasks.json --nb-steps ${NB_STEPS_DIAG} \
  --output-metrics logs/hybrid_mr_diagnostic.json 2>/dev/null || true

# Full Hybrid (full + diagnostic)
python benchmark.py --agent agents/hybrid_agents.py full-hybrid \
  --api-key "$API_KEY" --vqvae-checkpoint "$VQVAE_CKPT" \
  --graph-weight 0.3 --vqvae-weight 0.3 --memory-weight 0.2 --react-weight 0.2 \
  $COMMON --envs $ENVS --nb-steps $NB_STEPS 2>/dev/null || true

python benchmark.py --agent agents/hybrid_agents.py full-hybrid \
  --api-key "$API_KEY" --vqvae-checkpoint "$VQVAE_CKPT" \
  --graph-weight 0.3 --vqvae-weight 0.3 --memory-weight 0.2 --react-weight 0.2 \
  $COMMON --diagnostic-tests data/diagnostic_tasks.json --nb-steps ${NB_STEPS_DIAG} \
  --output-metrics logs/hybrid_full_diagnostic.json 2>/dev/null || true
fi

# --- Step 7: Generate plots ---
# diagnostic_comparison.png, skill_profiles.png, hybrid_comparison.png in plots/
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
