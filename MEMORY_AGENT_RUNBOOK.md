# Memory Agent Runbook

This file explains how to run the custom memory agent in TALES with different parameter combinations.

## 1) Prerequisites

- From repo root: `/root/tale-suite`
- Python env with dependencies installed.
- For LLM-enabled runs, set your API key in shell:

```bash
export TRITON_API_KEY="<your_key_here>"
```

Default Triton endpoint/model used by this agent:
- URL: `https://tritonai-api.ucsd.edu/v1/chat/completions`
- Model: `api-llama-4-scout`

## 2) Minimal No-LLM Run (Structured Memory)

Uses:
- heuristic observation parser
- heuristic compression
- heuristic action policy

```bash
python benchmark.py \
  --agent agents/memory_agent.py \
  memory-agent \
  --envs textworld \
  --memory-variant structured \
  --compress-every 8 \
  --admissible-commands
```

## 3) No-LLM RAG Variant

Uses retrieval over compressed memories, no remote model calls.

```bash
python benchmark.py \
  --agent agents/memory_agent.py \
  memory-agent \
  --envs textworld \
  --memory-variant rag \
  --use-retrieval \
  --compress-every 8 \
  --admissible-commands
```

## 4) LLM Policy Only

Uses LLM only for action selection. Scratchpad parser/compressor remain heuristic.

```bash
python benchmark.py \
  --agent agents/memory_agent.py \
  memory-agent \
  --envs textworld \
  --memory-variant structured \
  --compress-every 8 \
  --admissible-commands \
  --use-llm-policy \
  --llm-model api-llama-4-scout \
  --llm-api-url https://tritonai-api.ucsd.edu/v1/chat/completions \
  --llm-api-key-env TRITON_API_KEY
```

## 5) LLM Scratchpad + LLM Compression (Heuristic Policy)

Uses LLM for parser/compressor only; action policy remains heuristic.

```bash
python benchmark.py \
  --agent agents/memory_agent.py \
  memory-agent \
  --envs textworld \
  --memory-variant structured \
  --compress-every 8 \
  --admissible-commands \
  --use-llm-parser \
  --use-llm-compressor \
  --llm-model api-llama-4-scout \
  --llm-api-url https://tritonai-api.ucsd.edu/v1/chat/completions \
  --llm-api-key-env TRITON_API_KEY
```

## 6) Full LLM Stack (Policy + Parser + Compressor)

Uses LLM in all three components.

```bash
python benchmark.py \
  --agent agents/memory_agent.py \
  memory-agent \
  --envs textworld \
  --memory-variant structured \
  --compress-every 8 \
  --admissible-commands \
  --use-llm-policy \
  --use-llm-parser \
  --use-llm-compressor \
  --llm-model api-llama-4-scout \
  --llm-api-url https://tritonai-api.ucsd.edu/v1/chat/completions \
  --llm-api-key-env TRITON_API_KEY
```

## 7) Option Variant Example

```bash
python benchmark.py \
  --agent agents/memory_agent.py \
  memory-agent \
  --envs textworld \
  --memory-variant option \
  --use-option-module \
  --use-retrieval \
  --compress-every 8 \
  --admissible-commands
```

## 8) Run Multiple Variants with Wrapper Script

Dry run first:

```bash
python scripts/memory_benchmark_variants.py \
  --dry-run \
  --envs textworld \
  --variants baseline compressed structured rag option
```

Run selected variants without LLM:

```bash
python scripts/memory_benchmark_variants.py \
  --envs textworld \
  --variants structured rag option \
  --admissible-commands
```

Run selected variants with full LLM stack:

```bash
python scripts/memory_benchmark_variants.py \
  --envs textworld \
  --variants structured rag option \
  --admissible-commands \
  --use-llm-policy \
  --use-llm-parser \
  --use-llm-compressor \
  --llm-model api-llama-4-scout \
  --llm-api-key-env TRITON_API_KEY
```

## 9) Useful Runtime Parameters

- `--nb-steps 100`: maximum environment steps per game.
- `--compress-every 8`: compression interval.
- `--max-context-items 12`: cap retained memory items.
- `--seed 20241001`: reproducible randomness.
- `--envs textworld`: run all textworld environments in TALES.

## 10) Notes

- If LLM calls fail, parser/compressor/policy paths fall back to heuristic behavior where implemented.
- For faster local smoke checks, use very small `--nb-steps` (e.g., `10`).
- Logs are written under `logs/` by default.
