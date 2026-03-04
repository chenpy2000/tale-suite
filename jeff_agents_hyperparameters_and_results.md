# Jeff's Agents: Setup, Hyperparameters, and Results

This document describes how to run the RAG and Graph-Based RAG agents implemented in this project, details their hyperparameters, and records their evaluation results on the `textworld` (TWCooking) environments.

## API Key Requirements

Before running either agent, you must set the following environment variables:

```bash
export WEAVIATE_URL="<your-weaviate-cloud-url>"
export WEAVIATE_API_KEY="<your-weaviate-api-key>"
export OPENAI_API_KEY="<your-openai-key>"         # Required for Weaviate Vectorizer
export TRITON_API_KEY="<your-triton-api-key>"     # Required for api-gpt-oss-120b custom model
```

---

## 1. Graph Agent (`graph`)

The Graph Agent uses Information Extraction (IE) to parse text observations into a Knowledge Graph (NetworkX) of entity relationships. This graph sub-state is included in the prompt to prevent logic hallucinations.

### How to Run
```bash
python benchmark.py graph \
    --llm api-gpt-oss-120b \
    --envs textworld \
    --nb-steps 100 \
    --conversation \
    --admissible-commands
```
> **Note**: To force a re-run of a benchmark and ignore cached previous results, add the `-ff` (or `--force-all`) flag to this command!

### Hyperparameters
- **LLM Endpoint**: `api-gpt-oss-120b` (Triton AI Proxy)
- **IE Temperature**: `0.0` (Hardcoded in `agents/graph_agent.py` for deterministic extraction)
- **LLM Temperature**: `0.0` (Default for benchmark evaluation)
- **Number of Steps (`--nb-steps`)**: `100`
- **Use Admissible Commands**: `True`
- **Include Conversation History**: `True`
- **Graph State**: Full NetworkX Edge list stringified in `[Knowledge Graph Tracker]`

### Results (TextWorld Cooking Levels 1-10)
- **Mean Normalized Score**: **57.95%**
- **Average Token Efficiency**: Logged per game
- **Average Doom Loop Count**: Suppressed/Minimized by graph logic.

---

## 2. RAG Agent (`rag`)

The RAG Agent uses Weaviate to store every sequence of `(Observation, Action, Feedback, Reward)`. It retrieves the most semantically similar past states to augment its prompt.

### How to Run
```bash
python benchmark.py rag \
    --llm api-gpt-oss-120b \
    --envs textworld \
    --rag-top-k 3 \
    --nb-steps 100 \
    --conversation \
    --admissible-commands
```
> **Note**: To force a re-run of a benchmark and ignore cached previous results, add the `-ff` (or `--force-all`) flag to this command!

### Hyperparameters
- **LLM Endpoint**: `api-gpt-oss-120b` (Triton AI Proxy)
- **Vectorizer**: `text2vec-openai` (Configured in Weaviate Collection)
- **Retrieval K (`--rag-top-k`)**: `3`
- **LLM Temperature**: `0.0` (Default)
- **Number of Steps (`--nb-steps`)**: `100`
- **Use Admissible Commands**: `True`
- **Include Conversation History**: `True`

### Results (TextWorld Cooking Levels 1-10)
- **Mean Normalized Score**: **0.00%**
- **Conclusion**: Pure text RAG suffers from severe Context Pollution in rigid logic environments like TextWorld, confusing the Triton endpoint. Structured compression or memory graphs are strictly required here.
