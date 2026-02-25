# How to run Jeff's Agents

Jeff's agents include the integrated baseline agent (e.g., `baseline`) and `zero-shot` agent.

To run Jeff's agents, use the `benchmark.py` script from the root directory.

## Examples

**Run the Baseline Agent on TWCookingLevel1:**
```bash
python benchmark.py baseline --envs TWCookingLevel1 --nb-steps 20
```

**Run the Zero-Shot Agent on an ALFWorld Task:**
```bash
python benchmark.py zero-shot --envs ALFWorldPickAndPlaceSimpleSeen --nb-steps 50
```

## Hyperparameters
Hyperparameters are set via command-line arguments. Default values are defined in each agent's argparser function or right in `agents/baseline.py` and `agents/llm.py`.

- `--llm`: The model to use (default: `gpt-4o-mini`)
- `--seed`: Seed for the LLM generation (default: `20241001`)
- `--act-temp`: Temperature for action generation (default: `0.0`)
- `--context-limit`: The number of steps to keep in the conversation context. For example, if you set it to `10`, the agent will only see the last 10 actions and observations.
- `--conversation`: Whether to maintain conversation history structure (turns) or flatten it.

You can overwrite any of these parameters. Example:
```bash
python benchmark.py baseline --envs TWCookingLevel1 --nb-steps 20 --act-temp 0.5 --context-limit 15
```
