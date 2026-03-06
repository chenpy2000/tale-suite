# How to run Chenrui's Backlog Agent

Chenrui's agent is a 3-stage LLM agent implemented in `agents/backlog.py`:
- Goal Maker (long-term goal)
- Planner (short-term task maintenance)
- Actor (single-step action execution)

Use `benchmark_for_backlog.py` from the project root.

## Command

```bash
python benchmark_for_backlog.py --agent agents/backlog.py backlog --conversation --envs textworld --nb-steps 50 --admissible-commands --llm api-llama-4-scout -ff
```

## Agent Design Overview

### 1) Goal Maker (Long-Term Goal)
- Called once at the beginning of evaluation.
- Infers a stable long-term objective from the initial game observation.
- Purpose: reduce wasted interactions with irrelevant items and keep behavior aligned with the final objective.

### 2) Planner (Short-Term Tasks)
- Maintains and updates the active short-term tasks.
- Judges task state (e.g., progressing, completed, failed, unprogressed).
- Proposes candidate next tasks that should help advance the long-term goal.

### 3) Actor (Action Selection)
- Chooses one concrete action per step.
- Uses both:
  - long-term goal (global direction),
  - short-term active task (immediate target).

## Map-Augmented Navigation

During testing, some models showed weak localization/navigation ability.

To address this, the agent keeps a lightweight map memory:
- visited rooms,
- explored/tried exits per room,
- discovered objects,
- known room-to-room connections.

This map summary is injected into prompts to support systematic exploration and navigation.

In practice, this sometimes helped the agent locate locations of interest (especially `kitchen`) in more complex layouts such as:
- `TWCookingLevel5`
- `TWCookingLevel6`
- `TWCookingLevel10`

## About `benchmark_for_backlog.py`

`benchmark_for_backlog.py` is mainly the same as `benchmark.py` for evaluation flow and reporting.
It only adds extra logging for goal maker, planner and actor outputs, which makes model behavior easier to inspect and debug.

## Experimental Finding

In our tests, `api-llama-4-scout` showed weaker behavior in the planner stage:
- it did not always correctly judge the completion state of the previous task,
- this could cause incorrect task updates and then mislead actor decisions.

This was especially visible when tasks were broad (for example, `find kitchen`):
- the planner could classify the task as `unprogressed` too early instead of `progressing`,
- this could make the actor stop searching for the location of interest.

We also observed a tradeoff:
- shorter-horizon tasks are easier for the planner to evaluate correctly,
- but if tasks become too short, the planner can degenerate into an action recommender.

A positive sign:
- the model often uses task history to avoid re-planning tasks that failed earlier.
