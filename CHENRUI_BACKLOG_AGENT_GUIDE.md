# How to Run Chenrui's Backlog Agents

Backlog Agents currently has two variants:


## 1. Backlog Agent 1: `backlog.py`

`agents/backlog.py` is a **3-stage LLM agent**:

- Goal Maker (long-term goal)
- Planner (short-term task maintenance)
- Actor (single-step action execution)

### Command

```bash
python benchmark_for_backlog.py --agent agents/backlog.py backlog --conversation --envs textworld --nb-steps 50 --admissible-commands --llm api-llama-4-scout -ff --seed 1
```

### Design Overview

#### Goal Maker (Long-Term Goal)
- Called once at the beginning of evaluation.
- Infers a stable long-term objective from the initial game observation.
- Purpose: reduce wasted interactions with irrelevant items and keep behavior aligned with the final objective.

#### Planner (Short-Term Tasks)
- Maintains and updates the active short-term task.
- Judges task state (for example: `progressing`, `completed_and_useful`, `failed`, `unprogressed`).
- Proposes candidate tasks that should help advance the long-term goal.

#### Actor (Action Selection)
- Chooses one concrete action per step.
- Uses both:
  - the long-term goal as global direction,
  - the current task as the immediate target.

### Map-Augmented Navigation

To improve navigation, the agent keeps a lightweight map memory, including:

- visited rooms,
- explored / tried exits per room,
- discovered objects,
- known room-to-room connections.

This map summary is injected into prompts to support more systematic exploration.

### Main Limitation Observed

In our tests, `api-llama-4-scout` showed weaker behavior in the **planner** stage:

- it did not always correctly judge the completion state of the previous task,
- this could cause incorrect task updates and then mislead actor decisions.

This was especially visible when tasks were broad, for example `find kitchen`:

- the planner could classify the task as `unprogressed` too early instead of `progressing`,
- this could make the actor stop searching for the location of interest.

We also observed a tradeoff:

- shorter-horizon tasks are easier for the planner to evaluate correctly,
- but if tasks become too short, the planner can degenerate into an action recommender.

---

## 2. Backlog Agent 2: `backlog2.py`

`agents/backlog2.py` is the agent that **merges planner and actor into one stage**.

Instead of maintaining explicit short-term tasks, it uses a single **action recommender** at each step.

### Why this version was introduced

This version was designed to address the main weakness above:

- some models were not reliable at judging whether a previous **task** was completed,
- task-state mistakes could cascade into bad replanning,
- the overall system became brittle when the planner was wrong.

The new design removes that task-state bottleneck.

### New High-Level Structure

`backlog2.py` uses:

- Goal Maker (long-term goal)
- Action Recommender (classify previous action + recommend next action)

So compared with `backlog.py`:

- **Planner is removed**
- **Actor is removed as a separate stage**
- both responsibilities are merged into a single recommender call per step

### Command

The command-line style is intentionally kept similar to the original version:

```bash
python benchmark_for_backlog2.py --agent agents/backlog2.py backlog2 --conversation --envs textworld --nb-steps 50 --admissible-commands --llm api-llama-4-scout -ff --seed 1
```

### What the Action Recommender Does

At each step, it does two things:

1. judges the outcome of the **previous action**,
2. recommends exactly one **next action**.

The previous action is classified into one of four categories:

- `useful_scoring`
- `useful_non_scoring`
- `useless`
- `failed`

This is easier for weaker models than judging broad task states such as `progressing` or `unprogressed` for multi-step tasks.

### Why this can help

Compared with task-level planning, action-level judgement is:

- shorter horizon,
- more concrete,
- easier to verify from the latest observation,
- less likely to drift because of ambiguous task boundaries.

### Action Memory

`backlog2.py` also keeps action memory grouped by outcome:

- useful scoring actions,
- useful non-scoring actions,
- useless actions,
- failed actions.

These memories are stored in a deduplicated way:

- the key is `(action, location)`,
- the same action at the same location is treated as the same entry,
- only the `last_episode` is updated when it happens again.

This keeps memory compact and avoids long repeated lists in prompts.


### Current Remaining Problem

One issue still observed in practice is:

- the model sometimes still repeats actions that failed previously.

So while the new design improves robustness against bad task-state judgement, it does **not completely eliminate repeated failed actions**.

This remains an active weakness of the current `backlog2.py` pipeline.

---

## 3. Model Notes

- `--llm api-llama-4-scout` uses a model registered in `.llm/extra-openai-models.yaml`.
- Put your own API key in `.llm/keys.json` and make sure it matches the model's `api_key_name` in `.llm/extra-openai-models.yaml`.
- You can similarly switch to other models registered in the same file.

Observed behavior:

- reasoning-style models sometimes returned empty outputs in this pipeline,
- a likely reason is response-format mismatch with the current parser assumptions.

Recommendation:

- prefer non-reasoning chat-style models for this setup,
- and validate output-format compliance carefully when changing models.

---

## 4. Benchmark Scripts

### `benchmark_for_backlog.py`
This is the benchmark script for the original 3-stage agent.

It is mainly the same as the normal benchmark flow, but adds extra logging for:

- goal maker,
- planner,
- actor.

### `benchmark_for_backlog2.py`
This is the benchmark script for the new single-stage recommender agent.

It is similarly based on the standard benchmark flow, but logs recommender-specific information such as:

- long-term goal,
- previous action outcome,
- next action,
- map summary,
- action-memory snapshots.

The command-line style is kept close to the original backlog benchmark so the transition is easy.

---

## 5. Summary

### `backlog.py`
- 3-stage design
- Goal Maker + Planner + Actor
- stronger explicit task structure
- vulnerable when the planner misjudges task state

### `backlog2.py`
- 2-stage design
- Goal Maker + Action Recommender
- planner and actor merged into one stage
- avoids the weakest task-state judgement bottleneck
- still may repeat previously failed actions in some cases

