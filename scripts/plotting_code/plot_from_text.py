#!/usr/bin/env python3
"""
Parse raw benchmark terminal output and generate comparison plots.

Handles interrupted/partial runs — no final summary line required.
Accepts plain text files, one per agent run.

INPUT FORMAT
------------
Each input is  FILE[:AgentName]
  - FILE      : path to a text file containing benchmark terminal output
  - AgentName : optional display name (defaults to filename stem)

USAGE
-----
# All comparison plots (mean score, per-level, skill radar, efficiency)
python scripts/plotting_code/plot_from_text.py compare \\
    runs/vqvae_100.txt:VQ-VAE \\
    runs/graph_100.txt:Graph \\
    runs/react_100.txt:ReAct \\
    runs/memory_100.txt:Memory \\
    runs/hybrid_100.txt:"Graph+VQ-VAE" \\
    --output plots/

# Step-budget line chart (multiple nb_steps for each agent)
# Format: "AgentName:file_at_10,file_at_20,file_at_50,file_at_100,file_at_200"
python scripts/plotting_code/plot_from_text.py step-budget \\
    "VQ-VAE:runs/vqvae_10.txt,runs/vqvae_20.txt,runs/vqvae_50.txt,runs/vqvae_100.txt,runs/vqvae_200.txt" \\
    "Graph:runs/graph_10.txt,runs/graph_50.txt,runs/graph_100.txt,runs/graph_200.txt" \\
    --output plots/step_budget.png

OUTPUT (compare mode)
---------------------
plots/mean_score.png      — overall mean normalized score per agent
plots/per_level.png       — per-environment grouped bar chart
plots/skill_radar.png     — radar chart (spatial/deductive/inductive/grounded)
plots/efficiency.png      — wall time and score-per-minute side-by-side
plots/doom_loops.png      — doom loop count per agent (omitted if all zero)
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ---------------------------------------------------------------------------
# Skill mapping (from data/diagnostic_tasks.json)
# ---------------------------------------------------------------------------

_SKILL_MAP_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "diagnostic_tasks.json"

SKILLS = ["spatial", "deductive", "inductive", "grounded"]

def _build_env_skill_map():
    if not _SKILL_MAP_PATH.exists():
        return {}
    with open(_SKILL_MAP_PATH) as f:
        cats = json.load(f)
    out = {}
    for skill, tasks in cats.items():
        for t in tasks:
            out[t["task"]] = skill
    return out

ENV_SKILL_MAP = _build_env_skill_map()

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

_PALETTE = [
    "#3498db",  # blue
    "#e74c3c",  # red
    "#2ecc71",  # green
    "#f39c12",  # orange
    "#9b59b6",  # purple
    "#1abc9c",  # teal
    "#e67e22",  # dark orange
    "#34495e",  # dark grey
    "#7f8c8d",  # grey
    "#8e44ad",  # violet
    "#16a085",  # turquoise
    "#c0392b",  # dark red
]


def _agent_color(name, idx):
    """Assign a (nearly) unique color per agent index.

    For up to len(_PALETTE) agents, uses the fixed hex palette.
    Beyond that, falls back to matplotlib's tab20 colormap.
    """
    if idx < len(_PALETTE):
        return _PALETTE[idx]
    cmap = plt.get_cmap("tab20")
    return cmap(idx % 20)

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

# Matches lines like:
#   TWCookingLevel1   Steps:   50/  50  Time: 0:01:35   1 resets  Score:   1/  3 (33.33%)  TokenEff:     0.00  DoomLoop:    0
_ENV_RE = re.compile(
    r"(?:^|\n)\s*"
    r"([\w]+)"                              # env name
    r"\s+Steps:\s+\d+/\s*(\d+)"            # budget (second number)
    r"\s+Time:\s+(\d+):(\d+):(\d+)"        # H:M:S
    r"\s+\d+\s+resets"
    r"\s+Score:\s+(\d+)/\s*(\d+)"          # score / max_score
    r"\s+\([\s\d.]+%\)"
    r"\s+TokenEff:\s+[\d.\-]+"
    r"\s+DoomLoop:\s+(\d+)",               # doom loops
    re.MULTILINE,
)

def parse_run(text: str) -> dict:
    """
    Parse one benchmark text stream. Returns a dict:
    {
        "envs":           [ { env, budget, score, max_score, norm_score,
                              time_s, doom_loops }, ... ],
        "mean_norm_score": float,       # computed from parsed envs
        "total_time_s":    float,
        "total_doom_loops": int,
        "budget":          int,         # most common step budget seen
    }
    Handles partial/interrupted runs — uses whatever envs were completed.
    De-duplicates on env name (keeps the last occurrence, in case of restarts).
    """
    env_map = {}  # env_name -> result dict (last occurrence wins for restarts)
    for m in _ENV_RE.finditer(text):
        env = m.group(1)
        budget = int(m.group(2))
        h, mi, s = int(m.group(3)), int(m.group(4)), int(m.group(5))
        score = int(m.group(6))
        max_score = int(m.group(7))
        doom = int(m.group(8))
        env_map[env] = {
            "env": env,
            "budget": budget,
            "score": score,
            "max_score": max_score,
            "norm_score": score / max_score if max_score > 0 else 0.0,
            "time_s": h * 3600 + mi * 60 + s,
            "doom_loops": doom,
        }

    envs = list(env_map.values())

    # Mean of per-game normalized scores — matches how benchmark.py reports it
    mean_norm = sum(r["norm_score"] for r in envs) / len(envs) if envs else 0.0

    # Determine representative step budget (most common value)
    if envs:
        budgets = [r["budget"] for r in envs]
        budget = max(set(budgets), key=budgets.count)
    else:
        budget = 0

    return {
        "envs": envs,
        "mean_norm_score": mean_norm,
        "total_time_s": sum(r["time_s"] for r in envs),
        "total_doom_loops": sum(r["doom_loops"] for r in envs),
        "budget": budget,
    }


def load_agent(spec: str) -> tuple:
    """
    Parse a FILE[:AgentName] spec.
    Returns (agent_name, parsed_run_dict).
    """
    if ":" in spec:
        # Split on first colon only (agent name may contain colons like Graph+VQ-VAE)
        colon = spec.index(":")
        path_str = spec[:colon]
        name = spec[colon + 1:]
    else:
        path_str = spec
        name = None

    path = Path(path_str)
    if not path.exists():
        print(f"[warn] File not found: {path}", file=sys.stderr)
        return name or path.stem, {"envs": [], "mean_norm_score": 0.0,
                                    "total_time_s": 0.0, "total_doom_loops": 0,
                                    "budget": 0}

    text = path.read_text(encoding="utf-8", errors="replace")

    if name is None:
        # Try to extract from log path line: "tales_AGENTNAME_..."
        m = re.search(r"tales_(\w+?)_", text)
        name = m.group(1) if m else path.stem

    return name, parse_run(text)

# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _savefig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def _bar_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25, axis="y", linestyle="--")

# ---------------------------------------------------------------------------
# Plot 1: Mean Score
# ---------------------------------------------------------------------------

def plot_mean_score(agents: list, out_dir: Path):
    """Bar chart: overall mean normalized score per agent."""
    names = [a for a, _ in agents]
    scores = [r["mean_norm_score"] * 100 for _, r in agents]
    colors = [_agent_color(n, i) for i, n in enumerate(names)]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.4), 5))
    bars = ax.bar(names, scores, color=colors, width=0.55, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean Normalized Score (%)", fontsize=12)
    ax.set_title("Overall Agent Performance", fontsize=14, fontweight="bold")
    # Dynamic y-scale with headroom to avoid excessive whitespace.
    if scores:
        max_score = max(scores)
        y_top = max(5.0, min(110.0, max_score * 1.25))
        ax.set_ylim(0, y_top)
    else:
        ax.set_ylim(0, 5)
    _bar_style(ax)
    plt.tight_layout()
    _savefig(fig, out_dir / "mean_score.png")


# ---------------------------------------------------------------------------
# Plot 2: Per-Level / Per-Environment
# ---------------------------------------------------------------------------

def plot_per_level(agents: list, out_dir: Path):
    """Grouped bar chart: per-environment normalized score."""
    # Collect all env names across agents (preserve order)
    env_order = []
    seen = set()
    for _, run in agents:
        for e in run["envs"]:
            if e["env"] not in seen:
                env_order.append(e["env"])
                seen.add(e["env"])

    if not env_order:
        print("  [skip] per_level.png — no environment data")
        return

    n_envs = len(env_order)
    n_agents = len(agents)
    w = min(0.7, 0.8 / n_agents)
    offsets = np.linspace(-(n_agents - 1) * w / 2, (n_agents - 1) * w / 2, n_agents)
    x = np.arange(n_envs)

    fig, ax = plt.subplots(figsize=(max(10, n_envs * 1.0), 5))
    for i, (name, run) in enumerate(agents):
        score_map = {e["env"]: e["norm_score"] * 100 for e in run["envs"]}
        vals = [score_map.get(env, float("nan")) for env in env_order]
        color = _agent_color(name, i)
        bars = ax.bar(x + offsets[i], vals, w, label=name, color=color,
                      alpha=0.88, edgecolor="white", linewidth=0.6)

    ax.set_ylabel("Normalized Score (%)", fontsize=12)
    ax.set_title("Per-Environment Score by Agent", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(env_order, rotation=40, ha="right", fontsize=9)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.9)
    _bar_style(ax)
    plt.tight_layout()
    _savefig(fig, out_dir / "per_level.png")


# ---------------------------------------------------------------------------
# Plot 3: Skill Radar
# ---------------------------------------------------------------------------

def _skill_scores(run: dict) -> dict:
    """Map env results to skill categories. Returns {skill: mean_norm_score}."""
    buckets = {s: [] for s in SKILLS}
    for e in run["envs"]:
        skill = ENV_SKILL_MAP.get(e["env"])
        if skill and skill in buckets:
            buckets[skill].append(e["norm_score"])
    return {s: (float(np.mean(v)) if v else None) for s, v in buckets.items()}


def plot_skill_radar(agents: list, out_dir: Path):
    """Radar chart: skill profile per agent."""
    # Determine which skills have coverage across any agent
    active_skills = []
    for s in SKILLS:
        for _, run in agents:
            sc = _skill_scores(run)
            if sc.get(s) is not None:
                active_skills.append(s)
                break

    if not active_skills:
        print("  [skip] skill_radar.png — no environments matched skill categories")
        return

    n = len(active_skills)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(projection="polar"))
    all_vals = []

    for i, (name, run) in enumerate(agents):
        sc = _skill_scores(run)
        vals = [sc.get(s, 0.0) or 0.0 for s in active_skills]
        all_vals.extend(vals)
        vals += vals[:1]
        color = _agent_color(name, i)
        ax.plot(angles, vals, "o-", lw=2, color=color, label=name, markersize=6)
        ax.fill(angles, vals, alpha=0.12, color=color)
        # Labels are placed after r_max is computed below.

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([s.capitalize() for s in active_skills], fontsize=12, fontweight="bold")
    # Dynamic radial scaling so low scores don't collapse near center.
    max_val = max(all_vals) if all_vals else 1.0
    if max_val <= 0:
        r_max = 0.1
    else:
        r_max = max_val * 1.25
        # Round to a readable cap.
        if r_max <= 0.2:
            r_max = 0.2
        elif r_max <= 0.4:
            r_max = 0.4
        elif r_max <= 0.6:
            r_max = 0.6
        elif r_max <= 0.8:
            r_max = 0.8
        else:
            r_max = 1.0
    ax.set_ylim(0, r_max)
    ticks = np.linspace(r_max / 4, r_max, 4)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t*100:.0f}%" for t in ticks], fontsize=8)
    # Now add value labels with dynamic offset/clipping to current radial max.
    label_offset = max(0.02, r_max * 0.06)
    for i, (name, run) in enumerate(agents):
        sc = _skill_scores(run)
        vals = [sc.get(s, 0.0) or 0.0 for s in active_skills]
        color = _agent_color(name, i)
        for theta, r_val in zip(angles[:-1], vals):
            ax.text(theta, min(r_val + label_offset, r_max), f"{r_val*100:.0f}%",
                    ha="center", va="bottom", fontsize=8, color=color)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=10)
    ax.set_title("Agent Skill Profiles", fontsize=14, fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _savefig(fig, out_dir / "skill_radar.png")


# ---------------------------------------------------------------------------
# Plot 4: Efficiency (wall time + score/minute)
# ---------------------------------------------------------------------------

def plot_efficiency(agents: list, out_dir: Path):
    """Two-panel bar chart: wall time (minutes) and score per minute."""
    names = [a for a, _ in agents]
    colors = [_agent_color(n, i) for i, n in enumerate(names)]
    times_min = [r["total_time_s"] / 60.0 for _, r in agents]
    score_per_min = [
        (r["mean_norm_score"] * 100) / (r["total_time_s"] / 60.0)
        if r["total_time_s"] > 0 else 0.0
        for _, r in agents
    ]

    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(names) * 2.2), 5))

    for ax, vals, ylabel, title in [
        (axes[0], times_min, "Wall Time (minutes)", "Total Wall Time"),
        (axes[1], score_per_min, "Score % per Minute", "Score Efficiency"),
    ]:
        bars = ax.bar(names, vals, color=colors, width=0.55,
                      edgecolor="white", linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.02,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticklabels(names, rotation=20, ha="right")
        _bar_style(ax)

    fig.suptitle("Computational Efficiency", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    _savefig(fig, out_dir / "efficiency.png")


# ---------------------------------------------------------------------------
# Plot 5: Doom Loops
# ---------------------------------------------------------------------------

def plot_doom_loops(agents: list, out_dir: Path):
    """Bar chart of total doom loops per agent. Skipped if all zero."""
    names = [a for a, _ in agents]
    loops = [r["total_doom_loops"] for _, r in agents]
    if all(v == 0 for v in loops):
        print("  [skip] doom_loops.png — all agents have 0 doom loops")
        return
    colors = [_agent_color(n, i) for i, n in enumerate(names)]
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.4), 4))
    bars = ax.bar(names, loops, color=colors, width=0.55,
                  edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, loops):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                str(v), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Total Doom Loops", fontsize=12)
    ax.set_title("Doom Loop Frequency per Agent", fontsize=14, fontweight="bold")
    _bar_style(ax)
    plt.tight_layout()
    _savefig(fig, out_dir / "doom_loops.png")


# ---------------------------------------------------------------------------
# Plot 6: Step Budget (separate subcommand)
# ---------------------------------------------------------------------------

def plot_step_budget(agent_specs: list, out_path: Path):
    """
    Line chart: score vs nb_steps, one line per agent.

    agent_specs: list of "AgentName:file1,file2,file3" strings.
    Each file is a text run at a different nb_steps.
    nb_steps is auto-detected from the parsed output.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, spec in enumerate(agent_specs):
        colon = spec.index(":")
        agent_name = spec[:colon]
        files = spec[colon + 1:].split(",")

        points = []
        for fpath in files:
            fpath = fpath.strip()
            if not Path(fpath).exists():
                print(f"  [warn] Not found: {fpath}", file=sys.stderr)
                continue
            text = Path(fpath).read_text(encoding="utf-8", errors="replace")
            run = parse_run(text)
            if run["envs"] and run["budget"] > 0:
                points.append((run["budget"], run["mean_norm_score"] * 100))

        if not points:
            print(f"  [warn] No valid data for agent: {agent_name}", file=sys.stderr)
            continue

        points.sort(key=lambda p: p[0])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        color = _agent_color(agent_name, i)
        marker = "D" if "hybrid" in agent_name.lower() else "o"
        ax.plot(xs, ys, marker=marker, lw=2.5, markersize=9,
                color=color, label=agent_name, alpha=0.9)
        for x_val, y_val in zip(xs, ys):
            ax.annotate(f"{y_val:.1f}%", (x_val, y_val),
                        textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=8, color=color)

    ax.set_xlabel("Step Budget (nb_steps)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Mean Normalized Score (%)", fontsize=13, fontweight="bold")
    ax.set_title("Agent Performance vs Step Budget", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left", framealpha=0.9)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Step Budget from runs/ text files
# ---------------------------------------------------------------------------

_RUN_NAME_RE = re.compile(
    r"^(?:cooking_)?(?P<agent>[A-Za-z0-9\-]+)_(?P<steps>\d+)\.txt$"
)


def plot_step_budget_from_runs(patterns: list, out_path: Path, metric: str = "score"):
    """
    Build a step-budget line plot directly from text files in runs/.

    Expected naming pattern (in runs/):
      [cooking_]AGENT_NB_STEPS.txt
        e.g., vqvae_50.txt, cooking_graph_100.txt, react_200.txt

    For each matching file, we parse the run, extract:
      - nb_steps from filename
      - mean score from parsed text
    and then plot mean score vs nb_steps (one line per agent).
    """
    runs_dir = Path("runs")
    if not runs_dir.exists():
        print(f"[error] runs/ directory not found at {runs_dir.resolve()}", file=sys.stderr)
        return

    # Expand simple globs like runs/vqvae_*.txt, runs/*_50.txt
    files = []
    for pat in patterns:
        p = Path(pat)
        if p.is_file():
            files.append(p)
        else:
            # Treat as glob relative to repo root
            for fp in Path(".").glob(pat):
                if fp.is_file():
                    files.append(fp)

    if not files:
        print("[error] No matching files for patterns:", patterns, file=sys.stderr)
        return

    agents = {}
    for path in files:
        name = path.name
        m = _RUN_NAME_RE.match(name)
        if not m:
            print(f"[warn] Skipping {name} (does not match [cooking_]agent_steps.txt pattern)", file=sys.stderr)
            continue
        agent = m.group("agent")
        steps = int(m.group("steps"))
        text = path.read_text(encoding="utf-8", errors="replace")
        run = parse_run(text)
        if not run["envs"]:
            print(f"[warn] Parsed 0 envs from {name}", file=sys.stderr)
            continue
        score_pct = run["mean_norm_score"] * 100.0
        agents.setdefault(agent, []).append((steps, score_pct))

    if not agents:
        print("[error] No usable runs parsed; nothing to plot.", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    all_y = []
    for i, (agent, pts) in enumerate(sorted(agents.items())):
        pts = sorted(pts, key=lambda p: p[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]  # mean score (%) by default
        all_y.extend(ys)
        color = _agent_color(agent, i)
        marker = "D" if "hybrid" in agent.lower() else "o"
        ax.plot(xs, ys, marker=marker, lw=2.5, markersize=9,
                color=color, label=agent, alpha=0.9)
        for x_val, y_val in zip(xs, ys):
            ax.annotate(f"{y_val:.1f}%", (x_val, y_val),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8, color=color)

    ax.set_xlabel("Step Budget (nb_steps)", fontsize=13, fontweight="bold")
    if metric == "wall-time":
        ax.set_ylabel("Wall Time (minutes)", fontsize=13, fontweight="bold")
        ax.set_title("Wall Time vs Step Budget", fontsize=15, fontweight="bold")
    elif metric == "efficiency":
        ax.set_ylabel("Score % per Minute", fontsize=13, fontweight="bold")
        ax.set_title("Score Efficiency vs Step Budget", fontsize=15, fontweight="bold")
    else:
        ax.set_ylabel("Mean Normalized Score (%)", fontsize=13, fontweight="bold")
        ax.set_title("Agent Performance vs Step Budget", fontsize=15, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left", framealpha=0.9)
    if all_y:
        y_top = max(all_y) * 1.25
        if metric == "efficiency":
            # efficiency values tend to be small; don't clamp to 100
            ax.set_ylim(0, max(all_y) * 1.5)
        else:
            ax.set_ylim(0, max(5.0, min(105.0, y_top)))
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_compare(args):
    agents = []
    for spec in args.inputs:
        name, run = load_agent(spec)
        n_envs = len(run["envs"])
        budget = run["budget"]
        print(f"  {name:<25} {n_envs} envs  budget={budget}  "
              f"mean={run['mean_norm_score']*100:.1f}%  "
              f"time={run['total_time_s']/60:.1f}min  "
              f"doom_loops={run['total_doom_loops']}")
        agents.append((name, run))

    out_dir = Path(args.output)
    print(f"\nGenerating plots in {out_dir}/")
    plot_mean_score(agents, out_dir)
    plot_per_level(agents, out_dir)
    plot_skill_radar(agents, out_dir)
    plot_efficiency(agents, out_dir)
    plot_doom_loops(agents, out_dir)
    print("\nDone.")


def cmd_step_budget(args):
    out = Path(args.output)
    print(f"Generating step-budget plot -> {out}")
    plot_step_budget(args.inputs, out)
    print("Done.")


def main():
    p = argparse.ArgumentParser(
        description="Parse benchmark terminal output and generate comparison plots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # compare subcommand
    c = sub.add_parser(
        "compare",
        help="All comparison plots for multiple agents.",
    )
    c.add_argument(
        "inputs",
        nargs="+",
        metavar="FILE[:AgentName]",
        help="Benchmark text files, optionally with agent display name after ':'",
    )
    c.add_argument(
        "-o", "--output",
        default="plots",
        metavar="DIR",
        help="Output directory for plots (default: plots/)",
    )

    # step-budget subcommand
    s = sub.add_parser(
        "step-budget",
        help="Line chart of score vs step budget for multiple agents.",
    )
    s.add_argument(
        "inputs",
        nargs="+",
        metavar="AgentName:file1,file2,...",
        help="Comma-separated list of text files per agent, one entry per agent",
    )
    s.add_argument(
        "-o", "--output",
        default="plots/step_budget.png",
        metavar="FILE",
        help="Output PNG path (default: plots/step_budget.png)",
    )

    # step-budget-from-runs subcommand
    s2 = sub.add_parser(
        "step-budget-from-runs",
        help="Line chart of score vs step budget using runs/[cooking_]agent_steps.txt files.",
    )
    s2.add_argument(
        "patterns",
        nargs="+",
        metavar="GLOB",
        help="Glob(s) for run files, e.g. 'runs/vqvae_*.txt' 'runs/graph_*.txt'",
    )
    s2.add_argument(
        "-o", "--output",
        default="plots/step_budget_runs.png",
        metavar="FILE",
        help="Output PNG path (default: plots/step_budget_runs.png)",
    )
    s2.add_argument(
        "--metric",
        choices=["score", "wall-time", "efficiency"],
        default="score",
        help="Y-axis: 'score' (mean %), 'wall-time' (minutes), or 'efficiency' (score %% per minute).",
    )

    args = p.parse_args()

    print(f"\n{'='*60}")
    if args.cmd == "compare":
        print(f"Comparing {len(args.inputs)} agent(s)")
        print("="*60)
        cmd_compare(args)
    elif args.cmd == "step-budget":
        print(f"Step-budget plot for {len(args.inputs)} agent(s)")
        print("="*60)
        cmd_step_budget(args)
    else:
        print(f"Step-budget-from-runs plot using patterns: {args.patterns}")
        print("="*60)
        plot_step_budget_from_runs(args.patterns, args.output, metric=args.metric)


if __name__ == "__main__":
    main()
