"""
Generate plots for diagnostic evaluation results.

Usage:
    # Plot 1: Radar chart for skill profiles
    python plot_diagnostic.py radar \
        --data vqvae_results.json graph_results.json react_results.json \
        --output plots/skill_radar.png
    
    # Plot 2: Line plot for step budget comparison
    python plot_diagnostic.py step-budget \
        --data step_budget_data.json \
        --output plots/step_budget.png
    
    # Plot 3: Bar chart for computation efficiency
    python plot_diagnostic.py efficiency \
        --data efficiency_data.json \
        --output plots/efficiency.png
    
    # Plot 4: Bar chart for per-skill comparison
    python plot_diagnostic.py skill-comparison \
        --data vqvae_results.json graph_results.json \
        --output plots/skill_comparison.png
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np


# Color scheme for agents
AGENT_COLORS = {
    'graph': '#2ecc71',
    'LLMVQVAE': '#3498db',
    'vqvae': '#3498db',
    'memory': '#f39c12',
    'react': '#e74c3c',
    'random': '#95a5a6',
    'graph-vqvae': '#9b59b6',
    'memory-react': '#e67e22',
    'full-hybrid': '#1abc9c',
    'hybrid_vqvae': '#9b59b6',
    'full_hybrid': '#1abc9c'
}

SKILL_ORDER = ['spatial', 'deductive', 'inductive', 'grounded']


def load_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def get_agent_color(agent_name):
    agent_lower = agent_name.lower()
    for key, color in AGENT_COLORS.items():
        if key in agent_lower:
            return color
    return '#34495e'


def auto_ylim(ax, values, margin=0.15):
    """Automatically scale y-axis based on data."""
    if not values:
        return
    ymax = max(values) * (1 + margin)
    ax.set_ylim(0, ymax)


# ============================================================================
# Plot 1: Radar Chart with TRUE Dynamic Scaling
# ============================================================================

def plot_radar_chart(data_files, output_path):
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    num_skills = len(SKILL_ORDER)
    angles = np.linspace(0, 2 * np.pi, num_skills, endpoint=False).tolist()
    angles += angles[:1]

    agent_scores = {}
    all_scores = []

    # Collect scores
    for data_file in data_files:
        data = load_data(data_file)
        agent = data['agent']

        if 'skill_scores' not in data:
            continue

        scores = [data['skill_scores'][s]['mean'] for s in SKILL_ORDER]
        agent_scores[agent] = scores
        all_scores.extend(scores)

    if not all_scores:
        print("No valid scores found.")
        return

    # -----------------------------
    # FIXED SCALE at 65%
    # -----------------------------
    r_max = 0.65  # outer ring fixed
    num_ticks = 4
    ticks = np.linspace(0, r_max, num_ticks)
    tick_labels = [f"{t*100:.0f}%" for t in ticks]

    # -----------------------------
    # Plot agents
    # -----------------------------
    for agent, scores in agent_scores.items():
        scores_plot = scores + scores[:1]
        color = get_agent_color(agent)

        ax.plot(
            angles,
            scores_plot,
            'o-',
            linewidth=2.5,
            markersize=9,
            label=agent,
            color=color
        )

        ax.fill(angles, scores_plot, alpha=0.15, color=color)

    # -----------------------------
    # Axis configuration
    # -----------------------------
    ax.set_ylim(0, r_max)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels, fontsize=11)

    # Ensure correct number of angle ticks
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        [s.capitalize() for s in SKILL_ORDER],
        fontsize=16,
        fontweight='bold'
    )

    ax.grid(True, alpha=0.3)

    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=12)

    plt.title(
        "Agent Skill Profiles",
        fontsize=20,
        fontweight='bold',
        pad=20
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved radar chart to {output_path}")
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    num_skills = len(SKILL_ORDER)
    angles = np.linspace(0, 2 * np.pi, num_skills, endpoint=False).tolist()
    angles += angles[:1]

    agent_scores = {}
    all_scores = []

    # Collect scores
    for data_file in data_files:
        data = load_data(data_file)
        agent = data['agent']

        if 'skill_scores' not in data:
            continue

        scores = [data['skill_scores'][s]['mean'] for s in SKILL_ORDER]
        agent_scores[agent] = scores
        all_scores.extend(scores)

    if not all_scores:
        print("No valid scores found.")
        return

    # -----------------------------
    # FIXED scaling to 65%
    # -----------------------------
    r_max = 0.65  # Fixed max radius at 65%
    ticks = np.linspace(r_max / 4, r_max, 4)
    tick_labels = [f"{t*100:.0f}%" for t in ticks]

    # -----------------------------
    # Plot agents
    # -----------------------------
    for agent, scores in agent_scores.items():
        scores_plot = scores + scores[:1]
        color = get_agent_color(agent)

        ax.plot(
            angles,
            scores_plot,
            'o-',
            linewidth=2.5,
            markersize=9,
            label=agent,
            color=color
        )

        ax.fill(angles, scores_plot, alpha=0.15, color=color)

    # -----------------------------
    # Axis configuration
    # -----------------------------
    ax.set_ylim(0, r_max)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels, fontsize=11)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        [s.capitalize() for s in SKILL_ORDER],
        fontsize=16,
        fontweight='bold'
    )

    ax.grid(True, alpha=0.3)

    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=12)

    plt.title(
        "Agent Skill Profiles",
        fontsize=20,
        fontweight='bold',
        pad=20
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    print(f"Saved radar chart to {output_path}")

    fig, ax = plt.subplots(figsize=(10,10), subplot_kw=dict(projection='polar'))

    num_skills = len(SKILL_ORDER)
    angles = np.linspace(0, 2*np.pi, num_skills, endpoint=False).tolist()
    angles += angles[:1]

    agent_scores = {}
    all_scores = []

    # Collect scores
    for data_file in data_files:
        data = load_data(data_file)
        agent = data['agent']

        if 'skill_scores' not in data:
            continue

        scores = [data['skill_scores'][s]['mean'] for s in SKILL_ORDER]
        agent_scores[agent] = scores
        all_scores.extend(scores)

    if not all_scores:
        print("No valid scores found.")
        return

    # -----------------------------
    # TRUE dynamic scaling
    # -----------------------------
    max_score = max(all_scores)

    if max_score <= 0:
        r_max = 0.1
    else:
        r_max = max_score * 1.01

        # round to nice values
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

    # ticks based on r_max
    ticks = np.linspace(r_max/4, r_max, 4)
    tick_labels = [f"{t*100:.0f}%" for t in ticks]

    # -----------------------------
    # Plot agents
    # -----------------------------
    for agent, scores in agent_scores.items():
        scores_plot = scores + scores[:1]
        color = get_agent_color(agent)

        ax.plot(
            angles,
            scores_plot,
            'o-',
            linewidth=2.5,
            markersize=9,
            label=agent,
            color=color
        )

        ax.fill(angles, scores_plot, alpha=0.15, color=color)

    # -----------------------------
    # Axis configuration
    # -----------------------------
    ax.set_ylim(0, r_max)

    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels, fontsize=11)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        [s.capitalize() for s in SKILL_ORDER],
        fontsize=16,
        fontweight='bold'
    )

    ax.grid(True, alpha=0.3)

    plt.legend(loc='upper right', bbox_to_anchor=(1.25,1.1), fontsize=12)

    plt.title(
        "Agent Skill Profiles",
        fontsize=20,
        fontweight='bold',
        pad=5
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    print(f"Saved radar chart to {output_path}")

# ============================================================================
# Plot 2: Step Budget Line Plot
# ============================================================================

def plot_step_budget(data_file, output_path):

    data = load_data(data_file)

    nb_steps = data['nb_steps']
    agents = data['agents']

    fig, ax = plt.subplots(figsize=(12, 7))

    all_scores = []

    for agent_name, scores in agents.items():
        all_scores.extend(scores)
        color = get_agent_color(agent_name)
        marker = 'o' if 'hybrid' not in agent_name.lower() else 'D'

        ax.plot(nb_steps, scores, marker=marker, linewidth=2.5, markersize=10,
               color=color, label=agent_name, alpha=0.9)

    ax.set_xlabel('Number of Steps per Episode', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Score (%)', fontsize=14, fontweight='bold')
    ax.set_title('Agent Performance vs Episode Length',
                 fontsize=16, fontweight='bold')

    auto_ylim(ax, all_scores)

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='best')

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved step budget plot to {output_path}")


# ============================================================================
# Plot 3: Efficiency Bar Chart
# ============================================================================

def plot_efficiency(data_file, output_path):

    data = load_data(data_file)

    nb_steps = data['nb_steps']
    agents = data['agents']

    fig, ax = plt.subplots(figsize=(12, 7))

    all_times = []

    for agent_name, times in agents.items():
        all_times.extend(times)

        color = get_agent_color(agent_name)

        ax.plot(
            nb_steps,
            times,
            marker='o',
            linewidth=2.5,
            markersize=8,
            label=agent_name,
            color=color,
            alpha=0.9
        )

    ax.set_xlabel('Episode Length (steps)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Wall Time (minutes)', fontsize=13, fontweight='bold')
    ax.set_title('Computational Cost vs Episode Length',
                 fontsize=15, fontweight='bold')

    auto_ylim(ax, all_times)

    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved efficiency plot to {output_path}")

# ============================================================================
# Plot 4: Skill Comparison
# ============================================================================

def plot_skill_comparison(data_files, output_path):

    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(SKILL_ORDER))
    width = 0.15
    num_agents = len(data_files)

    all_scores = []

    for i, data_file in enumerate(data_files):
        data = load_data(data_file)
        agent_name = data['agent']

        if 'skill_scores' not in data:
            continue

        scores = [data['skill_scores'][skill]['mean'] * 100
                 for skill in SKILL_ORDER]

        all_scores.extend(scores)

        offset = (i - num_agents/2 + 0.5) * width
        color = get_agent_color(agent_name)

        ax.bar(x + offset, scores, width, label=agent_name,
              color=color, alpha=0.85)

    ax.set_xlabel('Reasoning Skill', fontsize=13, fontweight='bold')
    ax.set_ylabel('Mean Score (%)', fontsize=13, fontweight='bold')
    ax.set_title('Agent Performance by Reasoning Skill',
                 fontsize=15, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in SKILL_ORDER])

    auto_ylim(ax, all_scores)

    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved skill comparison plot to {output_path}")


# ============================================================================
# CLI
# ============================================================================

def main():

    parser = argparse.ArgumentParser(
        description="Plot diagnostic evaluation results"
    )

    subparsers = parser.add_subparsers(dest='plot_type')

    radar = subparsers.add_parser('radar')
    radar.add_argument('--data', nargs='+', required=True)
    radar.add_argument('--output', required=True)

    step = subparsers.add_parser('step-budget')
    step.add_argument('--data', required=True)
    step.add_argument('--output', required=True)

    eff = subparsers.add_parser('efficiency')
    eff.add_argument('--data', required=True)
    eff.add_argument('--output', required=True)

    skill = subparsers.add_parser('skill-comparison')
    skill.add_argument('--data', nargs='+', required=True)
    skill.add_argument('--output', required=True)

    args = parser.parse_args()

    if args.plot_type == 'radar':
        plot_radar_chart(args.data, args.output)

    elif args.plot_type == 'step-budget':
        plot_step_budget(args.data, args.output)

    elif args.plot_type == 'efficiency':
        plot_efficiency(args.data, args.output)

    elif args.plot_type == 'skill-comparison':
        plot_skill_comparison(args.data, args.output)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()