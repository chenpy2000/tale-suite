"""
Reusable Agent Comparison Plot
Edit the data dictionaries below to update the plot with new numbers.
"""

import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# EDIT THIS SECTION TO UPDATE DATA
# ============================================================================

# X-axis: episode lengths tested
nb_steps = [20, 50, 100, 200]

# Agent data: add/remove agents or update scores here
agents = {
    'Random Agent': {
        'scores': [10.67, 11.58, 29.92, 37.74],
        'color': '#95a5a6',  # gray
        'marker': 'o'
    },
    'React Agent': {
        'scores': [7.33, 20.67, 19.67, 20.17],
        'color': '#e74c3c',  # red
        'marker': 's'
    },
    'Memory Agent': {
        'scores': [13.33,18.74,29.91,37.74],
        'color': '#f39c12',  # orange
        'marker': '^'
    },
    'Latent Action (VQ-VAE)': {
        'scores': [22.41, 35.73, 38.32, 44.45],
        'color': '#3498db',  # blue
        'marker': 'D'
    },
    'Graph Agent': {
        'scores': [35.71, 68.67, 76.00, 80.73],
        'color': '#2ecc71',  # green
        'marker': 'v'
    }
}

# Plot settings (optional to customize)
plot_title = 'Agent Performance Comparison Across Allowed Step Budget'
xlabel = 'Step Budget'
ylabel = 'Mean Score (%)'
output_file = 'agent_comparison.png'

# ============================================================================
# PLOTTING CODE (no need to edit below unless customizing style)
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 7))

# Plot each agent
for agent_name, data in agents.items():
    ax.plot(nb_steps, data['scores'], 
            marker=data['marker'], 
            linewidth=2.5, 
            markersize=10,
            color=data['color'],
            label=agent_name,
            alpha=0.9)

# Formatting
ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
ax.set_title(plot_title, fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=11, loc='upper left', framealpha=0.9)

# Auto-set axis limits
ax.set_xlim(min(nb_steps) - 10, max(nb_steps) + 10)
max_score = max(max(data['scores']) for data in agents.values())
ax.set_ylim(0, max_score + 10)

# Add final score annotations
for agent_name, data in agents.items():
    final_score = data['scores'][-1]
    ax.annotate(f'{final_score:.1f}%', 
               (nb_steps[-1], final_score),
               textcoords="offset points",
               xytext=(10, 0),
               ha='left',
               fontsize=9,
               color=data['color'],
               fontweight='bold')

plt.tight_layout()
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_file}")
plt.show()