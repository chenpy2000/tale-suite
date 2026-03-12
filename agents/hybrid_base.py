import tales


class HybridAgent(tales.Agent):
    """
    Base class for hybrid agents that combine multiple agent scores.

    Subclasses specify which agents to combine and weights.
    """

    def __init__(self, **kwargs):
        super().__init__()

        # Subclass must set these
        self.component_agents = {}  # {name: agent_instance}
        self.weights = {}  # {name: weight}

        # Store kwargs for component agents
        self.agent_kwargs = kwargs
        self.uid = "hybrid-base"

    @property
    def params(self):
        return {"agent_type": "hybrid", "weights": dict(self.weights)}

    def initialize_components(self, agent_configs):
        """
        Initialize component agents.

        Args:
            agent_configs: list of (name, AgentClass, weight) tuples
        """
        weight_keys = {"graph_weight", "vqvae_weight", "memory_weight", "react_weight"}
        filtered_kwargs = {k: v for k, v in self.agent_kwargs.items() if k not in weight_keys}
        if "key" not in filtered_kwargs and "api_key" in filtered_kwargs:
            filtered_kwargs["key"] = filtered_kwargs["api_key"]
        for name, AgentClass, weight in agent_configs:
            try:
                agent = AgentClass(**filtered_kwargs)
                self.component_agents[name] = agent
                self.weights[name] = weight
            except Exception as e:
                print(f"Warning: Could not initialize {name} agent: {e}")
                print(f"Skipping {name} component")

        if not self.component_agents:
            raise ValueError("No component agents could be initialized")

        # Normalize weights to sum to 1.0
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}
        print(f"Initialized hybrid with components: {list(self.component_agents.keys())}")
        print(f"Normalized weights: {self.weights}")

    def score_actions(self, obs, admissible_commands, info):
        """Combine scores from all component agents."""
        combined_scores = {action: 0.0 for action in admissible_commands}

        for name, agent in self.component_agents.items():
            agent_scores = agent.score_actions(obs, admissible_commands, info)
            weight = self.weights[name]

            for action in admissible_commands:
                combined_scores[action] += weight * agent_scores.get(action, 0.0)

        return combined_scores

    def act(self, obs, reward, done, info):
        """Select action with highest combined score."""
        admissible = info.get("admissible_commands", [])
        if not admissible:
            return "look", {}

        scores = self.score_actions(obs, admissible, info)
        best_action = max(scores, key=scores.get)

        # Log component contributions for debugging
        stats = {
            "combined_scores": scores,
            "component_weights": self.weights,
        }

        return best_action, stats

    def reset(self, obs, info, env_name=None):
        """Reset all component agents."""
        for agent in self.component_agents.values():
            if hasattr(agent, "reset"):
                agent.reset(obs, info, env_name)
