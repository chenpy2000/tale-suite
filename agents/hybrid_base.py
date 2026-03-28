import tales


class HybridAgent(tales.Agent):
    """
    Base class for hybrid agents that combine multiple agent scores.

    Subclasses specify which agents to combine and weights.
    Uses act() on each component (full LLM reasoning) and weighted voting to combine.
    """

    def __init__(self, **kwargs):
        super().__init__()

        # Subclass must set these
        self.component_agents = {}  # {name: agent_instance}
        self.weights = {}  # {name: weight}

        # Store kwargs for component agents
        self.agent_kwargs = kwargs
        self._uid = "hybrid-base"

    @property
    def uid(self):
        return self._uid

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

    def act(self, obs, reward, done, info):
        """
        Call act() on each component (full LLM reasoning), then weighted voting.
        Each agent proposes an action; we sum weights per action and pick the highest.
        """
        admissible = list(info.get("admissible_commands") or [])
        if not admissible:
            return "look", {}

        votes = {}  # action -> sum of weights from agents that proposed it
        component_actions = {}

        for name, agent in self.component_agents.items():
            weight = self.weights[name]
            try:
                action, _ = agent.act(obs, reward, done, info)
                action = str(action).strip() if action else ""
                # Align to admissible (case-insensitive match)
                matched = None
                for adm in admissible:
                    if adm and action and adm.lower() == action.lower():
                        matched = adm
                        break
                if not matched and action:
                    for adm in admissible:
                        if adm and action.lower() in adm.lower() or adm.lower() in action.lower():
                            matched = adm
                            break
                if matched:
                    votes[matched] = votes.get(matched, 0.0) + weight
                    component_actions[name] = matched
            except Exception as e:
                component_actions[name] = f"(error: {e})"

        if not votes:
            # Fallback: use first component's act or first admissible
            for name, agent in self.component_agents.items():
                try:
                    action, _ = agent.act(obs, reward, done, info)
                    action = str(action).strip()
                    for adm in admissible:
                        if adm and action and adm.lower() == action.lower():
                            return adm, {"component_actions": component_actions}
                    break
                except Exception:
                    pass
            return admissible[0], {"component_actions": component_actions}

        best_action = max(votes, key=votes.get)
        stats = {"votes": votes, "component_actions": component_actions, "component_weights": dict(self.weights)}
        return best_action, stats

    def reset(self, obs, info, env_name=None):
        """Reset all component agents."""
        for agent in self.component_agents.values():
            if hasattr(agent, "reset"):
                agent.reset(obs, info, env_name)
