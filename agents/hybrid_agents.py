"""Concrete hybrid agents combining multiple agent scores."""

from agents.hybrid_base import HybridAgent
from agents.graph_agent import GraphAgent
from agents.llm_vqvae_agent import LLMVQVAEAgent
from agents.memory_agent import MemoryAgent
from agents.react import ReactAgent

from tales.agent import register


# ============================================================================
# Hybrid 1: Graph + VQ-VAE (Spatial + Inductive)
# ============================================================================


class GraphVQVAEHybrid(HybridAgent):
    """Combines spatial reasoning (Graph) with learned patterns (VQ-VAE)."""

    def __init__(self, graph_weight=0.6, vqvae_weight=0.4, **kwargs):
        super().__init__(**kwargs)
        self.graph_weight = float(kwargs.get("graph_weight", graph_weight))
        self.vqvae_weight = float(kwargs.get("vqvae_weight", vqvae_weight))
        self.initialize_components([
            ("graph", GraphAgent, self.graph_weight),
            ("vqvae", LLMVQVAEAgent, self.vqvae_weight),
        ])
        self._uid = f"graph-vqvae-g{self.graph_weight}-v{self.vqvae_weight}"


# ============================================================================
# Hybrid 2: Memory + ReAct (Grounded + Deductive)
# ============================================================================


class MemoryReActHybrid(HybridAgent):
    """Combines episodic memory with reasoning."""

    def __init__(self, memory_weight=0.5, react_weight=0.5, **kwargs):
        super().__init__(**kwargs)
        self.memory_weight = float(kwargs.get("memory_weight", memory_weight))
        self.react_weight = float(kwargs.get("react_weight", react_weight))
        self.initialize_components([
            ("memory", MemoryAgent, self.memory_weight),
            ("react", ReactAgent, self.react_weight),
        ])
        self._uid = f"memory-react-m{self.memory_weight}-r{self.react_weight}"


# ============================================================================
# Hybrid 3: Full Hybrid (All Skills)
# ============================================================================


class FullHybrid(HybridAgent):
    """Combines all memory types for complete skill coverage."""

    def __init__(
        self,
        graph_weight=0.3,
        vqvae_weight=0.3,
        memory_weight=0.2,
        react_weight=0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.graph_weight = float(kwargs.get("graph_weight", graph_weight))
        self.vqvae_weight = float(kwargs.get("vqvae_weight", vqvae_weight))
        self.memory_weight = float(kwargs.get("memory_weight", memory_weight))
        self.react_weight = float(kwargs.get("react_weight", react_weight))
        self.initialize_components([
            ("graph", GraphAgent, self.graph_weight),
            ("vqvae", LLMVQVAEAgent, self.vqvae_weight),
            ("memory", MemoryAgent, self.memory_weight),
            ("react", ReactAgent, self.react_weight),
        ])
        self._uid = f"full-hybrid-g{self.graph_weight}-v{self.vqvae_weight}-m{self.memory_weight}-r{self.react_weight}"


# ============================================================================
# Registration with TALES
# ============================================================================


def add_hybrid_args(parser):
    """Add hybrid-specific arguments to parser."""
    g = parser.add_argument_group("Hybrid agent")
    g.add_argument("--graph-weight", type=float, default=0.6, help="Weight for graph agent")
    g.add_argument("--vqvae-weight", type=float, default=0.4, help="Weight for VQ-VAE agent")
    g.add_argument("--memory-weight", type=float, default=0.5, help="Weight for memory agent")
    g.add_argument("--react-weight", type=float, default=0.5, help="Weight for ReAct agent")
    # Component parsers add their required args (api-key, vqvae-checkpoint, etc.)
    from agents.llm_vqvae_agent import build_argparser as vqvae_parser
    from agents.graph_agent import build_argparser as graph_parser
    from agents.memory_agent import build_argparser as mem_parser
    from agents.react import build_argparser as react_parser

    # Avoid argparse conflicts when multiple components define the same flags (e.g., --seed).
    old_conflict_handler = parser.conflict_handler
    parser.conflict_handler = "resolve"
    try:
        vqvae_parser(parser)
        graph_parser(parser)
        mem_parser(parser)
        react_parser(parser)
    finally:
        parser.conflict_handler = old_conflict_handler
    return parser


register("graph-vqvae", "Graph + VQ-VAE hybrid (spatial + inductive)", GraphVQVAEHybrid, add_hybrid_args)
register("memory-react", "Memory + ReAct hybrid (grounded + deductive)", MemoryReActHybrid, add_hybrid_args)
register("full-hybrid", "Full hybrid (all skills)", FullHybrid, add_hybrid_args)
