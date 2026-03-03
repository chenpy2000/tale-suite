import argparse
import os
import ast
import json
import re
import requests
from unittest.mock import patch

import networkx as nx

from agents.llm import LLMAgent
from tales.agent import register


class GraphAgent(LLMAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph = nx.DiGraph()
        self.current_room = "Unknown"
        
        # We will use raw HTTP requests for graph extraction to bypass llm key conflicts
        self.triton_url = "https://tritonai-api.ucsd.edu/v1/chat/completions"
        self.triton_key = os.environ.get("TRITON_API_KEY", "")

    @property
    def uid(self):
        return super().uid + "_graph"

    @property
    def params(self):
        p = super().params
        p["agent_type"] = "graph"
        return p

    def reset(self, obs, info, env_name):
        super().reset(obs, info, env_name)
        self.graph.clear()
        self.current_room = "Unknown"
        self.last_action = "None"

    def _extract_triplets(self, text, action):
        """
        Uses the base LLM to parse the observation text into (Subject, Relation, Object) triplets.
        """
        prompt = f"""
        You are a structured Information Extraction engine for a text adventure game.
        Your task is to parse the following game observation and the last action taken, and extract the exact physical relationships of objects and locations into a list of Python tuples (Subject, Relation, Object).
        
        Last Action: {action}
        Observation: {text}
        
        Rules:
        1. Subjects and Objects should be properly capitalized nouns (e.g., "Apple", "Kitchen", "Red Door").
        2. Relations should be uppercase keywords. Use these primarily: INSIDE, ON_TOP_OF, CONNECTED_TO, NORTH_OF, SOUTH_OF, EAST_OF, WEST_OF, HELD_BY, PART_OF.
        3. Only output a strict valid Python list of tuples. Do not output ANY other text or explanations.
        
        Example Output:
        [("Apple", "INSIDE", "Refrigerator"), ("Kitchen", "WEST_OF", "Living Room"), ("Knife", "HELD_BY", "Player")]
        """
        messages = [{"role": "system", "content": prompt}, {"role": "user", "content": "Extract triplets now."}]

        try:
            # We call Triton natively to bypass LiteLLM's aggressive API key resolution from os.environ
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.triton_key}"
            }
            payload = {
                "model": "api-gpt-oss-120b",
                "messages": messages,
                "temperature": 0.0,
            }
            response = requests.post(self.triton_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            response_data = response.json()
            response_text = response_data["choices"][0]["message"]["content"]
            
            # Clean up potential markdown formatting from the response
            clean_text = re.sub(r"```python|```", "", response_text).strip()
            triplets = ast.literal_eval(clean_text)

            if isinstance(triplets, list) and all(
                isinstance(t, tuple) and len(t) == 3 for t in triplets
            ):
                return triplets
            return []
        except Exception as e:
            # If the LLM output is malformed, fail gracefully
            print(f"[Graph Extraction Error]: {e}")
            return []

    def _update_graph(self, triplets):
        """Updates the NetworkX graph with the extracted triplets."""
        for subject, relation, obj in triplets:
            self.graph.add_edge(subject, obj, relation=relation)

    def _get_graph_context(self):
        """Stringifies the current graph state for the LLM context."""
        if len(self.graph.nodes) == 0:
            return "No mapped surroundings yet."

        context = "Current Graph State (Known Map & Objects):\n"
        for u, v, data in self.graph.edges(data=True):
            context += f"- {u} is {data.get('relation', 'RELATED_TO')} {v}\n"
        return context

    def build_messages(self, observation):
        messages = super().build_messages(observation)

        # Inject the current graph state into the most recent prompt
        graph_state = self._get_graph_context()
        messages[-1]["content"] += f"\n\n[Knowledge Graph Tracker]\n{graph_state}"

        return messages

    def act(self, obs, reward, done, infos):
        # 1. Parse current observation into triplets
        last_action = getattr(self, "last_action", "None")
        triplets = self._extract_triplets(obs, last_action)

        # 2. Update the internal Graph DB
        self._update_graph(triplets)

        # 3. Proceed with standard LLM reasoning augmented by the Graph Context
        action, stats = super().act(obs, reward, done, infos)
        
        self.last_action = action

        return action, stats


def build_argparser(parser=None):
    from agents.llm import build_argparser as llm_build_argparser

    parser = llm_build_argparser(parser)
    return parser


register(
    name="graph",
    desc="Knowledge Graph state-tracking agent using NetworkX.",
    klass=GraphAgent,
    add_arguments=build_argparser,
)
