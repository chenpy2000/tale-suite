import argparse
import os
import ast
import json
import re
import requests
from unittest.mock import patch  # (still unused, safe to delete)

import networkx as nx

from agents.llm import LLMAgent
from tales.agent import register


class GraphAgent(LLMAgent):
    # --- Relations that should be mutually exclusive per SUBJECT ---
    # If we add (X, HELD_BY, Player), we should delete prior (X, INSIDE, ...), (X, ON_TOP_OF, ...), etc.
    EXCLUSIVE_OUTGOING = {
        "LOCATION": {"INSIDE", "ON_TOP_OF", "HELD_BY", "IN_ROOM"},
        # You can extend later if you start extracting these:
        # "DOOR_STATE": {"OPEN", "CLOSED", "LOCKED", "UNLOCKED"},
        # "CUT_STATE": {"WHOLE", "SLICED", "DICED", "CHOPPED"},
        # "COOK_STATE": {"RAW", "COOKED", "BURNED"},
    }
    REL_TO_GROUP = {rel: grp for grp, rels in EXCLUSIVE_OUTGOING.items() for rel in rels}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # MultiDiGraph lets us store multiple relations between the same nodes.
        # We'll store relation as the EDGE KEY (key=relation).
        self.graph = nx.MultiDiGraph()

        self.current_room = "Unknown"
        self.turn_id = 0
        self.last_action = "None"

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
        self.turn_id = 0
        self.last_action = "None"

    # -------------------------
    # Canonicalization helpers
    # -------------------------
    _ARTICLE_RE = re.compile(r"^(a|an|the)\s+", re.IGNORECASE)

    def _canon_entity(self, s: str) -> str:
        """Normalize entity strings to reduce trivial duplicates."""
        if not isinstance(s, str):
            return ""
        s = s.strip()
        if not s:
            return ""

        # remove leading articles
        s = self._ARTICLE_RE.sub("", s).strip()

        # normalize whitespace
        s = re.sub(r"\s+", " ", s)

        # normalize Player
        if s.lower() in {"player", "you", "yourself", "me"}:
            return "Player"

        # mild title-casing; keeps 'UCSD' etc mostly fine
        # (If you hate this, delete and just return s.)
        return s[0].upper() + s[1:]

    def _canon_relation(self, r: str) -> str:
        if not isinstance(r, str):
            return ""
        return r.strip().upper()

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

            clean_text = re.sub(r"```python|```", "", response_text).strip()
            triplets = ast.literal_eval(clean_text)

            if isinstance(triplets, list) and all(isinstance(t, tuple) and len(t) == 3 for t in triplets):
                return triplets
            return []
        except Exception as e:
            print(f"[Graph Extraction Error]: {e}")
            return []

    # -------------------------
    # Fix #1: Stateful deletion
    # -------------------------
    def _remove_conflicting_edges(self, subject: str, relation: str):
        """
        Enforce mutual exclusivity groups (e.g., LOCATION).
        Removes all outgoing edges from `subject` whose relation-key is in the same group.
        """
        group = self.REL_TO_GROUP.get(relation)
        if not group:
            return

        rels = self.EXCLUSIVE_OUTGOING[group]
        # MultiDiGraph: out_edges(..., keys=True) yields (u, v, k)
        for u, v, k in list(self.graph.out_edges(subject, keys=True)):
            if k in rels:
                self.graph.remove_edge(u, v, key=k)

    def _update_graph(self, triplets):
        """Updates the NetworkX graph with the extracted triplets (with stateful deletion)."""
        for subject, relation, obj in triplets:
            s = self._canon_entity(subject)
            r = self._canon_relation(relation)
            o = self._canon_entity(obj)

            if not s or not r or not o:
                continue

            # 1) delete conflicting facts (e.g. old location) BEFORE adding new one
            self._remove_conflicting_edges(s, r)

            # 2) add the new fact; relation stored as edge KEY
            self.graph.add_edge(s, o, key=r, last_seen=self.turn_id)

            # OPTIONAL: keep current room updated if you ever extract it
            # If you extend extraction to emit ("Player","INSIDE","Kitchen"), this will track it.
            if s == "Player" and r in {"INSIDE", "IN_ROOM"}:
                self.current_room = o

    def _get_graph_context(self):
        """Stringifies the current graph state for the LLM context."""
        if len(self.graph.nodes) == 0:
            return "No mapped surroundings yet."

        context = "Current Graph State (Known Map & Objects):\n"
        # MultiDiGraph: edges(keys=True, data=True) yields u, v, k, data
        for u, v, k, data in self.graph.edges(keys=True, data=True):
            # k is the relation
            context += f"- {u} is {k} {v}\n"

        # OPTIONAL: include current room if known
        if self.current_room != "Unknown":
            context += f"\n[Player Location]\n- Player is in {self.current_room}\n"

        return context

    def build_messages(self, observation):
        messages = super().build_messages(observation)
        graph_state = self._get_graph_context()
        messages[-1]["content"] += f"\n\n[Knowledge Graph Tracker]\n{graph_state}"
        return messages

    def act(self, obs, reward, done, infos):
        # advance a logical clock so edges can store last_seen
        self.turn_id += 1

        last_action = getattr(self, "last_action", "None")
        triplets = self._extract_triplets(obs, last_action)

        self._update_graph(triplets)

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
