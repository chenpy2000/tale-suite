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
        4. CRITICAL: Do NOT extract physical relationships from written text, recipes, or descriptions. If a Cookbook says "Ingredients: block of cheese", the cheese is NOT physically `INSIDE` the cookbook! It is just text. Ignore recipes when extracting physical state.
        
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
        
        adm = getattr(self, "_current_admissible", [])
        if adm:
            adm_str = "\n".join([f"  {a}" for a in adm])
            valid_commands_help = f"[Current Admissible Commands]\n{adm_str}\n\n[CRITICAL INSTRUCTION]\nYou MUST ONLY reply with a single exact command from the [Current Admissible Commands] list above.\nDo NOT use synonyms. Do NOT invent new verbs."
        else:
            # Inject the valid commands list to prevent hallucinating actions
            valid_commands_help = """
[Available Commands Reference]
  look:                describe the current room
  goal:                print the goal of this game
  inventory:           print player's inventory
  go <dir>:            move the player north, east, south or west
  examine ...:         examine something more closely
  eat ...:             eat edible food
  open ...:            open a door or a container
  close ...:           close a door or a container
  drop ...:            drop an object on the floor
  take ...:            take an object that is on the floor
  put ... on ...:      place an object on a supporter
  take ... from ...:   take an object from a container or a supporter
  insert ... into ...: place an object into a container
  lock ... with ...:   lock a door or a container with a key
  unlock ... with ...: unlock a door or a container with a key
  cook ... with ...:   cook cookable food with something providing heat
  slice ... with ...:  slice cuttable food with something sharp
  chop ... with ...:   chop cuttable food with something sharp
  dice ... with ...:   dice cuttable food with something sharp
  prepare meal:        combine ingredients from inventory into a meal

[CRITICAL INSTRUCTION]
You MUST ONLY reply with a single command chosen EXACTLY from the formats in the [Available Commands Reference] above.
Do NOT use synonyms. Do NOT invent new verbs like "use", "make", or "pick up".
If you want to cook a carrot, you MUST output: `cook carrot with oven` or `cook carrot with stove`. NOT `use oven to roast carrot`.
"""
        
        # Override the generic LLMAgent system prompt to be strictly compliant
        messages[0]["content"] = (
            "You are an expert player of a text-based cooking adventure game. Your goal is to finish the recipe with the highest score.\n"
            "You must meticulously read the [Knowledge Graph Tracker] to understand your environment.\n"
            "You must strictly output ONLY ONE valid command per turn."
        )

        messages[-1]["content"] += f"\n\n{valid_commands_help}\n[Knowledge Graph Tracker]\n{graph_state}"
        return messages

    def act(self, obs, reward, done, infos):
        # advance a logical clock so edges can store last_seen
        self.turn_id += 1

        adm = infos.get("admissible_commands") if isinstance(infos, dict) else None
        self._current_admissible = [str(a) for a in adm] if adm else []

        last_action = getattr(self, "last_action", "None")
        triplets = self._extract_triplets(obs, last_action)

        self._update_graph(triplets)

        action, stats = super().act(obs, reward, done, infos)

        # Fallback to admissible actions if invalid
        if self._current_admissible:
            matched = False
            for cmd in self._current_admissible:
                if action.lower() in cmd.lower() or cmd.lower() in action.lower():
                    action = cmd
                    matched = True
                    break
            
            if not matched:
                action = self.rng.choice(self._current_admissible)
            
            # Overwrite the action in the history so we don't pollute context
            if self.history:
                last_obs, _ = self.history[-1]
                self.history[-1] = (last_obs, f"{action}\n")

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
