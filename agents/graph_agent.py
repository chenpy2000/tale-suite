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
    EXCLUSIVE_OUTGOING = {
        "LOCATION": {"INSIDE", "ON_TOP_OF", "HELD_BY", "IN_ROOM"},
        "CONN_NORTH": {"CONNECTED_TO_NORTH"},
        "CONN_SOUTH": {"CONNECTED_TO_SOUTH"},
        "CONN_EAST": {"CONNECTED_TO_EAST"},
        "CONN_WEST": {"CONNECTED_TO_WEST"},
    }
    REL_TO_GROUP = {rel: grp for grp, rels in EXCLUSIVE_OUTGOING.items() for rel in rels}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.graph = nx.MultiDiGraph()
        self.current_room = "Unknown"
        self.turn_id = 0
        self.last_action = "None"
        
        # --- NEW: Episodic Memory for Reflexion ---
        # This list deliberately persists across episodes/resets so the agent learns over time.
        self.lessons_learned = []

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
        
        # Reset the reflection trigger so it can learn again if it dies in the new episode
        self.reflected_this_episode = False  
        
        # WE NOW WIPE THE MEMORY BETWEEN EPISODES SO IT STARTS WITH A BLANK SLATE
        self.lessons_learned = []

    # -------------------------
    # Reflexion / Learning from Mistakes
    # -------------------------
    def _reflect_on_failure(self, final_obs):
        print("\n[Reflexion System] Game Over detected. Analyzing failure to extract a safety rule...")
        
        # Grab the last 3 turns from the agent's history for context
        history_context = ""
        recent = self.history[-3:] if hasattr(self, 'history') else []
        for past_obs, past_act in recent:
            history_context += f"Observation: {past_obs.strip()}\nAction Taken: {past_act.strip()}\n\n"
            
        history_context += f"Final Game Over Observation: {final_obs.strip()}\n"

        prompt = f"""
        You are an AI playing a text adventure game. You just triggered a GAME OVER or failure.
        Read the following transcript of your final moves:
        
        [Transcript]
        {history_context}
        
        Analyze what specific action caused the failure based on the physical state (adjectives) of the objects involved. 
        Write a generalized 1-sentence behavioral rule to prevent this without using specific item names (e.g., use 'ingredients' or 'items' instead of 'cheese').
        
        Output EXACTLY one line in this format:
        RULE: <your 1-sentence generalized rule>
        """
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.triton_key}"
            }
            payload = {
                "model": "api-gpt-oss-120b", 
                "messages": [{"role": "system", "content": prompt}],
                "temperature": 0.0,
            }
            response = requests.post(self.triton_url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                text = response.json()["choices"][0]["message"]["content"]
                match = re.search(r"RULE:\s*(.+)", text, re.IGNORECASE)
                if match:
                    rule = match.group(1).strip()
                    if rule not in self.lessons_learned:
                        self.lessons_learned.append(rule)
                        print(f"\n>>> [LEARNED NEW SURVIVAL RULE]: {rule} <<<\n")
        except Exception as e:
            print(f"[Reflexion Error]: {e}")

    # -------------------------
    # Canonicalization helpers
    # -------------------------
    _ARTICLE_RE = re.compile(r"^(a|an|the)\s+", re.IGNORECASE)

    def _canon_entity(self, s: str) -> str:
        if not isinstance(s, str): return ""
        s = self._ARTICLE_RE.sub("", s.strip()).strip()
        s = re.sub(r"\s+", " ", s)
        if s.lower() in {"player", "you", "yourself", "me"}: return "Player"
        return s[0].upper() + s[1:] if s else ""

    def _canon_relation(self, r: str) -> str:
        return r.strip().upper() if isinstance(r, str) else ""

    def _extract_triplets(self, text, action):
        known_state = self._get_graph_context()
        prompt = f"""
        You are a structured Information Extraction engine for a text adventure game.
        Your task is to parse the following game observation and the last action taken, and extract the exact physical relationships of objects and locations into a list of Python tuples (Subject, Relation, Object).
        
        [Current Known State]
        {known_state}
        
        Last Action: {action}
        Observation: {text}
        
        Rules:
        1. Subjects and Objects should be properly capitalized nouns. Include descriptive adjectives when present.
        2. Relations should be uppercase keywords. Use these primarily: INSIDE, ON_TOP_OF, CONNECTED_TO, NORTH_OF, SOUTH_OF, EAST_OF, WEST_OF, HELD_BY, PART_OF.
        3. Only output a strict valid Python list of tuples. Do not output ANY other text.
        4. CRITICAL: Do NOT extract physical relationships from written text or descriptions.
        5. CRITICAL: For navigation and exits, capture the direction specifically using the current room as the subject and "Unexplored Room" as the object if the destination is unknown.
        6. MATCH KNOWN ENTITIES: If the observation mentions a generic item but a specific version exists in the [Current Known State], you MUST use the specific name!
        """
        messages = [{"role": "system", "content": prompt}, {"role": "user", "content": "Extract triplets now."}]

        try:
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.triton_key}"}
            payload = {"model": "api-gpt-oss-120b", "messages": messages, "temperature": 0.0}
            response = requests.post(self.triton_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            response_text = response.json()["choices"][0]["message"]["content"]

            clean_text = re.sub(r"```python|```", "", response_text).strip()
            triplets = ast.literal_eval(clean_text)

            match = re.search(r"-=\s*(.*?)\s*=-", text)
            new_room = match.group(1).strip() if match else None

            if new_room:
                if action and action.startswith("go ") and getattr(self, "current_room", "Unknown") not in ("Unknown", new_room):
                    direction = action.split(" ")[1].lower()
                    opposite = {"north": "south", "south": "north", "east": "west", "west": "east"}
                    
                    if direction in opposite:
                        u_norm, v_norm = self._canon_entity(self.current_room), self._canon_entity(new_room)
                        dir_rel, opp_rel = f"CONNECTED_TO_{direction.upper()}", f"CONNECTED_TO_{opposite[direction].upper()}"
                        
                        self.graph.add_edge(u_norm, v_norm, key=dir_rel, last_seen=self.turn_id)
                        self.graph.add_edge(v_norm, u_norm, key=opp_rel, last_seen=self.turn_id)
                        
                        unexp = self._canon_entity("Unexplored Room")
                        edges_to_remove = [(u, v, k) for u, v, k, d in self.graph.edges(keys=True, data=True)
                                           if (u == u_norm and v == unexp and k == dir_rel) or 
                                              (u == v_norm and v == unexp and k == opp_rel)]
                        for e in edges_to_remove:
                            self.graph.remove_edge(*e)

                self.current_room = new_room

            if isinstance(triplets, list) and all(isinstance(t, tuple) and len(t) == 3 for t in triplets):
                return triplets
            return []
        except Exception as e:
            return []

    def _remove_conflicting_edges(self, subject: str, relation: str):
        group = self.REL_TO_GROUP.get(relation)
        if not group: return
        rels = self.EXCLUSIVE_OUTGOING[group]
        for u, v, k in list(self.graph.out_edges(subject, keys=True)):
            if k in rels:
                self.graph.remove_edge(u, v, key=k)

    def _update_graph(self, triplets):
        for subject, relation, obj in triplets:
            s, r, o = self._canon_entity(subject), self._canon_relation(relation), self._canon_entity(obj)
            if not s or not r or not o: continue

            if o == self._canon_entity("Unexplored Room") and self.REL_TO_GROUP.get(r):
                group = self.REL_TO_GROUP[r]
                rels = self.EXCLUSIVE_OUTGOING[group]
                if any(k in rels and v != self._canon_entity("Unexplored Room") for u, v, k in list(self.graph.out_edges(s, keys=True))):
                    continue

            self._remove_conflicting_edges(s, r)
            self.graph.add_edge(s, o, key=r, last_seen=self.turn_id)

            if s == "Player" and r in {"INSIDE", "IN_ROOM"}:
                self.current_room = o

    def score_actions(self, obs, admissible_commands, info):
        """Graph heuristic: prefer unexplored navigation, object interactions."""
        admissible = list(admissible_commands or [])
        if not admissible:
            return {}
        scores = {a: 0.0 for a in admissible}
        lowered = (obs or "").lower()

        # Unexplored directions: highest
        unexplored_dirs = []
        if hasattr(self, "graph") and self.current_room != "Unknown":
            unexp_node = self._canon_entity("Unexplored Room")
            curr_node = self._canon_entity(self.current_room)
            for u, v, k, data in self.graph.edges(data=True, keys=True):
                if u == curr_node and v == unexp_node and k.startswith("CONNECTED_TO_"):
                    unexplored_dirs.append(f"go {k.replace('CONNECTED_TO_', '').lower()}")
        for cmd in admissible:
            if cmd in unexplored_dirs:
                scores[cmd] = 1.0

        # Object mentions in obs: examine/take
        nouns = re.findall(r"\b[a-z]{4,}\b", lowered)
        for cmd in admissible:
            if scores[cmd] > 0:
                continue
            for noun in nouns[:5]:
                if noun in cmd.lower() and any(cmd.lower().startswith(v) for v in ["examine ", "take "]):
                    scores[cmd] = 0.8
                    break

        # look, inventory: baseline
        for cmd in admissible:
            if scores[cmd] == 0 and cmd in ("look", "inventory"):
                scores[cmd] = 0.5

        if scores:
            mx = max(scores.values())
            if mx > 0:
                scores = {a: s / mx for a, s in scores.items()}
        return scores

    def _get_graph_context(self):
        if len(self.graph.nodes) == 0: return "No mapped surroundings yet."
        context = "Current Graph State (Known Map & Objects):\n"
        for u, v, k, data in self.graph.edges(keys=True, data=True):
            context += f"- {u} is {k} {v}\n"

        if self.current_room != "Unknown":
            context += f"\n[Player Location]\n- Player is in {self.current_room}\n"
            context += "\n[Navigation Assistant] (Shortest paths from your location):\n"
            try:
                curr_node = self._canon_entity(self.current_room)
                if curr_node in self.graph:
                    rooms = {u for u, v, k in self.graph.edges(keys=True) if k.startswith("CONNECTED_TO_")} | \
                            {v for u, v, k in self.graph.edges(keys=True) if k.startswith("CONNECTED_TO_")}
                    for target in rooms:
                        if target != curr_node and target != self._canon_entity("Unexplored Room") and nx.has_path(self.graph, curr_node, target):
                            path = nx.shortest_path(self.graph, curr_node, target)
                            dirs = [f"go {k.replace('CONNECTED_TO_', '').lower()}" for i in range(len(path)-1) 
                                    for u, v, k, data in self.graph.edges(data=True, keys=True) 
                                    if u == path[i] and v == path[i+1] and k.startswith("CONNECTED_TO_")]
                            if dirs: context += f"- To reach {target}: {', '.join(dirs)}\n"
            except: pass
        return context

    def build_messages(self, observation):
        messages = super().build_messages(observation)
        
        action_recommender_prompt = (
           "You are an action recommender for a text-based game agent.\n"
           "At each step your job is to do TWO things:\n"
           " 1. judge the outcome of the previous action, if there was a previous action\n"
           " 2. recommend exactly ONE next action\n\n"
           "Outcome labels for the previous action:\n"
           " - useful_scoring: the previous action directly increased score or clearly completed a scoring milestone\n"
           " - useful_non_scoring: the previous action helped progress without increasing score\n"
           " - useless: the previous action produced no useful progress\n"
           " - failed: the previous action directly caused failure, loss, death, or restart\n\n"
           "Decision rules for the next action:\n"
           " - Use the long-term goal as the overall objective.\n"
           " - Use the map summary to navigate systematically.\n"
           " - Avoid repeating failed or useless actions.\n"
           " - Prefer actions that help reach goal-relevant locations and objects.\n\n"
           "CRITICAL RULES:\n"
           "- Use the WORLD STATE to know where things are. Do NOT take something you are already holding. Do NOT put something where it already is.\n"
           "- NEVER undo a previous action. If you just put X somewhere, do NOT pick it up again unless you need it for a SPECIFIC next step.\n"
           # --- NEW: State-Awareness Logic ---
           "- STATE-AWARENESS: Pay close attention to the adjectives of objects (e.g., 'fried block of cheese'). Adjectives indicate an object's current state. NEVER apply a state-changing action (cook, fry, roast, slice, chop) to an item that already possesses that target state, or you will destroy it.\n\n"
        )

        # --- NEW: Inject Past Lessons Learned ---
        if self.lessons_learned:
            action_recommender_prompt += "[LESSONS LEARNED FROM PAST FAILURES - DO NOT REPEAT]\n"
            for lesson in set(self.lessons_learned):
                action_recommender_prompt += f"- {lesson}\n"
            action_recommender_prompt += "\n"

        action_recommender_prompt += (
           "OUTPUT FORMAT:\n"
           "You must evaluate the state of objects BEFORE acting using Chain-of-Thought.\n"
           "Output exactly in this format:\n"
           "THOUGHT: <Briefly reason about the item adjectives and your goals. Is the target state already achieved?>\n"
           "COMMAND: <the exact in-game command>\n"
        )

        messages[0]["content"] = action_recommender_prompt
        graph_state = self._get_graph_context()
        
        adm = getattr(self, "_current_admissible", [])
        if adm:
            adm_str = "\n".join([f"  {a}" for a in adm])
            valid_commands_help = f"[Current Admissible Commands]\n{adm_str}\n\nAnalyze the graph and your goal. Which Valid Command gets you closer to the objective? Output ONLY using the THOUGHT: and COMMAND: format."
        else:
            valid_commands_help = "\n[CRITICAL INSTRUCTION]\nYou MUST ONLY reply using the THOUGHT: and COMMAND: format."

        messages[-1]["content"] += f"\n\n{valid_commands_help}\n[WORLD STATE]\n{graph_state}"
        return messages

    def act(self, obs, reward, done, infos):
        self.turn_id += 1

        # --- NEW: Trigger Reflexion upon Death ---
        if ("*** You lost! ***" in obs or reward < 0) and not getattr(self, 'reflected_this_episode', False):
            self._reflect_on_failure(obs)
            self.reflected_this_episode = True

        adm = infos.get("admissible_commands") if isinstance(infos, dict) else None
        self._current_admissible = [str(a) for a in adm] if adm else []
        last_action = getattr(self, "last_action", "None")

        triplets = self._extract_triplets(obs, last_action)
        self._update_graph(triplets)

        if self._current_admissible:
            for implicit_cmd in ["inventory", "look"]:
                if implicit_cmd not in self._current_admissible:
                    self._current_admissible.append(implicit_cmd)
                    
            if hasattr(self, 'graph'):
                for u, v, k, data in self.graph.edges(data=True, keys=True):
                    if k == 'HELD_BY':
                        item = u if v.lower() == 'player' else v
                        exam_cmd = f"examine {item.lower()}"
                        if exam_cmd not in self._current_admissible:
                            self._current_admissible.append(exam_cmd)

        if self._current_admissible and hasattr(self, 'graph') and self.current_room != "Unknown":
            unexplored_dirs = []
            unexp_node, curr_node = self._canon_entity("Unexplored Room"), self._canon_entity(self.current_room)
            for u, v, k, data in self.graph.edges(data=True, keys=True):
                if u == curr_node and v == unexp_node and k.startswith("CONNECTED_TO_"):
                    unexplored_dirs.append(f"go {k.replace('CONNECTED_TO_', '').lower()}")
            if not unexplored_dirs and unexp_node in self.graph and curr_node in self.graph:
                try:
                    if nx.has_path(self.graph, curr_node, unexp_node):
                        path = nx.shortest_path(self.graph, curr_node, unexp_node)
                        if len(path) > 1:
                            for u, v, k, data in self.graph.edges(data=True, keys=True):
                                if u == curr_node and v == path[1] and k.startswith("CONNECTED_TO_"):
                                    unexplored_dirs.append(f"go {k.replace('CONNECTED_TO_', '').lower()}")
                except: pass
            if unexplored_dirs:
                pruned = [cmd for cmd in self._current_admissible if not cmd.startswith("go ") or cmd in unexplored_dirs]
                if pruned: self._current_admissible = pruned

        graph_scores = self.score_actions(obs, self._current_admissible, infos) if self._current_admissible else {}

        # --- Base LLM inference (Now returning THOUGHT and COMMAND) ---
        raw_output, stats = super().act(obs, reward, done, infos)

        # --- Parse out the actual command from the LLM's thought process ---
        cmd_match = re.search(r"COMMAND:\s*(.+)", raw_output, re.IGNORECASE)
        if cmd_match:
            action = cmd_match.group(1).strip()
        else:
            # Fallback in case LLM ignored formatting tags
            lines = [l.strip() for l in raw_output.split('\n') if l.strip()]
            action = lines[-1].replace("COMMAND:", "").replace("Action:", "").strip() if lines else raw_output.strip()
        
        # Remove stray quotes
        action = re.sub(r"['\"]", "", action).strip()

        # Fallback to admissible actions if invalid
        if self._current_admissible:
            matched = False
            for cmd in self._current_admissible:
                if action.lower() == cmd.lower():
                    action, matched = cmd, True
                    break
            if not matched:
                for cmd in self._current_admissible:
                    if action.lower() in cmd.lower() or cmd.lower() in action.lower():
                        action, matched = cmd, True
                        break
            if not matched:
                if graph_scores:
                    action = max(graph_scores, key=graph_scores.get)
                else:
                    action = str(self.rng.choice(self._current_admissible))
            
            # --- CRITICAL: Context Window Cleanup ---
            # Rewrite history so we don't pollute the LLM's context token limit with "THOUGHT: ..."
            if self.history:
                last_obs, _ = self.history[-1]
                self.history[-1] = (last_obs, f"{action}\n")
        else:
            if self.history:
                last_obs, _ = self.history[-1]
                self.history[-1] = (last_obs, f"{action}\n")

        self.last_action = str(action)
        if graph_scores:
            stats["action_scores"] = graph_scores
        return str(action), stats


def build_argparser(parser=None):
    from agents.llm import build_argparser as llm_build_argparser
    return llm_build_argparser(parser)

register(
    name="graph",
    desc="Knowledge Graph state-tracking agent using NetworkX with Reflexion CoT.",
    klass=GraphAgent,
    add_arguments=build_argparser,
)