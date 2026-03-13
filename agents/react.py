import argparse
import re

import llm
import numpy as np
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

import tales
from tales.agent import register
from tales.token import get_token_counter
from tales.utils import (
    format_messages_to_markdown,
    is_recoverable_error,
    merge_messages,
    messages2conversation,
)

THINK_SYSTEM_PROMPT = (
    "You are playing a text-based game and your goal is to finish it with the highest score.\n"
    "For this step, think about the best next move.\n\n"
    "Rules:\n"
    "1. Use the current observation and admissible commands when available.\n"
    "2. Reason step by step, but stay focused on choosing the next best action.\n"
    "3. Do not output any action in this stage.\n"
    "4. Output exactly in this format:\n"
    "<thinking>\n"
    "your reasoning here\n"
    "</thinking>"
)

ACTION_SYSTEM_PROMPT = (
    "You are playing a text-based game and your goal is to finish it with the highest score.\n"
    "Choose the single best next command.\n\n"
    "Rules:\n"
    "1. Output exactly one game command.\n"
    "2. If admissible commands are provided, the action must be exactly one of them.\n"
    "3. Do not explain.\n"
    "4. Output exactly in this format:\n"
    "<action>\n"
    "single valid game command here\n"
    "</action>"
)


class ReactAgent(tales.Agent):
    def __init__(self, *args, **kwargs):
        self.llm = kwargs["llm"]
        self.model = llm.get_model(self.llm)
        self.token_counter = get_token_counter(self.model)
        self.allows_system_prompt = self.llm not in ["o1-mini", "o1-preview"]

        self.model.key = llm.get_key(
            kwargs.get("key"), kwargs["llm"], self.model.key_env_var
        ) or llm.get_key(None, self.model.needs_key, self.model.key_env_var)

        self.seed = kwargs["seed"]
        self.rng = np.random.RandomState(self.seed)

        self.history = []
        self.context_limit = kwargs["context_limit"]
        if self.context_limit is not None:
            assert self.context_limit > 0, "--context-limit must be greater than 0."

        self.act_temp = kwargs["act_temp"]
        self.cot_temp = kwargs["cot_temp"]
        self.cot_max_tokens = kwargs["cot_max_tokens"]
        self.conversation = kwargs["conversation"]
        self._score_cache = None

    def score_actions(self, obs, admissible_commands, info):
        """LLM rates each admissible action 0-10. Returns dict[action, score] in [0, 1]."""
        admissible = list(admissible_commands or [])
        if not admissible:
            return {}
        cache_key = (obs, tuple(sorted(admissible)))
        if self._score_cache is not None and self._score_cache[0] == cache_key:
            return self._score_cache[1]
        prompt = (
            f"Observation:\n{obs}\n\n"
            f"Rate each valid action from 0 (bad) to 10 (best). One line per action.\n"
            f"Format: action: score\n\nValid actions:\n" + "\n".join(f"- {a}" for a in admissible[:50])
        )
        if len(admissible) > 50:
            prompt += f"\n... and {len(admissible) - 50} more"
        prompt += "\n\nOutput ratings:"
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        try:
            response = self._llm_call_from_messages(
                messages, temperature=0.0, max_tokens=512, seed=self.seed, stream=False
            )
            text = response.text().strip()
        except Exception:
            self._score_cache = (cache_key, {a: 0.0 for a in admissible})
            return self._score_cache[1]
        scores = {a: 0.0 for a in admissible}
        adm_lower = {a.lower(): a for a in admissible}
        for line in text.split("\n"):
            m = re.search(r"(.+?)\s*[:=]\s*(\d+(?:\.\d+)?)", line, re.IGNORECASE)
            if m:
                act_part = m.group(1).strip().strip("'\"-*")
                try:
                    sc = float(m.group(2))
                    sc = max(0, min(10, sc))
                    for cand in admissible:
                        if cand.lower() in act_part or act_part.lower() in cand.lower():
                            scores[cand] = max(scores[cand], sc)
                            break
                    if act_part.lower() in adm_lower:
                        scores[adm_lower[act_part.lower()]] = max(
                            scores[adm_lower[act_part.lower()]], sc
                        )
                except ValueError:
                    pass
        if scores:
            mx = max(scores.values())
            if mx > 0:
                scores = {a: s / mx for a, s in scores.items()}
        self._score_cache = (cache_key, scores)
        return scores

    @property
    def uid(self):
        return (
            f"ReactAgent_{self.llm}"
            f"_s{self.seed}"
            f"_c{self.context_limit}"
            f"_t{self.act_temp}"
            f"_cotT{self.cot_temp}"
            f"_cotN{self.cot_max_tokens}"
            f"_conv{self.conversation}"
        )

    @property
    def params(self):
        return {
            "agent_type": "react",
            "llm": self.llm,
            "seed": self.seed,
            "context_limit": self.context_limit,
            "act_temp": self.act_temp,
            "cot_temp": self.cot_temp,
            "cot_max_tokens": self.cot_max_tokens,
            "conversation": self.conversation,
        }

    @retry(
        retry=retry_if_exception(is_recoverable_error),
        wait=wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(100),
    )
    def _llm_call_from_conversation(self, conversation, *args, **kwargs):
        response = conversation.prompt(*args, **kwargs)
        response.duration_ms()
        return response

    def _llm_call_from_messages(self, messages, *args, **kwargs):
        conversation = messages2conversation(self.model, messages)
        prompt = messages[-1]["content"]
        system = messages[0]["content"] if self.allows_system_prompt else None

        return self._llm_call_from_conversation(
            conversation, prompt=prompt, system=system, *args, **kwargs
        )

    def act(self, obs, reward, done, infos):
    self._score_cache = None
    admissible = list(infos.get("admissible_commands") or []) if isinstance(infos, dict) else []
    if not admissible:
        return "look", {"nb_tokens": 0, "action_scores": {}}

    question = "// Based on the above information (history), what is the best action to take? Let's think step by step.\n"
    messages = self.build_messages(obs, question, [])
    response = self._llm_call_from_messages(
        messages,
        temperature=self.cot_temp,
        max_tokens=self.cot_max_tokens,
        seed=self.seed,
        stream=False,
    )

    answer = response.text().strip()
    nb_tokens_cot = self.token_counter(messages=messages, text=response.text())

    prompt = "// Provide your chosen action on a single line while respecting the desired format.\n> "
    messages = self.build_messages(obs, prompt, [(question, f"{answer}\n")])
    response = self._llm_call_from_messages(
        messages,
        temperature=self.act_temp,
        max_tokens=100,
        seed=self.seed,
        stream=False,
    )

    action = response.text().strip()
    admissible_set = {a.lower(): a for a in admissible}

    if action.lower() not in admissible_set:
        for cmd in admissible:
            if (
                cmd.lower() == action.lower()
                or action.lower() in cmd.lower()
                or cmd.lower() in action.lower()
            ):
                action = cmd
                break
        else:
            scores = self.score_actions(obs, admissible, infos)
            if scores:
                action = max(scores, key=scores.get)
            else:
                action = str(self.rng.choice(admissible))
    else:
        action = admissible_set[action.lower()]

    self.history.append((f"{obs}\n> ", f"{action}\n"))
    nb_tokens_act = self.token_counter(messages=messages, text=response.text())
    scores = self.score_actions(obs, admissible, infos)

    stats = {
        "prompt": format_messages_to_markdown(messages),
        "response": response.text(),
        "nb_tokens": nb_tokens_cot + nb_tokens_act,
        "action_scores": scores,
    }

    return action, stats

    def _format_admissible_commands(self, admissible):
        if not admissible:
            return ""
        return "Admissible commands:\n" + "\n".join(f"- {cmd}" for cmd in admissible)

    def parse_thinking(self, text: str) -> str:
        text = text.strip()

        m = re.search(
            r"<thinking>\s*(.*?)\s*</thinking>",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if m:
            thinking = m.group(1).strip()
            if thinking:
                return thinking

        m = re.search(
            r"<thinking>\s*(.*)",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if m:
            thinking = m.group(1).strip()
            if thinking:
                return thinking

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            return "\n".join(lines)

        return "No valid reasoning produced."

    def parse_action(self, text: str, admissible=None) -> str:
        text = text.strip()

        m = re.search(
            r"<action>\s*(.*?)\s*</action>",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if m:
            action = m.group(1).strip()
            if action:
                return action

        m = re.search(
            r"<action>\s*(.*)",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if m:
            action = m.group(1).strip().splitlines()[0].strip()
            if action:
                return action

        if admissible:
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            for line in lines:
                cleaned = re.sub(r"^>+\s*", "", line).strip()
                if cleaned in admissible:
                    return cleaned

            lowered_text = text.lower()
            for cmd in admissible:
                if cmd.lower() in lowered_text:
                    return cmd

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            last = re.sub(r"^>+\s*", "", lines[-1]).strip()
            if last:
                return last

        if admissible and "help" in admissible:
            return "help"
        return "help"

    

    def build_messages(self, observation, question, qa_history, system_prompt):
        messages = [{"role": "system", "content": system_prompt}]
        limit = self.context_limit or len(self.history) + 1

        for i, (obs, action) in enumerate(self.history[-limit:]):
            if len(self.history) >= limit and i == 0:
                obs = f"// History has been truncated to the last {limit} steps.\n...\n> "

            messages.append({"role": "user", "content": obs})
            messages.append({"role": "assistant", "content": action})

        messages.append({"role": "user", "content": observation})

        for q, a in qa_history:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})

        messages.append({"role": "user", "content": question})

        messages = merge_messages(messages)

        if not self.conversation:
            content = "".join([msg["content"] for msg in messages[1:]])
            messages = messages[:1] + [{"role": "user", "content": content}]

        if not self.allows_system_prompt:
            system_prompt_text = messages.pop(0)["content"]
            messages[0]["content"] = f"{system_prompt_text}\n\n{messages[0]['content']}"

        return messages


def build_argparser(parser=None):
    parser = parser or argparse.ArgumentParser()
    group = parser.add_argument_group("LLMAgent settings")

    group.add_argument(
        "--llm",
        default="gpt-4o-mini",
        help="LLM to be used for evaluation. Default: %(default)s",
    )
    group.add_argument(
        "--seed",
        type=int,
        default=20241001,
        help="Seed for LLM (not all endpoints support this). Default: %(default)s",
    )
    group.add_argument(
        "--cot-temp",
        type=float,
        default=0.0,
        help="Temperature for LLM when doing chain-of-thoughts. Default: %(default)s",
    )
    group.add_argument(
        "--cot-max-tokens",
        type=int,
        default=1024,
        help="Maximum number of token for chain-of-thoughts. Default: %(default)s",
    )
    group.add_argument(
        "--act-temp",
        type=float,
        default=0.0,
        help="Temperature for LLM when taking actions. Default: %(default)s",
    )
    group.add_argument(
        "--context-limit",
        type=int,
        help="Limit context for LLM (in conversation turns). Default: no limit",
    )
    group.add_argument(
        "--conversation",
        required=True,
        action=argparse.BooleanOptionalAction,
        help="Enable conversation mode. Otherwise, use single prompt.",
    )

    return parser


register(
    name="react",
    desc=(
        "This agent uses a LLM to decide which action to take by following a more robust two-stage CoT/ReAct approach."
    ),
    klass=ReactAgent,
    add_arguments=build_argparser,
)