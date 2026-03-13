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
        admissible = infos.get("admissible_commands", None)
        admissible_text = self._format_admissible_commands(admissible)

        state_text = obs
        if admissible_text:
            state_text = f"{obs}\n\n{admissible_text}"

        # -------------------------
        # Stage 1: reasoning
        # -------------------------
        think_question = (
            "Think about the best next action.\n"
            "Do not output the final command yet.\n"
            "Output exactly:\n"
            "<thinking>...</thinking>"
        )

        think_messages = self.build_messages(
            observation=state_text,
            question=think_question,
            qa_history=[],
            system_prompt=THINK_SYSTEM_PROMPT,
        )

        think_response = self._llm_call_from_messages(
            think_messages,
            temperature=self.cot_temp,
            max_tokens=self.cot_max_tokens,
            seed=self.seed,
            stream=False,
        )

        think_raw = think_response.text().strip()
        # print("\n=== STAGE 1 RAW ===", flush=True)
        # print(think_raw, flush=True)

        reasoning = self.parse_thinking(think_raw)
        # print("\n=== STAGE 1 PARSED ===", flush=True)
        # print(reasoning, flush=True)

        nb_tokens_cot = self.token_counter(messages=think_messages, text=think_response.text())

        # -------------------------
        # Stage 2: action selection
        # -------------------------
        action_question = (
            "Reasoning from the previous step:\n"
            f"{reasoning}\n\n"
            "Now provide the chosen action.\n"
            "Output exactly:\n"
            "<action>\n"
            "single valid game command here\n"
            "</action>"
        )

        action_messages = self.build_messages(
            observation=state_text,
            question=action_question,
            qa_history=[],
            system_prompt=ACTION_SYSTEM_PROMPT,
        )

        action_response = self._llm_call_from_messages(
            action_messages,
            temperature=self.act_temp,
            max_tokens=100,
            seed=self.seed,
            stream=False,
        )

        action_raw = action_response.text().strip()
        # print("\n=== STAGE 2 RAW ===", flush=True)
        # print(action_raw, flush=True)

        action = self.parse_action(action_raw, admissible)
        # print("\n=== STAGE 2 PARSED ===", flush=True)
        # print(action, flush=True)

        if not action:
            action = "help"

        self.history.append((f"{obs}\n> ", f"{action}\n"))

        nb_tokens_act = self.token_counter(messages=action_messages, text=action_response.text())
        stats = {
            "thinking_prompt": format_messages_to_markdown(think_messages),
            "thinking_response": think_response.text(),
            "action_prompt": format_messages_to_markdown(action_messages),
            "action_response": action_raw,
            "nb_tokens": nb_tokens_cot + nb_tokens_act,
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