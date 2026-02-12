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

SYSTEM_PROMPT = (
    "You are playing a text-based game and your goal is to finish it with the highest score. "
    "Upon reading the text observation, provide a thought process explaining your reasoning, "
    "followed by a single short phrase to interact with the game. "
    "When stuck, try using the `help` command to see what commands are available.\n\n"
    "Format your response as follows:\n"
    "Thought: <your reasoning here>\n"
    "Action: <command>"
)


class CoTAgent(tales.Agent):
    """
    Baseline Chain-of-Thought (CoT) Agent.
    Generates a thought process before taking an action.
    """

    def __init__(self, *args, **kwargs):
        self.llm = kwargs.get("llm", "gpt-4o-mini")
        self.model = llm.get_model(self.llm)
        self.token_counter = get_token_counter(self.model)
        self.allows_system_prompt = self.llm not in ["o1-mini", "o1-preview"]

        # API key need
        self.model.key = llm.get_key(
            kwargs.get("key"), kwargs["llm"], self.model.key_env_var
        ) or llm.get_key(None, self.model.needs_key, self.model.key_env_var)

        self.seed = kwargs.get("seed", 1234)
        self.rng = np.random.RandomState(self.seed)

        self.history = []
        self.context_limit = kwargs.get("context_limit")
        if self.context_limit is not None:
            assert self.context_limit > 0, "--context-limit must be greater than 0."

        self.act_temp = kwargs.get("act_temp", 0.0)
        self.conversation = kwargs.get("conversation", True)

    @property
    def uid(self):
        return (
            f"CoTAgent_{self.llm}"
            f"_s{self.seed}"
            f"_c{self.context_limit}"
            f"_t{self.act_temp}"
        )

    @property
    def params(self):
        return {
            "agent_type": "baseline-cot",
            "llm": self.llm,
            "seed": self.seed,
            "context_limit": self.context_limit,
            "act_temp": self.act_temp,
            "conversation": self.conversation,
        }

    @retry(
        retry=retry_if_exception(is_recoverable_error),
        wait=wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(100),
    )
    def _llm_call_from_conversation(self, conversation, *args, **kwargs):
        response = conversation.prompt(*args, **kwargs)
        # Force response computation if lazy
        if hasattr(response, "duration_ms"):
            response.duration_ms()
        return response

    def _llm_call_from_messages(self, messages, *args, **kwargs):
        conversation = messages2conversation(self.model, messages)
        prompt = messages[-1]["content"]
        system = messages[0]["content"] if self.allows_system_prompt else None

        return self._llm_call_from_conversation(
            conversation, prompt=prompt, system=system, *args, **kwargs
        )

    def parsing(self, response_text):
        """
        Parses the response to extract Thought and Action.
        """
        thought = ""
        action = ""

        # Case 1: Standard format "Thought: ... Action: ..."
        match = re.search(
            r"Thought:(.*?)Action:(.*)", response_text, re.DOTALL | re.IGNORECASE
        )
        if match:
            thought = match.group(1).strip()
            action = match.group(2).strip()
        else:
            # Case 2: Just Action: ...
            match = re.search(r"Action:(.*)", response_text, re.DOTALL | re.IGNORECASE)
            if match:
                action = match.group(1).strip()
                thought = response_text[: match.start()].strip()
            else:
                # Case 3: No format, assume last line is action, rest is thought
                lines = response_text.strip().split("\n")
                if len(lines) > 0:
                    action = lines[-1].strip()
                    thought = "\n".join(lines[:-1]).strip()

        # Cleanup quotes if present
        if action.startswith("`") and action.endswith("`"):
            action = action[1:-1]

        return thought, action

    def act(self, obs, reward, done, infos):
        messages = self.build_messages(f"{obs}\n> ")
        llm_kwargs = {
            "temperature": self.act_temp,
            "max_tokens": 300,  # Increased for thought process
            "seed": self.seed,
            "stream": False,
        }

        # Model specific adjustments (copied from LLMAgent)
        if self.llm in [
            "claude-3.5-haiku",
            "claude-3.5-sonnet",
            "claude-3.5-sonnet-latest",
        ]:
            llm_kwargs.pop("seed")

        if "gemini" in self.llm or "gemma" in self.llm:
            llm_kwargs.pop("seed")
            llm_kwargs["max_output_tokens"] = llm_kwargs.pop("max_tokens")

        response = self._llm_call_from_messages(messages, **llm_kwargs)
        response_text = response.text()

        thought, action = self.parsing(response_text)

        # Store the FULL response in history to maintain context of thoughts
        self.history.append((f"{obs}\n> ", f"{response_text}\n"))

        # Compute usage statistics
        stats = {
            "prompt": format_messages_to_markdown(messages),
            "response": response_text,
            "thought": thought,
            "action": action,
            "nb_tokens": self.token_counter(messages=messages, text=response_text),
        }

        return action, stats

    def build_messages(self, observation):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        limit = self.context_limit or len(self.history) + 1

        # We append history. Note: action in history includes Thoughts.
        for i, (obs, action) in enumerate(self.history[-limit:]):
            if len(self.history) >= limit and i == 0:
                obs = (
                    f"// History has been truncated to the last {limit} steps.\n...\n> "
                )

            messages.append({"role": "user", "content": obs})
            messages.append({"role": "assistant", "content": action})

        messages.append({"role": "user", "content": observation})

        messages = merge_messages(messages)

        if not self.conversation:
            content = "".join([msg["content"] for msg in messages[1:]])
            messages = messages[:1] + [{"role": "user", "content": content}]

        if not self.allows_system_prompt:
            messages.pop(0)
            messages[1]["content"] = f"{SYSTEM_PROMPT}\n\n{messages[1]['content']}"

        return messages


def build_argparser(parser=None):
    parser = parser or argparse.ArgumentParser()
    group = parser.add_argument_group("CoTAgent settings")

    group.add_argument(
        "--llm",
        default="gpt-4o-mini",
        help="LLM to be used. Default: %(default)s",
    )
    group.add_argument(
        "--seed",
        type=int,
        default=20241001,
        help="Seed for LLM. Default: %(default)s",
    )
    group.add_argument(
        "--act-temp",
        type=float,
        default=0.0,
        help="Temperature for LLM. Default: %(default)s",
    )
    group.add_argument(
        "--context-limit",
        type=int,
        help="Limit context for LLM (in conversation turns). Default: no limit.",
    )
    group.add_argument(
        "--conversation",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable conversation mode. Default: True",
    )

    return parser


register(
    name="baseline-cot",
    desc=("Baseline Chain-of-Thought agent that generates thoughts before actions."),
    klass=CoTAgent,
    add_arguments=build_argparser,
)
