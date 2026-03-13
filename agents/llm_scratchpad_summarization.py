import argparse

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
    "You are playing a text-based game and your goal is to finish it with the highest score and record high-level principles to maximize scores."
    " Upon reading the text observation, provide an answer with two components separated by '|'"
    " First part is a *single* short phrase to interact with the game, e.g. `get lamp` (without the backticks)."
    " Second part is a scratchpad used to store principles that you will be seeing it every time, so you want to polish it over iterations."
    " When stuck, try using the `help` command to see what commands are available."
)

SCRATCHPAD = "[Scratchpad: (empty)]"
# For gpt-4o-mini:
# SCRATCHPAD = "[Scratchpad: \nAlways read any documents or items found for clues or information. Collect useful items to aid in your progress. Avoid dangerous areas without preparation.]"
# SCRATCHPAD = "[Scratchpad: \nIf stuck in a closet, repeatedly attempt to go south to exit. If unable to exit, explore other rooms or check for items that might help. If you find yourself in a loop, try different commands or explore other areas.]"
# SCRATCHPAD = "[Scratchpad: \nIf you enter a closet, try to find a way out immediately.]"
# SCRATCHPAD = "[Scratchpad: \nAlways pick up items to increase your score. Avoid dangerous areas unless prepared. Use items wisely to navigate challenges. If you encounter a dead end, explore other options. If you have an item, remember you can't pick it up again.]"
# SCRATCHPAD = "[Scratchpad: \nIf in a closet, look for other exits or return to the last known area.]"
# SCRATCHPAD = "[Scratchpad: \nAlways check for items that may seem unimportant, as they could be crucial to solving the case.]"
# SCRATCHPAD = "[Scratchpad: \nIf stuck in a closet, try to explore other rooms or directions instead of repeating the same command.]"

# For gpt-4o:
# SCRATCHPAD = "[Scratchpad: \n1. Collect items that may be useful later. 2. Explore all accessible areas for clues and items. 3. Use items creatively to solve problems or progress. 4. Pay attention to character interactions for hints. 5. Keep track of directions and locations to avoid getting lost. 6. Use the 'look' command to gather more information about your surroundings. 7. Use 'help' to understand available commands. 8. Restart if necessary to try]"
# SCRATCHPAD = "[Scratchpad: \n1. Always collect items that may be useful later. 2. Use your badge to gain access to restricted areas. 3. Avoid dangerous situations unless prepared. 4. Explore all directions in new areas to gather information. 5. Prioritize visiting crime scenes for clues. 6. Keep track of your score to measure progress. 7. Restart if you die to try different strategies. 8. Read all documents for potential hints. 9. Use weapons]"
# SCRATCHPAD = "[Scratchpad: \n1. Always collect items that may be useful later. 2. Use your badge to gain access to restricted areas. 3. Avoid dangerous situations unless prepared. 4. Explore all directions in new areas to gather information. 5. Prioritize visiting crime scenes for clues. 6. Keep track of your score to measure progress. 7. Restart if you die to try different strategies. 8. Read all documents for potential hints. 9. Use weapons]"
# SCRATCHPAD = "[Scratchpad: \n1. Always collect items that may be useful later. 2. Use your badge to gain access to restricted areas. 3. Avoid dangerous situations unless prepared. 4. Explore all directions in new areas to gather information. 5. Prioritize visiting crime scenes for clues. 6. Keep track of your score to measure progress. 7. Restart if you die to try different strategies. 8. Read all documents for potential hints. 9. Use weapons]"

# For gpt-4.5-preview:

class LLMAgent(tales.Agent):

    def __init__(self, *args, **kwargs):
        self.llm = kwargs["llm"]
        self.model = llm.get_model(self.llm)
        self.token_counter = get_token_counter(self.model)
        self.allows_system_prompt = self.llm not in ["o1-mini", "o1-preview"]

        # Provide the API key, if one is needed and has been provided
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
        self.conversation = kwargs["conversation"]

        self.scratchpad = SCRATCHPAD

    @property
    def uid(self):
        return (
            f"LLMAgent_{self.llm}"
            f"_s{self.seed}"
            f"_c{self.context_limit}"
            f"_t{self.act_temp}"
            f"_conv{self.conversation}"
        )

    @property
    def params(self):
        return {
            "agent_type": "zero-shot",
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
        response.duration_ms()  # Forces the response to be computed.
        return response

    def _llm_call_from_messages(self, messages, *args, **kwargs):
        conversation = messages2conversation(self.model, messages)
        prompt = messages[-1]["content"]
        system = messages[0]["content"] if self.allows_system_prompt else None

        return self._llm_call_from_conversation(
            conversation, prompt=prompt, system=system, *args, **kwargs
        )

    def act(self, obs, reward, done, infos):
        messages = self.build_messages(f"{obs}\n> ")
        llm_kwargs = {
            "temperature": self.act_temp,
            "max_tokens": 100,  # Text actions are short phrases.
            "seed": self.seed,
            "stream": False,
        }
        if self.llm in [
            "claude-3.5-haiku",
            "claude-3.5-sonnet",
            "claude-3.5-sonnet-latest",
            "claude-3.7-sonnet",
        ]:
            # For these models, we cannot set the seed.
            llm_kwargs.pop("seed")

        if "gemini" in self.llm or "gemma" in self.llm:
            # For these models, we cannot set the seed and max_tokens has a different name.
            llm_kwargs.pop("seed")
            llm_kwargs["max_output_tokens"] = llm_kwargs.pop("max_tokens")

        response = self._llm_call_from_messages(messages, **llm_kwargs)
        raw_response = response.text().strip()

        if not raw_response or "|" not in raw_response:
            action = ''
            scratchpad = ''
        else:
            action_part, scratchpad_part = raw_response.split("|", 1)
            action = action_part.strip()
            scratchpad = scratchpad_part.strip()

        self.scratchpad = f"[Scratchpad: \n{scratchpad}]"

        self.history.append((f"{obs}\n> ", f"{action}\n"))

        # Compute usage statistics
        stats = {
            "prompt": format_messages_to_markdown(messages),
            "response": response.text(),
            "action": action,
            "scratchpad": self.scratchpad,
            "nb_tokens_prompt": self.token_counter(messages=messages),
            "nb_tokens_response": self.token_counter(text=response.text()),
        }

        stats["nb_tokens"] = stats["nb_tokens_prompt"] + stats["nb_tokens_response"]

        return action, stats

    def build_messages(self, observation):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.append({"role": "user", "content": self.scratchpad})
        limit = self.context_limit or len(self.history) + 1

        for i, (obs, action) in enumerate(self.history[-limit:]):
            if len(self.history) >= limit and i == 0:
                # Add the current observation.
                obs = (
                    f"// History has been truncated to the last {limit} steps.\n...\n> "
                )

            messages.append({"role": "user", "content": obs})
            messages.append({"role": "assistant", "content": action})

        messages.append({"role": "user", "content": observation})

        # Just in case, let's avoid having multiple messages from the same role.
        messages = merge_messages(messages)

        if not self.conversation:
            # Merge all messages into a single message except for the system.
            content = "".join([msg["content"] for msg in messages[1:]])
            messages = messages[:1] + [{"role": "user", "content": content}]

        if not self.allows_system_prompt:
            # Make sure the system prompt is added to the following message.
            messages.pop(0)
            messages[1]["content"] = f"{SYSTEM_PROMPT}\n\n{messages[1]['content']}"

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
        "--act-temp",
        type=float,
        default=0.0,
        help="Temperature for LLM when taking actions. Default: %(default)s",
    )
    group.add_argument(
        "--context-limit",
        type=int,
        help="Limit context for LLM (in conversation turns). Default: no limit.",
    )
    group.add_argument(
        "--conversation",
        required=True,
        action=argparse.BooleanOptionalAction,
        help="Enable conversation mode. Otherwise, use single prompt.",
    )

    return parser


register(
    name="zero-shot",
    desc=(
        "This agent uses a LLM to decide which action to take in a zero-shot manner."
    ),
    klass=LLMAgent,
    add_arguments=build_argparser,
)
