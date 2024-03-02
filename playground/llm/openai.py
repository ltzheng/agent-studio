import json
import logging
from typing import Any

import backoff
from numpy.typing import NDArray
from openai import APIError, APITimeoutError, OpenAI, RateLimitError

from playground.config.config import Config
from playground.llm.base_model import BaseModel
from playground.llm.utils import encode_image

config = Config()
logger = logging.getLogger(__name__)


class OpenAIProvider(BaseModel):
    def __init__(self, **kwargs: Any) -> None:
        with open(config.api_key_path, "r") as f:
            api_keys = json.load(f)
        self.client = OpenAI(api_key=api_keys["openai"])

    # def compose_messages(
    #     self,
    #     obs: NDArray | None,
    #     trajectory: list[dict[str, Any]],
    #     system_prompt: str,
    # ) -> list[dict[str, Any]]:
    #     """
    #     Composes a message from the trajectory, system prompt and obs.
    #     """
    #     messages: list[dict[str, Any]] = []
    #     messages.append({"role": "system", "content": system_prompt})
    #     for step in trajectory:
    #         if not all(key in step for key in ["obs", "act", "res"]):
    #             raise ValueError(
    #                 "Each step in the trajectory must contain 'obs', 'act' and 'res'"
    #                 f" keys. Got {step} instead."
    #             )
    #         messages.append(
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {"type": "text", "text": "[Observation]: \n"},
    #                     {
    #                         "type": "image_url",
    #                         "image_url": {"url": encode_image(step["obs"])},
    #                     },
    #                 ],
    #             }
    #         )
    #         messages.append(
    #             {"role": "assistant", "content": f"[Action]: \n{step['act']}"}
    #         )
    #         messages.append(
    #             {
    #                 "role": "user",
    #                 "content": f"[Result]: \n{step['res']}",
    #             }
    #         )
    #     user_content = [
    #         {"type": "text", "text": "[Observation]: \n"},
    #         {
    #             "type": "image_url",
    #             "image_url": {"url": encode_image(obs)},
    #         },
    #     ]
    #     if messages[-1]["role"] == "user":
    #         messages[-1]["content"].extend(user_content)
    #     else:
    #         messages.append(
    #             {
    #                 "role": "user",
    #                 "content": user_content,
    #             }
    #         )
    #     return messages

    def generate_response(
        self, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[str, dict[str, int]]:
        """Creates a chat completion using the OpenAI API."""

        model = kwargs.get("model", config.exec_model)
        temperature = kwargs.get("temperature", config.temperature)
        max_tokens = kwargs.get("max_tokens", config.max_tokens)
        logger.info(f"Creating chat completion with model {model}...")

        @backoff.on_exception(
            backoff.constant,
            (APIError, RateLimitError, APITimeoutError),
            max_tries=config.max_retries,
            interval=10,
        )
        def _generate_response_with_retry() -> tuple[str, dict[str, int]]:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                seed=config.seed,
                max_tokens=max_tokens,
            )

            if response is None:
                logger.error("Failed to get a response from OpenAI. Try again.")

            message = response.choices[0].message.content

            info = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "system_fingerprint": response.system_fingerprint,
            }

            logger.info(f"Response received from {model}")

            return message, info

        return _generate_response_with_retry()
