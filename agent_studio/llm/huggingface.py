import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    PaliGemmaForConditionalGeneration,
)

from agent_studio.config.config import Config
from agent_studio.llm.base_model import BaseModel
from agent_studio.utils.types import MessageList

config = Config()
logger = logging.getLogger(__name__)


class HuggingFaceProvider(BaseModel):
    name = "huggingface"

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.tokenizer = None
        self.model = None
        self.processor = None

    def _format_messages(
        self,
        raw_messages: MessageList,
    ) -> Any:
        model_message: list[dict[str, Any]] = []
        for msg in raw_messages:
            if isinstance(msg.content, str):
                model_message.append({"text": msg.content})
            elif isinstance(msg.content, Path):
                model_message.append({"image": msg.content.as_posix()})
            elif isinstance(msg.content, np.ndarray):
                model_message.append({"image": msg.content})
            else:
                assert (
                    False
                ), f"Unknown message type: {type(msg.content)}, {msg.content}"

        return model_message

    def generate_response(
        self, messages: MessageList, **kwargs
    ) -> tuple[str, dict[str, Any]]:
        """Creates a chat completion using the Gemini API."""
        model_message = self._format_messages(raw_messages=messages)

        if self.model is None:
            self.model_name = kwargs.pop("model", None)
            tokenizer_name = kwargs.pop("tokenizer", self.model_name)
            self.dtype = torch.bfloat16
            if "paligemma" in self.model_name:
                self.model = (
                    PaliGemmaForConditionalGeneration.from_pretrained(
                        self.model_name,
                        torch_dtype=self.dtype,
                    )
                    .to("cuda")
                    .eval()
                )
                self.processor = AutoProcessor.from_pretrained(self.model_name)
            elif "llava" in self.model_name:

                from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
                from llava.conversation import conv_templates
                from llava.model.builder import load_pretrained_model
                from llava.utils import disable_torch_init
                from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

                disable_torch_init()
                model_n=get_model_name_from_path(self.model_name)
                self.tokenizer,self.model,self.image_processor,self.context_len=load_pretrained_model(self.model_name,None,model_n)
                self.default_image_token=DEFAULT_IMAGE_TOKEN
                self.image_token_index=IMAGE_TOKEN_INDEX
                self.default_im_start_token=DEFAULT_IM_START_TOKEN
                self.default_im_end_token=DEFAULT_IM_END_TOKEN
                self.conv_templates=conv_templates
                self.tokenizer_image_token=tokenizer_image_token
                self.process_images=process_images

            else:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_name, trust_remote_code=True
                )
                self.model = (
                    AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=self.dtype,
                        trust_remote_code=True,
                    )
                    .to("cuda")
                    .eval()
                )
        else:
            kwargs.pop("model", None)
            kwargs.pop("tokenizer", None)

        assert self.model is not None, "Model is not loaded."

        logger.info(f"Creating chat completion with model {self.model_name}.")

        if "cogvlm" in self.model_name:
            assert (
                len(model_message) == 2
                and "text" in model_message[1]
                and "image" in model_message[0]
            ), "Expected only 1 image and 1 text for cogvlm."
            query = model_message[1]["text"]
            if isinstance(model_message[0]["image"], str):
                image = Image.open(model_message[0]["image"]).convert("RGB")
            else:  # is numpy array
                image = Image.fromarray(model_message[0]["image"]).convert("RGB")
            input_by_model = self.model.build_conversation_input_ids(
                self.tokenizer,
                query=query,
                history=[],
                images=[image],
                template_version="chat",
            )
            inputs = {
                "input_ids": input_by_model["input_ids"].unsqueeze(0).to("cuda"),
                "token_type_ids": input_by_model["token_type_ids"]
                .unsqueeze(0)
                .to("cuda"),
                "attention_mask": input_by_model["attention_mask"]
                .unsqueeze(0)
                .to("cuda"),
                "images": (
                    [[input_by_model["images"][0].to("cuda").to(self.dtype)]]
                    if image is not None
                    else None
                ),
            }
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **kwargs)
                outputs = outputs[:, inputs["input_ids"].shape[1] :]
                response = self.tokenizer.decode(outputs[0])
                response = response.split("<|end_of_text|>")[0]

        elif "cogagent" in self.model_name:
            assert (
                len(model_message) == 2
                and "text" in model_message[1]
                and "image" in model_message[0]
            ), "Expected only 1 image and 1 text for cogagent."
            query = model_message[1]["text"]
            if isinstance(model_message[0]["image"], str):
                image = Image.open(model_message[0]["image"]).convert("RGB")
            else:  # is numpy array
                image = Image.fromarray(model_message[0]["image"]).convert("RGB")
            input_by_model = self.model.build_conversation_input_ids(
                self.tokenizer,
                query=query,
                history=[],
                images=[image],
                # template_version="chat",
            )
            inputs = {
                "input_ids": input_by_model["input_ids"].unsqueeze(0).to("cuda"),
                "token_type_ids": input_by_model["token_type_ids"]
                .unsqueeze(0)
                .to("cuda"),
                "attention_mask": input_by_model["attention_mask"]
                .unsqueeze(0)
                .to("cuda"),
                "images": (
                    [[input_by_model["images"][0].to("cuda").to(self.dtype)]]
                    if image is not None
                    else None
                ),
            }
            if "cross_images" in input_by_model and input_by_model["cross_images"]:
                inputs["cross_images"] = [
                    [input_by_model["cross_images"][0].to("cuda").to(self.dtype)]
                ]
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **kwargs)
                outputs = outputs[:, inputs["input_ids"].shape[1] :]
                response = self.tokenizer.decode(outputs[0])
                response = response.split("</s>")[0]

        elif "paligemma" in self.model_name:
            assert (
                len(model_message) == 2
                and "text" in model_message[1]
                and "image" in model_message[0]
            ), "Expected only 1 image and 1 text for paligemma."
            query = model_message[1]["text"]
            if isinstance(model_message[0]["image"], str):
                image = Image.open(model_message[0]["image"]).convert("RGB")
            else:  # is numpy array
                image = Image.fromarray(model_message[0]["image"]).convert("RGB")
            inputs = self.processor(query, image, return_tensors="pt").to("cuda")
            output = self.model.generate(**inputs, **kwargs)
            response = self.processor.decode(output[0], skip_special_tokens=True)[
                len(query) :
            ]

        elif "Qwen" in self.model_name or "SeeClick" in self.model_name:
            query = self.tokenizer.from_list_format(model_message)
            response, _ = self.model.chat(
                self.tokenizer, query=query, history=None, **kwargs
            )

        elif "MiniCPM" in self.model_name:
            assert (
                len(model_message) == 2
                and "text" in model_message[1]
                and "image" in model_message[0]
            ), "Expected only 1 image and 1 text for paligemma."
            if isinstance(model_message[0]["image"], str):
                image = Image.open(model_message[0]["image"]).convert("RGB")
            else:  # is numpy array
                image = Image.fromarray(model_message[0]["image"]).convert("RGB")
            response = self.model.chat(
                image=image,
                msgs=[{"role": "user", "content": model_message[1]["text"]}],
                tokenizer=self.tokenizer,
                **kwargs,
            )

        elif "llava" in self.model_name:

            assert (
                len(model_message) == 2
                and "text" in model_message[1]
                and "image" in model_message[0]
            ), "Expected only 1 image and 1 text for LLaVA."

            query = model_message[1]["text"]
            image = model_message[0]['image']
            if isinstance(model_message[0]["image"], str):
                image = Image.open(model_message[0]["image"]).convert("RGB")
            else:  # is numpy array
                image = Image.fromarray(model_message[0]["image"]).convert("RGB")

            if self.model.config.mm_use_im_start_end:
                query = self.default_im_start_token + self.default_image_token + self.default_im_end_token + '\n' + query
            else:
                query = self.default_image_token + '\n' + query

            conv = self.conv_templates['llava_v1'].copy()
            conv.append_message(conv.roles[0],query)
            conv.append_message(conv.roles[1],None)
            prompt = conv.get_prompt()
            input_ids = self.tokenizer_image_token(prompt,self.tokenizer,self.image_token_index,return_tensors='pt').unsqueeze(0).cuda()
            image_tensors = self.process_images([image],self.image_processor,self.model.config)[0]

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensors.unsqueeze(0).half().cuda(),
                    image_sizes=[image.size],
                    do_sample=True,
                    max_new_tokens=1024,
                    use_cache=True)

            response = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        else:
            raise ValueError(f"Model {self.model_name} is not supported.")

        logger.info(f"\nReceived response:\n{response}")

        return response, {"total_tokens": None}
