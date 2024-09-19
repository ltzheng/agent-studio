import dataclasses
import json
import os
import uuid
from typing import Any
import re

import cv2
import numpy as np
from PIL import Image

from agent_studio.agent.base_agent import TrajectoryInfo
from agent_studio.llm.utils import decode_image
from agent_studio.utils.types import TaskConfig


def read_jsonl(file_path: str, start_idx: int = 0, end_idx: int | None = None) -> list:
    """Reads lines from a .jsonl file between start_idx and end_idx.

    Args:
        file_path (str): Path to the .jsonl file
        start_idx (int, optional): The starting index of lines to read
        end_idx (int | None, optional): The ending index of lines to read

    Returns:
        list[dict]: A list of dictionaries, each dictionary is a line from
            the .jsonl file
    """
    if end_idx is not None and start_idx > end_idx:
        raise ValueError("start_idx must be less or equal to end_idx")

    data = []
    with open(file_path, "r") as file:
        for i, line in enumerate(file):
            if end_idx is not None and i >= end_idx:
                break
            if i >= start_idx:
                data.append(json.loads(line))

    return data


def add_jsonl(data: list, file_path: str, mode="a"):
    """Adds a list of dictionaries to a .jsonl file.

    Args:
        data (list[dict]): A list of json objects to add to the file
        file_path (str): Path to the .jsonl file
    """
    with open(file_path, mode) as file:
        for item in data:
            json_str = json.dumps(item)
            file.write(json_str + "\n")


def read_json(
    file_path: str, start_idx: int = 0, end_idx: int | None = None
) -> dict | list:
    """Reads a dictionary from a .json file.

    Args:
        file_path (str): Path to the .json file
    """
    with open(file_path, "r") as file:
        if end_idx is None:
            data = json.load(file)[start_idx:]
        else:
            data = json.load(file)[start_idx:end_idx]
    return data


def apply_env_vars(task_config: TaskConfig, env_vars: dict) -> TaskConfig:
    def replace_placeholders(text, env_vars):
        # Regex to find patterns like ${VAR_NAME}
        pattern = re.compile(r"\$\{(\w+)\}")

        def replacer(match):
            var_name = match.group(1)
            if var_name in env_vars:
                return env_vars[var_name]
            else:
                raise ValueError(
                    f"Variable {var_name} not found in environment variables")
        return pattern.sub(replacer, text)

    def traverse_and_replace(obj, env_vars):
        if isinstance(obj, dict):
            return {k: traverse_and_replace(v, env_vars) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [traverse_and_replace(item, env_vars) for item in obj]
        elif isinstance(obj, str):
            return replace_placeholders(obj, env_vars)
        else:
            return obj

    json_dict = task_config.model_dump()
    replaced_json = traverse_and_replace(json_dict, env_vars)
    return TaskConfig.model_validate(replaced_json)


def read_task_jsons(path: str) -> list[TaskConfig]:
    """
    Read task configs from folder or file
    """
    task_configs: list[TaskConfig] = []
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(".json"):
                    with open(os.path.join(root, file), "r") as f:
                        task_configs.append(TaskConfig.model_validate(json.load(f)))
    else:
        with open(path, "r") as f:
            task_configs.append(TaskConfig.model_validate(json.load(f)))
    return task_configs


def add_json(data: dict, file_path: str, mode="a"):
    """Adds a dictionary to a .json file.

    Args:
        data (dict): The dictionary to add to the file
        file_path (str): Path to the .json file
    """
    with open(file_path, mode) as file:
        json.dump(data, file)
        file.write("\n")


def format_json(data: dict, indent=4, sort_keys=False):
    """Prints a dictionary in a formatted way.

    Args:
        data (dict): The dictionary to print
    """
    return json.dumps(data, indent=indent, sort_keys=sort_keys)


def export_trajectories(
    task_config: TaskConfig,
    trajectory: TrajectoryInfo,
    record_path: str,
    score: float | None,
    feedback: str | None,
    token_count: int | None,
    video_meta: dict | None = None,
    jsonl_name: str = "results.jsonl",
) -> None:
    """Exports the trajectory data to a .jsonl file."""
    if not os.path.exists(record_path):
        os.makedirs(record_path, exist_ok=True)
    media_path = os.path.join(record_path, task_config.task_id)
    results = {
        "video": video_meta,
        "task_id": task_config.task_id,
        "instruction": task_config.instruction,
        "trajectory": trajectory,
        "token_count": token_count,
    }
    if score is not None:
        results["score"] = score
    if feedback is not None:
        results["feedback"] = feedback
    parse_and_save_objects(results, media_path)
    add_jsonl(
        data=[results],
        file_path=os.path.join(record_path, jsonl_name),
    )


def save_image_or_array(obj: Any, folder_path: str) -> str:
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)

    # A unique identifier for the filename.
    unique_filename = str(uuid.uuid4())
    if isinstance(obj, Image.Image):
        file_path = os.path.join(folder_path, f"{unique_filename}.png")
        obj.save(file_path)
    elif isinstance(obj, np.ndarray):
        file_path = os.path.join(folder_path, f"{unique_filename}.png")
        cv2.imwrite(file_path, obj)
    else:
        raise ValueError("Unsupported object type for saving.")
    return file_path


def parse_and_save_objects(obj: Any, folder_path: str) -> Any:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, (Image.Image, np.ndarray)):
                obj[key] = save_image_or_array(value, folder_path)
            elif isinstance(value, (dict, list)):
                obj[key] = parse_and_save_objects(value, folder_path)
            elif (
                isinstance(value, str)
                and key == "url"
                and value.startswith("data:image")
            ):
                obj[key] = save_image_or_array(decode_image(value), folder_path)
            elif dataclasses.is_dataclass(value) and not isinstance(value, type):
                obj[key] = parse_and_save_objects(
                    dataclasses.asdict(value), folder_path
                )
    elif isinstance(obj, list):
        for i in range(len(obj)):
            if isinstance(obj[i], (Image.Image, np.ndarray)):
                obj[i] = save_image_or_array(obj[i], folder_path)
            elif isinstance(obj[i], (dict, list)):
                obj[i] = parse_and_save_objects(obj[i], folder_path)
            elif dataclasses.is_dataclass(obj[i]):
                obj[i] = parse_and_save_objects(dataclasses.asdict(obj[i]), folder_path)
    return obj
