import json
import os

from desktop_env.computer.env import ComputerEnv
from desktop_env.eval.bridges.bridge_helper import environment_init
from desktop_env.eval.evaluator_helper import eval_tasks


def test_calendar(
    computer_env: ComputerEnv,
) -> None:
    config_file = "desktop_env/eval/examples/google_calendar.json"
    with open(config_file, "r") as f:
        task_configs = json.load(f)

    env_comb = environment_init("config/environments.json")
    env_comb.reset(task_configs["tasks"][0]["reset_actions"])

    score = eval_tasks(
        task_configs,
        env_comb,
    )
    assert score == 1.0
