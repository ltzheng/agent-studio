import ast
import importlib
import logging
import os
from pathlib import Path

from agent_studio.config import Config
from agent_studio.envs.desktop_env.evaluators.evaluator import Evaluator
from agent_studio.utils.types import Procedure, TaskConfig

config = Config()
logger = logging.getLogger(__name__)


class EvaluatorComb:
    def __init__(self, evaluators: dict[str, Evaluator]) -> None:
        self.evaluators = evaluators

    def reset(self, reset_procedure: list[Procedure]) -> None:
        for procedure in reset_procedure:
            if procedure.evaluator not in self.evaluators:
                raise ValueError(f"Evaluator {procedure.evaluator} not found")
            self.evaluators[procedure.evaluator].reset(procedure)

    def __call__(
        self, eval_procedure: list[Procedure], **as_kwargs
    ) -> tuple[float, str]:
        score = 1.0
        feedback = ""
        for procedure in eval_procedure:
            if procedure.evaluator not in self.evaluators:
                raise ValueError(f"Evaluator {procedure.evaluator} not found")
            cur_score, cur_feedback = self.evaluators[procedure.evaluator](
                procedure, as_kwargs=as_kwargs
            )
            score *= cur_score
            feedback += cur_feedback
        # TODO: use bool instead of float
        return score, feedback


def register_evaluators(
    base_path: str | Path = "agent_studio/envs/desktop_env/evaluators",
) -> dict[str, type[Evaluator]]:
    registered_classes = {}
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)

                # Parse the Python file
                with open(file_path, "r") as f:
                    file_contents = f.read()
                try:
                    tree = ast.parse(file_contents)
                except SyntaxError:
                    logger.error(f"Error parsing {file_path}. Skipping...")
                    continue
                # Check each class definition in the file
                for node in ast.walk(tree):
                    module_name = (
                        os.path.relpath(file_path, ".")
                        .replace(os.sep, ".")
                        .rstrip(".py")
                    )
                    if isinstance(node, ast.ClassDef):
                        for base in node.bases:
                            if isinstance(base, ast.Name) and base.id == "Evaluator":
                                try:
                                    module = importlib.import_module(module_name)
                                    new_class: type[Evaluator] | None = getattr(
                                        module, node.name, None
                                    )
                                    if (
                                        new_class is not None
                                        and new_class.name not in registered_classes
                                    ):
                                        registered_classes[new_class.name] = new_class
                                    else:
                                        raise AttributeError
                                except Exception as e:
                                    logger.error(
                                        f"Skip importing {module_name} {node.name} "
                                        f"due to {e}."
                                    )
                                break
    return registered_classes


def evaluator_router(
    task_config: TaskConfig,
) -> EvaluatorComb:
    """Router to get the evaluator class"""

    registered_evaluators: dict[str, type[Evaluator]] = register_evaluators()
    evaluators: dict[str, Evaluator] = {}
    logger.info(f"Registered evaluators: {registered_evaluators.keys()}")

    procedures = task_config.eval_procedure.copy()
    if task_config.reset_procedure is not None:
        procedures += task_config.reset_procedure
    if task_config.cleanup_procedure is not None:
        procedures += task_config.cleanup_procedure

    for procedure in procedures:
        eval_type: str = procedure.evaluator
        if eval_type in registered_evaluators:
            if eval_type not in evaluators:
                evaluators[eval_type] = registered_evaluators[eval_type]()
        else:
            raise ValueError(
                f"The eval_type '{eval_type}' is not registered. "
                f"This probably indicates a bug in the code."
            )

    return EvaluatorComb(evaluators)
