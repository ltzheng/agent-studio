import ast
import importlib
import logging
import os

from agent_studio.agent.base_agent import BaseAgent

logger = logging.getLogger(__name__)


def register_agents(
    base_path: str = "agent_studio/agent",
) -> dict[str, type[BaseAgent]]:
    registered_classes = {}
    for file in os.listdir(base_path):
        if file.endswith(".py"):
            file_path = os.path.join(base_path, file)

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
                    os.path.relpath(file_path, ".").replace(os.sep, ".").rstrip(".py")
                )
                if isinstance(node, ast.ClassDef):
                    for base in node.bases:
                        if isinstance(base, ast.Name) and base.id == "BaseAgent":
                            try:
                                module = importlib.import_module(module_name)
                                new_class: type[BaseAgent] | None = getattr(
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
                                logger.warning(
                                    f"Skip importing {module_name} {node.name} "
                                    f"due to {e}."
                                )
                            break
    return registered_classes


def setup_agent(agent_name: str, **kwargs) -> BaseAgent:
    registered_agents: dict[str, type[BaseAgent]] = register_agents()
    logger.info(f"Registered agents: {registered_agents.keys()}")
    if agent_name not in registered_agents:
        logger.error(f"Agent '{agent_name}' is not registered")
        raise ValueError(f"Agent '{agent_name}' is not registered")
    else:
        logger.info(f"Setting up agent: {agent_name}")
        agent = registered_agents[agent_name](**kwargs)

    return agent


__all__ = [
    "setup_agent",
]
