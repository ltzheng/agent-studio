.. _create_agents:

Create Your Own Agent
=====================

Not satisfied with the existing agents? Create your own agent! This section will guide you through the process of creating a new agent.

Agents are stored in the ``agent_studio/agent`` directory. Each agent is a Python class that inherits from the ``BaseAgent`` class. The agent class must implement the following methods:

1. ``name: str``:
    A unique name for the agent. **The name should match the ``agent`` field in ``agent_studio/config/config.py``**.
2. ``trajectory2intermediate_msg(self) -> list[dict[str, Any]]``:
    Convert the trajectory to a list of intermediate messages. You can access all internal states of the agent, e.g. ``self.trajectories``, ``self.system_prompt``, and ``self.instruction``. Construct the intermediate messages based on these states. The intermediate messages is a list of messages.Each message is a dictionary with the following fields:

    ``role: str``:
        The role of the message. It can be ``system``, ``user``, or ``assistant``.
    ``content: str | np.array``:
        The content can be a string or a numpy array.
3. ``eval(self, final_obs: np.ndarray | None = None) -> dict[str, Any]``:
    Agent self-evaluation. The input is the final observation of the task (a screenshot if the current task is a visual task). You can use this information and past trajectories to construct the self-evaluation prompt. The method should return a dictionary with the following fields:

    ``score: float``:
        The score of the agent trajectories.
    ``feedback: str``:
        The feedback from the agent.
    ``prompt: str``:
        The self-evaluation prompt. **Must be in the format of the intermediate message format.**
    ``response: str``:
        The original response from the agent.

---------

You may also want to implement the following methods:

1. ``reset(self, instruction: str) -> None:``:
    Reset the agent for a new task. This will read the task instructions and reset the agent's internal state. You can also read the system prompt here.

