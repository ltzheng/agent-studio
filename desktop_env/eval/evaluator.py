from desktop_env.eval.bridges.bridge import Bridge

"""base class for evaluation"""


class Evaluator(object):
    def __init__(
        self,
        reference_answer: dict,
        env: Bridge,
        reset_actions: list[dict],
        env_settings: dict | None = None,
        eval_tag: str = "",
    ) -> None:
        self.reference_answer = reference_answer
        self.env = env
        self.env_settings = env_settings
        self.eval_tag = eval_tag
        self.reset_actions = reset_actions

    def reset(self) -> bool:
        return self.env.reset(self.reset_actions)

    def __call__(self) -> float:
        raise NotImplementedError
