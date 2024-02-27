from pathlib import Path

from playground.utils.singleton import Singleton


class Config(metaclass=Singleton):
    """
    Singleton for config.
    """

    seed: int = 42
    headless: bool = True
    python_timeout: int = 10
    need_human_confirmation: bool = True
    on_ssh = True

    task_config_paths: dict = {
        "desktop": "playground_data/tasks/windows_easy.jsonl",
    }

    stop_code: str = "\nexit()"
    use_video = False

    # Env server config
    env_server_addr: str = "127.0.0.1"
    env_server_host: str = "0.0.0.0"
    vnc_port: int = 5900
    env_server_port: int = 8000
    vnc_password: str = "123456"

    # Recorder config
    record_path = "playground_data/trajectories"
    video_fps: int = 5
    mouse_fps: int = 5

    # Human annotator hotkeys
    stop_hotkeys: str = "<ctrl>+<shift>+h"

    # sleep_after_execution: float = 2.0
    max_step: int = 30
    system_prompt_path: str = "playground/agent/prompts/system_prompt.txt"
    # parsing_failure_th: int = 3
    # repeating_action_failure_th = 3

    # LM config
    provider: str = "gemini"
    agent: str = "direct"
    max_retries: int = 3
    # model: str = "gpt-4-1106-vision-preview"
    model: str = "gemini-pro-vision"
    model_server: str | None = None
    # eval_model: str = "gpt-4-1106-vision-preview"
    eval_model: str = "gemini-pro-vision"
    temperature: float = 0.0
    max_tokens: int = 4096
    OPENAI_API_KEY: str = "your_openai_api_key"
    GEMINI_API_KEY: str = "your_gemini_api_key"

    google_credential_path: str = "playground/config/credentials.json"
    google_calendar_id: str = "primary"
    gmail_recipient: str = "gduser1@workspacesamples.dev"
    vscode_workspace_path: str = "tmp/vscode_workspace"
    vscode_executable_path: str = "code"

    # Pyrogram config
    telegram_workdir: str = "playground/config"
    telegram_api_id: int | str = "your telegram_api_id"
    telegram_api_hash: str = "your_telegram_api_hash"

    project_root: Path = Path(__file__).resolve().parents[2]
    log_dir: Path = project_root / "logs/"

    def __init__(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def __str__(self) -> str:
        return str(self.__dict__)
