from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import threading
import time
from pathlib import Path

import cv2
import jsonpickle
import numpy as np
import requests
from PyQt6.QtCore import (
    QEvent,
    QMutex,
    QObject,
    QSize,
    Qt,
    QThread,
    QTimer,
    QWaitCondition,
    pyqtSignal,
)
from PyQt6.QtGui import QAction, QImage
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from tqdm import tqdm

from agent_studio.agent import setup_agent
from agent_studio.agent.base_agent import BaseAgent
from agent_studio.config.config import Config
from agent_studio.envs.desktop_env.evaluators.evaluator_helper import evaluator_router
from agent_studio.envs.desktop_env.vnc_client import (
    LocalStreamer,
    VNCFrame,
    VNCStreamer,
)
from agent_studio.utils.communication import (
    AgentStudioEvalRequest,
    AgentStudioResetRequest,
    AgentStudioStatusResponse,
    AgentStudioTextRequest,
)
from agent_studio.utils.gui import (
    ChoiceDialog,
    ChoiceDialogPython,
    InputDialog,
    JSONEditor,
    TimerLabel,
)
from agent_studio.utils.json_utils import (
    apply_env_vars,
    export_trajectory,
    read_task_jsons,
    read_unfinished_tasks,
)
from agent_studio.utils.types import TaskConfig, VideoMeta

config = Config()

logger = logging.getLogger(__name__)
REMOTE_SERVER_ADDR = f"http://{config.env_server_addr}:{config.env_server_port}"


class FrameBuffer:
    def __init__(self):
        self.queue = []
        self.lock = threading.Lock()

    def add_frame(self, frame):
        with self.lock:
            self.queue.append(frame)

    def clear(self):
        with self.lock:
            self.queue.clear()

    def get_frames(self):
        with self.lock:
            frames = self.queue.copy()
        return frames


class WorkerSignals(QObject):
    show_dialog_signal = pyqtSignal(str, str)
    show_dialog_signal_python = pyqtSignal(str, str)
    show_input_dialog_signal = pyqtSignal(str, str)
    runtime_output_signal = pyqtSignal(dict)
    evaluation_result_signal = pyqtSignal(str)
    exec_finish_signal = pyqtSignal()
    eval_finish_signal = pyqtSignal()
    status_bar_signal = pyqtSignal(str, str)


class TaskThread(QThread):
    def __init__(
        self,
        agent: BaseAgent,
        task_config: TaskConfig,
        signal: WorkerSignals,
        args: argparse.Namespace,
        results_dir: Path,
        interface: GUI,
    ):
        super().__init__()
        self.mutex = QMutex()
        self.wait_condition = QWaitCondition()
        self.agent: BaseAgent = agent
        self.task_config: TaskConfig = task_config
        self.signals = signal
        self.args = args
        self.interface = interface
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def wait_finish(self, is_eval: bool, response: AgentStudioStatusResponse):
        if response.status == "finished":
            return response
        elif response.status == "wait_for_input":
            if is_eval:
                # evaluation
                self.mutex.lock()
                self.signals.status_bar_signal.emit(
                    "color: red;", "Waiting for user input..."
                )
                self.signals.show_input_dialog_signal.emit(
                    "Human Evaluation", response.content
                )
                self.wait_condition.wait(self.mutex)
                self.mutex.unlock()
                user_input = self.user_input
                self.signals.status_bar_signal.emit("color: blue;", "Evaluating...")
            else:
                # reset
                if config.need_human_confirmation:
                    self.mutex.lock()
                    self.signals.status_bar_signal.emit(
                        "color: red;", "Waiting for user input..."
                    )
                    self.signals.show_dialog_signal.emit(
                        "Confirm Action", response.content
                    )
                    self.wait_condition.wait(self.mutex)
                    self.mutex.unlock()
                    user_input = self.user_input
                else:
                    user_input = "y"
                self.signals.status_bar_signal.emit("color: blue;", "Resetting Task...")
            response_raw = requests.post(
                url=f"{REMOTE_SERVER_ADDR}/task/confirm",
                json=AgentStudioTextRequest(message=user_input).model_dump(),
            )
            assert response_raw.status_code == 200
            response = AgentStudioStatusResponse(**response_raw.json())
            return self.wait_finish(is_eval, response)
        else:
            raise ValueError(f"Unknown status: {response.status}, {response.content}")

    def run(self):
        if not self.args.remote:
            raise ValueError("Local mode is not supported.")
        try:
            logger.info(f"Start task: {self.task_config.task_id}")
            # Get remote env_vars
            response_raw = requests.get(f"{REMOTE_SERVER_ADDR}/env_vars")
            response = AgentStudioStatusResponse(**response_raw.json())
            assert (
                response.status == "success"
            ), f"Fail to reset task: {response.message}"
            env_vars = response.message
            assert isinstance(env_vars, dict), "Invalid env_vars"
            logger.debug(f"Env vars: {env_vars}")
            logger.debug(f"Task config before: {self.task_config}")
            self.task_config = apply_env_vars(self.task_config, env_vars)
            logger.debug(f"Task config after: {self.task_config}")
            # Reset
            if self.task_config.reset_procedure is not None:
                self.signals.status_bar_signal.emit("color: blue;", "Resetting Task...")
                response_raw = requests.post(
                    f"{REMOTE_SERVER_ADDR}/task/reset",
                    json=AgentStudioResetRequest(
                        procedures=self.task_config.reset_procedure
                    ).model_dump(),
                )
                response = AgentStudioStatusResponse(**response_raw.json())
                response = self.wait_finish(is_eval=False, response=response)
                if not (
                    response.status == "finished" and response.content == "success"
                ):
                    raise ValueError(
                        f"Fail to reset task: {response.message}"
                        f", get response {response.content}"
                    )

            instruction = self.task_config.instruction
            logger.info(f"Task instruction: {instruction}")

            # Reset the agent
            self.signals.status_bar_signal.emit("color: blue;", "Resetting Agent...")
            self.agent.reset(task_config=self.task_config)
            if self.task_config.visual:
                assert (
                    self.interface is not None
                ), "Interface has to be open for visual tasks."
                self.interface.start_recording()

            # Loop until the task is done or the max step is reached.
            start_time = time.time()
            current_step = 0
            action_memory = []
            while True:
                logger.info(f"Step {current_step}")
                if self.task_config.visual:
                    obs = self.interface.get_screenshot()
                else:
                    obs = None
                self.signals.status_bar_signal.emit(
                    "color: blue;", "Generating Action..."
                )
                step_info = self.agent.generate_action(
                    obs=obs, model_name=self.args.model
                )
                action = step_info.action
                action_memory.append(action)

                failure_msg: None | str = None
                if config.need_human_confirmation:
                    self.mutex.lock()
                    self.signals.status_bar_signal.emit(
                        "color: red;", "Waiting for user input..."
                    )
                    self.signals.show_dialog_signal_python.emit(
                        "Execute Action?", action
                    )
                    self.wait_condition.wait(self.mutex)
                    self.mutex.unlock()
                    if self.user_input.strip().lower() != "y":
                        failure_msg = "Cancelled by human."
                # If the max step is reached.
                elif current_step >= self.task_config.max_steps:
                    failure_msg = "Max step reached."
                # If the time limit is reached.
                elif (
                    self.args.use_time_limit
                    and time.time() - start_time > self.task_config.max_time
                ):
                    failure_msg = "Time limit reached."
                # If the action is empty.
                elif action == "":
                    failure_msg = "Failed to generate action."
                # If the action is the same as the previous two actions.
                elif (
                    len(action_memory) >= 3
                    and action_memory[-1] == action_memory[-2] == action_memory[-3]
                ):
                    failure_msg = "Repeated action."
                self.signals.status_bar_signal.emit(
                    "color: blue;", "Executing Command..."
                )
                runtime_output, done = self.agent.step_action(
                    failure_msg=failure_msg, step_info=step_info
                )
                self.signals.runtime_output_signal.emit(runtime_output)
                # Wait for the action to be executed
                time.sleep(config.min_action_interval)
                if done:
                    break
                current_step += 1
            stop_time = time.time()

            self.signals.exec_finish_signal.emit()
            self.signals.status_bar_signal.emit("color: blue;", "Evaluating Task...")
            if not self.args.no_log:
                video_meta: VideoMeta | None = None
                task_result_path = Path(self.results_dir) / self.task_config.task_id
                if not task_result_path.exists():
                    task_result_path.mkdir(parents=True, exist_ok=True)
                if self.task_config.visual:
                    video_folder = task_result_path
                    video_folder.mkdir(parents=True, exist_ok=True)
                    video_path = video_folder / "video.mp4"
                    video_meta = self.interface.save_video(video_path)
                    logger.info(f"Video saved to {video_path}")

            response_raw = requests.post(
                f"{REMOTE_SERVER_ADDR}/task/eval",
                json=AgentStudioEvalRequest(
                    procedures=self.task_config.eval_procedure,
                    as_kwargs=str(
                        jsonpickle.encode({"trejectory": self.agent.trajectory})
                    ),
                ).model_dump(),
            )
            response = AgentStudioStatusResponse(**response_raw.json())
            response = self.wait_finish(is_eval=True, response=response)
            if not (
                response.status == "finished"
                and isinstance(response.message, dict)  # noqa: E501
            ):
                raise ValueError(f"Fail to evaluate task: {response.message}")
            score, feedback = (
                response.message["score"],
                response.message["feedback"],
            )

            if score == 1.0:
                logger.info(f"[Result] (PASS): {feedback}")
            else:
                logger.info(f"[Result] (FAIL): {feedback}")
            self.signals.evaluation_result_signal.emit(
                f"Score: {score}\nFeedback: {feedback}"
            )

            if not self.args.no_log:
                self.signals.status_bar_signal.emit(
                    "color: blue;", "Exporting Trajectory..."
                )
                export_trajectory(
                    task_config=self.task_config,
                    trajectory=self.agent.trajectory,
                    path=task_result_path,
                    score=score,
                    feedback=feedback,
                    token_count=self.agent.get_token_count(),
                    time_cost=stop_time - start_time,
                    video_meta=video_meta,
                )
        except Exception as e:
            import traceback

            logger.error(f"[Unhandled Error] {repr(e)}]")
            traceback.print_exc()
        finally:
            # Cleanup
            if self.task_config.cleanup_procedure is not None:
                self.signals.status_bar_signal.emit(
                    "color: blue;", "Cleaning up Task..."
                )
                response_raw = requests.post(
                    f"{REMOTE_SERVER_ADDR}/task/reset",
                    json=AgentStudioResetRequest(
                        procedures=self.task_config.cleanup_procedure
                    ).model_dump(),
                )
                response = AgentStudioStatusResponse(**response_raw.json())
                response = self.wait_finish(is_eval=False, response=response)
                if not (
                    response.status == "finished" and response.content == "success"
                ):
                    logger.error(f"Fail to cleanup task: {response.message}")
            self.signals.status_bar_signal.emit("color: green;", "Ready")
            self.signals.eval_finish_signal.emit()

    def receive_user_input(self, text: str):
        self.mutex.lock()
        self.user_input = text  # Store the user input
        self.wait_condition.wakeAll()  # Resume the thread
        self.mutex.unlock()


class GUI(QMainWindow):
    """Main class for the agent monitor."""

    def __init__(
        self,
        args: argparse.Namespace,
        remote: bool,
        window_width: int,
        window_height: int,
    ) -> None:
        """Initializes the UI."""
        super().__init__()
        self.frame_buffer: FrameBuffer = FrameBuffer()
        self.args = args
        self.remote = remote
        self.results_dir = Path(
            f"{self.args.log_dir}/{self.args.model}/{self.args.agent}"
        )
        self.task_config_path = Path(self.args.task_configs_path)
        # Setup tasks
        self.load_task_configs()
        self.selected_task: TaskConfig
        self.selected_task_idx: int | None = None
        self.window_width = window_width
        self.window_height = window_height

        # Setup a QTimer to periodically update the screen.
        self.refresh_timer = QTimer(self)
        self.refresh_timer.setInterval(int(1000 / config.video_fps))
        self.refresh_timer.timeout.connect(self.render)
        self.refresh_timer.start()
        self.refreshing_screen = False

        self.dlg: QDialog | None = None
        self.task_thread: TaskThread | None = None

        self.agent = setup_agent(
            agent_name=self.args.agent,
            model=self.args.model,
            remote=self.args.remote,
            runtime_server_addr=config.env_server_addr,
            runtime_server_port=config.env_server_port,
        )

        # self.task_thread: None | TaskThread = None
        self.capture_thread: VNCStreamer | LocalStreamer | None = None
        if remote:
            self.capture_thread = VNCStreamer(
                config.env_server_addr, config.vnc_port, config.vnc_password
            )
        else:
            self.capture_thread = LocalStreamer(config.monitor_idx)

        self.is_recording = False
        self.recording_thread: threading.Thread | None = None

        # Initialize the screenshot as a blank image.
        self.video_height, self.video_width = (
            self.capture_thread.video_height,
            self.capture_thread.video_width,
        )
        self.now_screenshot = np.zeros(
            (self.video_height, self.video_width, 4), dtype="uint8"
        )

        self.setup_ui()

        self.reset()

        # Install event filter on the main window
        self.installEventFilter(self)

    def load_task_configs(self):
        try:
            if self.args.ignore_finished:
                self.task_configs: list[TaskConfig] = read_unfinished_tasks(
                    Path(self.task_config_path), self.results_dir
                )
            else:
                self.task_configs: list[TaskConfig] = read_task_jsons(
                    Path(self.task_config_path)
                )
            self.task_configs.sort(key=lambda x: x.instruction)
        except Exception as e:
            logger.error(f"Failed to load task configs: {e}")
            self.task_configs = []
            self.status_bar.showMessage("Failed to load task configs", 3000)

    def setup_ui(self):
        """Sets up the UI, including the VNC frame (left) and the right layout."""
        # Setup the user interface for the application.
        self.setWindowTitle("Agent Monitor")

        # Create menu bar
        menubar = self.menuBar()
        assert menubar is not None
        file_menu = menubar.addMenu("File")
        assert file_menu is not None

        # Open Task Config
        open_file_action = QAction("Open Config", self)
        open_file_action.triggered.connect(self.open_task_config)
        file_menu.addAction(open_file_action)

        # Reload Task Configs
        reload_action = QAction("Reload Configs", self)
        reload_action.triggered.connect(self.reload_task_configs)
        file_menu.addAction(reload_action)

        # Central widget to hold the main layout.
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Setup the status bar.
        status_bar = self.statusBar()
        assert status_bar is not None
        self.status_bar: QStatusBar = status_bar
        self.task_status_bar = QLabel()
        self.set_task_status_bar_text("color: green;", "Task: Init")
        self.status_bar.addPermanentWidget(self.task_status_bar)

        if self.remote:
            # Setup the VNC frame for video display.
            self.frame_size_hint = QSize(self.window_width, self.window_height)
            # Left layout for VNC frame.
            left_layout = QVBoxLayout()
            self.vnc_frame = VNCFrame(
                self, self.frame_size_hint, enable_selection=False
            )
            left_layout.addWidget(self.vnc_frame)

            button_timer_layout = QHBoxLayout()
            button_timer_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

            self.timer_label = TimerLabel(self)
            button_timer_layout.addWidget(self.timer_label)

            # Add the horizontal layout to the main left layout
            left_layout.addLayout(button_timer_layout)

            vnc_widget = QWidget()
            vnc_widget.setLayout(left_layout)
            main_splitter.addWidget(vnc_widget)

        # Task layout #
        task_layout = QVBoxLayout()

        task_layout.addWidget(QLabel("Task configuration"))

        self.task_config_display = JSONEditor(editable=False, parent=self)

        # Add the JSONEditor to a splitter to allow resizing
        task_splitter = QSplitter(Qt.Orientation.Vertical)
        task_splitter.addWidget(self.task_config_display)

        task_layout.addWidget(task_splitter)

        task_layout.addWidget(QLabel("Task Selection (double click to select)"))
        self.instruction_selection = QListWidget(self)
        self.instruction_selection.itemDoubleClicked.connect(
            self.select_task_instruction
        )
        self.instruction_selection.clear()
        self.populate_instruction_selection_widget()

        # Add the QListWidget to the same splitter for resizing
        task_splitter.addWidget(self.instruction_selection)

        execution_button_layout = QHBoxLayout()

        self.start_button = QPushButton("Start")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.task_start)
        execution_button_layout.addWidget(self.start_button)

        task_layout.addLayout(execution_button_layout)

        self.interrupt_button = QPushButton("Interrupt")
        self.interrupt_button.clicked.connect(self.reset)
        task_layout.addWidget(self.interrupt_button)

        self.status_bar.showMessage("Ready")

        task_widget = QWidget()
        task_widget.setLayout(task_layout)
        main_splitter.addWidget(task_widget)

        # End of task layout #

        # Agent layout #
        agent_layout = QVBoxLayout()

        self.show_trajectory_button = QPushButton("Show Trajectory")
        agent_layout.addWidget(self.show_trajectory_button)

        agent_layout.addWidget(QLabel("Runtime output"))

        self.output_display = JSONEditor(editable=False, parent=self)
        self.output_display.setReadOnly(True)

        # Add the JSONEditor to a splitter to allow resizing
        agent_splitter = QSplitter(Qt.Orientation.Vertical)
        agent_splitter.addWidget(self.output_display)

        agent_layout.addWidget(agent_splitter)

        agent_layout.addWidget(QLabel("Evaluation Result"))

        self.evaluation_display = QTextEdit(self)
        self.evaluation_display.setReadOnly(True)

        # Add the QTextEdit to the same splitter for resizing
        agent_splitter.addWidget(self.evaluation_display)

        agent_widget = QWidget()
        agent_widget.setLayout(agent_layout)
        main_splitter.addWidget(agent_widget)

        main_layout.addWidget(main_splitter)
        # End of agent layout #

        self.setMouseTracking(True)

        # Add keyboard shortcuts
        self.start_button.setShortcut("Shift+Return")  # Shift + Enter to start task
        self.show_trajectory_button.setShortcut("Ctrl+T")  # Ctrl + T to show trajectory
        self.interrupt_button.setShortcut("Escape")
        self.instruction_selection.setFocusPolicy(
            Qt.FocusPolicy.StrongFocus
        )  # Ensure the list can receive focus
        self.instruction_selection.installEventFilter(
            self
        )  # Install event filter for key events

    def reload_task_configs(self):
        self.load_task_configs()
        self.populate_instruction_selection_widget()
        self.status_bar.showMessage("Task configs reloaded", 3000)

    def open_task_config(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Open Task Config Folder")
        if folder_path:
            self.task_config_path = Path(folder_path)
            try:
                self.load_task_configs()
                self.populate_instruction_selection_widget()
                self.status_bar.showMessage(
                    f"Task configs added from: {folder_path}", 3000
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Error", f"Failed to load task configs: {str(e)}"
                )

    def show_input_dialog(self, title: str, message: str):
        assert self.task_thread is not None
        self.dlg = InputDialog(title, message, self.task_thread.receive_user_input)
        self.dlg.setWindowModality(Qt.WindowModality.NonModal)
        self.dlg.show()

    def show_choice_dialog(self, title: str, message: str):
        self.dlg = ChoiceDialog(
            title,
            message,
            lambda: (
                self.task_thread.receive_user_input("y")
                if self.task_thread is not None
                else None
            ),
            lambda: (
                self.task_thread.receive_user_input("n")
                if self.task_thread is not None
                else None
            ),
        )
        self.dlg.setWindowModality(Qt.WindowModality.NonModal)
        self.dlg.show()

    def show_choice_dialog_python(self, title: str, message: str):
        assert self.task_thread is not None
        self.dlg = ChoiceDialogPython(
            title,
            message,
            lambda: (
                self.task_thread.receive_user_input("y")
                if self.task_thread is not None
                else None
            ),
            lambda: (
                self.task_thread.receive_user_input("n")
                if self.task_thread is not None
                else None
            ),
        )
        self.dlg.setWindowModality(Qt.WindowModality.NonModal)
        self.dlg.show()

    def set_task_status_bar_text(self, color: str, text: str) -> None:
        self.task_status_bar.setStyleSheet(color)
        self.task_status_bar.setText(text)

    def populate_instruction_selection_widget(self):
        self.instruction_selection.clear()
        for task in self.task_configs:
            item = QListWidgetItem(task.instruction)
            self.instruction_selection.addItem(item)

    def select_task_instruction(self, item):
        self.task_instruction = item.text()
        self.selected_task_idx = self.instruction_selection.currentRow()
        self.selected_task = self.task_configs[self.selected_task_idx]
        self.task_config_display.setText(self.selected_task.model_dump_json(indent=4))
        self.evaluation_display.clear()
        self.output_display.clear()
        # if self.selected_task.task_id in self.task_results:
        #     score = self.task_results[self.selected_task.task_id]["score"]
        #     feedback = self.task_results[self.selected_task.task_id]["feedback"]
        #     self.evaluation_display.setPlainText(
        #         f"Score: {score}\n" f"Feedback: {feedback}"
        #     )

        self.start_button.setEnabled(True)

    def task_start(self):
        self.start_button.setEnabled(False)
        self.output_display.clear()
        self.evaluation_display.clear()
        self.instruction_selection.setEnabled(False)
        signals = WorkerSignals()
        signals.show_dialog_signal.connect(self.show_choice_dialog)
        signals.show_dialog_signal_python.connect(self.show_choice_dialog_python)
        signals.show_input_dialog_signal.connect(self.show_input_dialog)
        signals.runtime_output_signal.connect(self.output_display.setText)
        signals.evaluation_result_signal.connect(self.evaluation_display.setPlainText)
        signals.exec_finish_signal.connect(self.exec_finished)
        signals.eval_finish_signal.connect(self.task_finished)
        signals.status_bar_signal.connect(self.set_task_status_bar_text)
        self.task_thread = TaskThread(
            agent=self.agent,
            task_config=self.selected_task,
            signal=signals,
            args=self.args,
            interface=self,
            results_dir=self.results_dir,
        )
        self.task_thread.start()
        self.timer_label.start()

    def exec_finished(self):
        self.timer_label.stop()

    def task_finished(self):
        if self.task_thread is not None:
            self.task_thread.wait()
            self.task_thread = None
        self.interrupt_button.setEnabled(True)
        self.instruction_selection.setEnabled(True)

    def reset(self) -> None:
        """Resets the UI elements to their default state."""
        self.refresh_timer.start()
        # set widgets to default state
        self.interrupt_button.setEnabled(True)
        self.start_button.setEnabled(False)
        self.instruction_selection.setEnabled(True)
        self.timer_label.stop()
        # clear displays and selections
        self.instruction_selection.clearSelection()
        self.output_display.clear()
        self.evaluation_display.clear()
        self.task_config_display.clear()
        # clear current dlg
        if self.dlg is not None:
            self.dlg.close()
            self.dlg = None
        # clear working thread
        if self.task_thread is not None:
            self.task_thread.terminate()
            self.task_thread = None
        self.is_recording = False
        if self.recording_thread is not None:
            self.recording_thread.join()
            self.recording_thread = None
        # reset status bar
        self.set_task_status_bar_text("color: green;", "Ready")
        self.status_bar.showMessage(
            "Connected. Please go to the terminal to check outputs."
        )

    def render(self):
        """Renders the screen by periodically updating the frame."""
        self.refresh_timer.stop()
        if self.refreshing_screen:
            self.refresh_timer.start()
            return
        self.refreshing_screen = True

        # Update the screen with the latest frame from the capture thread.
        try:
            assert self.capture_thread is not None
            frame = self.capture_thread.get_current_frame()
            if frame is not None:
                self.now_screenshot = frame
                qimage = QImage(
                    frame.tobytes(),
                    self.video_width,
                    self.video_height,
                    QImage.Format.Format_RGB888,
                )
                self.vnc_frame.update(qimage)

                if self.is_recording:
                    self.frame_buffer.add_frame(self.now_screenshot)
        except Exception as e:
            print("Failed to get screenshot.", e)

        self.refreshing_screen = False
        self.refresh_timer.start()

    def start_recording(self):
        """Starts recording the video."""
        assert not self.is_recording, "Recording is already in progress."
        self.is_recording = True
        self.frame_buffer.clear()
        self.status_bar.showMessage("Recording started.")

    def get_screenshot(self) -> np.ndarray:
        """Returns the current frame as a numpy array."""
        while np.all(self.now_screenshot == 0):
            print("Waiting for the first frame.")
            time.sleep(0.5)
        frame = cv2.cvtColor(self.now_screenshot, cv2.COLOR_BGRA2RGB)
        return np.array(frame)

    def save_video(self, video_path: Path) -> VideoMeta:
        """Stops recording and saves the video to the file system.

        Returns:
            dict: Metadata about the saved video.
        """
        assert (
            self.is_recording and self.frame_buffer is not None
        ), "No recording in progress."
        self.is_recording = False
        if self.recording_thread is not None:
            self.recording_thread.join()
            self.recording_thread = None
        writer = cv2.VideoWriter(
            video_path.as_posix(),
            cv2.VideoWriter.fourcc(*"mp4v"),
            config.video_fps,
            (self.video_width, self.video_height),
        )
        frames = self.frame_buffer.get_frames()
        logger.info(f"Captured {len(frames)} frames with FPS={config.video_fps}")
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        self.frame_buffer.clear()
        self.status_bar.showMessage("Recording stopped and video saved.")

        return VideoMeta(
            fps=config.video_fps,
            frame_count=len(frames),
            video_path=f"file://{video_path.name}",
            width=self.video_width,
            height=self.video_height,
        )

    def closeEvent(self, event):
        """Handles the close event by stopping the capture thread and timer.

        Args:
            event: The close event.
        """
        self.agent.close()

        self.refresh_timer.stop()
        if self.recording_thread is not None:
            self.recording_thread.join()
        if self.capture_thread is not None:
            self.capture_thread.stop()

        exit(0)

    def eventFilter(self, source, event):
        if event.type() == QEvent.Type.KeyPress:  # Change here
            if (
                event.key() == Qt.Key.Key_Return
                and event.modifiers() == Qt.KeyboardModifier.ControlModifier
            ):
                # Ctrl + Enter to start next task
                if self.selected_task_idx is None:
                    if len(self.task_configs) > 0:
                        self.instruction_selection.setCurrentRow(0)
                        self.select_task_instruction(self.instruction_selection.item(0))
                elif self.selected_task_idx < len(self.task_configs) - 1:
                    self.instruction_selection.setCurrentRow(self.selected_task_idx + 1)
                    self.select_task_instruction(
                        self.instruction_selection.item(self.selected_task_idx + 1)
                    )
                else:
                    return True
                self.start_button.click()
            if source is self.instruction_selection:
                if event.key() == Qt.Key.Key_Return:  # Enter to select item
                    self.select_task_instruction(
                        self.instruction_selection.currentItem()
                    )
                    return True
        return super().eventFilter(source, event)


class NonGUI:
    def __init__(
        self,
        args: argparse.Namespace,
        remote: bool,
        window_width: int,
        window_height: int,
    ) -> None:
        """Initializes the UI."""
        super().__init__()
        self.frame_buffer: FrameBuffer = FrameBuffer()
        self.args = args
        self.remote = remote
        self.window_width = window_width
        self.window_height = window_height

        self.fps = config.video_fps

        self.capture_thread: VNCStreamer | LocalStreamer | None = None
        if remote:
            self.capture_thread = VNCStreamer(
                config.env_server_addr, config.vnc_port, config.vnc_password
            )
        else:
            self.capture_thread = LocalStreamer(config.monitor_idx)
        self.data_lock = threading.Lock()

        # Initialize the screenshot as a blank image.
        self.video_height, self.video_width = (
            self.capture_thread.video_height,
            self.capture_thread.video_width,
        )
        self.now_screenshot = np.zeros(
            (self.video_height, self.video_width, 4), dtype="uint8"
        )
        self.recording_thread: threading.Thread | None = None

    def start_recording(self):
        self.is_recording = True
        self.recording_thread = threading.Thread(target=self._capture)
        self.recording_thread.start()
        self.frame_buffer.clear()

    def get_screenshot(self):
        while np.all(self.now_screenshot == 0):
            print("Waiting for the first frame.")
            time.sleep(0.5)
        with self.data_lock:
            frame = cv2.cvtColor(self.now_screenshot, cv2.COLOR_BGRA2RGB)
            return np.array(frame)

    def _capture(self):
        assert self.capture_thread is not None
        while self.is_recording:
            with self.data_lock:
                frame = self.capture_thread.get_current_frame()
                if frame is not None:
                    self.now_screenshot = frame.copy()
                    self.frame_buffer.add_frame(self.now_screenshot)
            time.sleep(1.0 / config.video_fps)
        logger.info("Recording thread stopped")

    def save_video(self, video_path: Path) -> VideoMeta:
        """Stops recording and saves the video to the file system.

        Returns:
            dict: Metadata about the saved video.
        """
        assert (
            self.is_recording
            and self.frame_buffer is not None
            and self.recording_thread is not None
        ), "No recording in progress."
        self.is_recording = False
        self.recording_thread.join()
        self.recording_thread = None
        writer = cv2.VideoWriter(
            video_path.as_posix(),
            cv2.VideoWriter.fourcc(*"mp4v"),
            config.video_fps,
            (self.video_width, self.video_height),
        )
        frames = self.frame_buffer.get_frames()
        logger.info(f"Captured {len(frames)} frames with FPS={config.video_fps}")
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        self.frame_buffer.clear()

        return VideoMeta(
            fps=config.video_fps,
            frame_count=len(frames),
            video_path=f"file://{video_path.name}",
            width=self.video_width,
            height=self.video_height,
        )

    def close(self):
        if self.recording_thread is not None:
            self.is_recording = False
            self.recording_thread.join()
        if self.capture_thread is not None:
            self.capture_thread.stop()


def wait_finish(is_eval: bool, response: AgentStudioStatusResponse):
    if response.status == "finished":
        return response
    elif response.status == "wait_for_input":
        # Can't override in eval mode
        if config.need_human_confirmation and not is_eval:
            user_input = input(response.content)
        else:
            user_input = "y"
        response_raw = requests.post(
            url=f"{REMOTE_SERVER_ADDR}/task/confirm",
            json=AgentStudioTextRequest(message=user_input).model_dump(),
        )
        assert response_raw.status_code == 200
        response = AgentStudioStatusResponse(**response_raw.json())
        return wait_finish(is_eval, response)
    else:
        raise ValueError(f"Unknown status: {response.status}, {response.content}")


def eval(args, interface: NonGUI | None = None) -> None:
    try:
        # Setup agent
        agent = setup_agent(
            agent_name=args.agent,
            model=args.model,
            remote=args.remote,
            runtime_server_addr=config.env_server_addr,
            runtime_server_port=config.env_server_port,
        )
        results_dir = Path(f"{args.log_dir}/{args.model}/{args.agent}")
        results_dir.mkdir(parents=True, exist_ok=True)

        # Setup tasks
        if args.ignore_finished:
            task_configs_json = read_unfinished_tasks(
                Path(args.task_configs_path), results_dir
            )
        else:
            task_configs_json = read_task_jsons(Path(args.task_configs_path))
        task_configs: list[TaskConfig] = []
        for task_config in task_configs_json:
            task_configs.append(TaskConfig.model_validate(task_config))

        # Run evaluation
        scores = {}
        for task_config in tqdm(task_configs, desc="Evaluating tasks"):
            logger.info(f"Start task: {task_config.task_id}")
            try:
                # Get remote env_vars
                if args.remote:
                    response_raw = requests.get(f"{REMOTE_SERVER_ADDR}/env_vars")
                    response = AgentStudioStatusResponse(**response_raw.json())
                    assert (
                        response.status == "success"
                    ), f"Fail to reset task: {response.message}"
                    env_vars = response.message
                    assert isinstance(env_vars, dict), "Invalid env_vars"
                else:
                    env_vars = config.env_vars
                logger.debug(f"Env vars: {env_vars}")
                logger.debug(f"Task config before: {task_config}")
                task_config = apply_env_vars(task_config, env_vars)
                logger.debug(f"Task config after: {task_config}")
                # Reset
                if task_config.reset_procedure is not None:
                    if args.remote:
                        response_raw = requests.post(
                            f"{REMOTE_SERVER_ADDR}/task/reset",
                            json=AgentStudioResetRequest(
                                procedures=task_config.reset_procedure
                            ).model_dump(),
                        )
                        response = AgentStudioStatusResponse(**response_raw.json())
                        response = wait_finish(is_eval=False, response=response)
                        assert (
                            response.status == "finished"
                            and response.content == "success"
                        ), f"Fail to reset task: {response.message}"
                    else:
                        evaluators = evaluator_router(task_config)
                        evaluators.reset(task_config.reset_procedure)

                instruction = task_config.instruction
                logger.info(f"Task instruction: {instruction}")

                # Reset the agent
                agent.reset(task_config=task_config)
                if task_config.visual:
                    assert (
                        interface is not None
                    ), "Interface has to be open for visual tasks."
                    interface.start_recording()

                # Loop until the task is done or the max step is reached.
                start_time = time.time()
                current_step = 0
                action_memory = []
                while True:
                    logger.info(f"Step {current_step}")
                    if task_config.visual:
                        assert (
                            interface is not None
                        ), "Interface has to be open for visual tasks."
                        obs = interface.get_screenshot()
                    else:
                        obs = None
                    step_info = agent.generate_action(obs=obs, model_name=args.model)
                    action = step_info.action
                    action_memory.append(action)

                    failure_msg: None | str = None
                    if config.need_human_confirmation and (
                        input(f"Action:\n{action}\nConfirm action (y/n): ")
                        .strip()
                        .lower()
                        != "y"
                    ):
                        failure_msg = "Cancelled by human."
                    # If the max step is reached.
                    elif current_step >= task_config.max_steps:
                        failure_msg = "Max step reached."
                    # If the time limit is reached, the action is not confirmed.
                    elif (
                        args.use_time_limit
                        and time.time() - start_time > task_config.max_time
                    ):
                        failure_msg = "Time limit reached."
                    # If the action is empty.
                    elif action == "":
                        failure_msg = "Failed to generate action."
                    # If the action is the same as the previous two actions.
                    elif (
                        len(action_memory) >= 3
                        and action_memory[-1] == action_memory[-2] == action_memory[-3]
                    ):
                        failure_msg = "Repeated action."
                    _, done = agent.step_action(
                        failure_msg=failure_msg, step_info=step_info
                    )
                    time.sleep(config.min_action_interval)
                    if done:
                        break
                    current_step += 1
                stop_time = time.time()

                if not args.no_log:
                    task_trajectory_path = results_dir / task_config.task_id
                    if not task_trajectory_path.exists():
                        task_trajectory_path.mkdir(parents=True, exist_ok=True)
                    video_meta: VideoMeta | None = None
                    if task_config.visual:
                        task_trajectory_path.mkdir(parents=True, exist_ok=True)
                        video_path = task_trajectory_path / "video.mp4"
                        assert interface is not None
                        video_meta = interface.save_video(video_path)
                        logger.info(f"Video saved to {video_path}")

                if args.remote:
                    response_raw = requests.post(
                        f"{REMOTE_SERVER_ADDR}/task/eval",
                        json=AgentStudioEvalRequest(
                            procedures=task_config.eval_procedure,
                            as_kwargs=str(
                                jsonpickle.encode({"trajectory": agent.trajectory})
                            ),
                        ).model_dump(),
                    )
                    response = AgentStudioStatusResponse(**response_raw.json())
                    response = wait_finish(is_eval=True, response=response)
                    if not (
                        response.status == "finished"
                        and isinstance(response.message, dict)  # noqa: E501
                    ):
                        raise ValueError(f"Fail to evaluate task: {response.message}")
                    score, feedback = (
                        response.message["score"],
                        response.message["feedback"],
                    )
                else:
                    logger.info("Start evaluation")
                    score, feedback = evaluators(task_config.eval_procedure)

                scores[task_config.task_id] = score
                if score == 1.0:
                    logger.info(f"[Result] (PASS): {feedback}")
                else:
                    logger.info(f"[Result] (FAIL): {feedback}")

                if not args.no_log:
                    task_result_path = results_dir / task_config.task_id
                    export_trajectory(
                        task_config=task_config,
                        trajectory=agent.trajectory,
                        path=task_result_path,
                        score=score,
                        feedback=feedback,
                        token_count=agent.get_token_count(),
                        time_cost=stop_time - start_time,
                        video_meta=video_meta,
                    )
            except Exception as e:
                import traceback

                logger.error(f"[Unhandled Error] {repr(e)}]")
                traceback.print_exc()
            finally:
                # Clean up
                if task_config.cleanup_procedure is not None:
                    if args.remote:
                        response_raw = requests.post(
                            f"{REMOTE_SERVER_ADDR}/task/reset",
                            json=AgentStudioResetRequest(
                                procedures=task_config.cleanup_procedure
                            ).model_dump(),
                        )
                        response = AgentStudioStatusResponse(**response_raw.json())
                        response = wait_finish(is_eval=False, response=response)
                        assert (
                            response.status == "finished"
                            and response.content == "success"
                        ), f"Fail to reset task: {response.message}"
                    else:
                        evaluators = evaluator_router(task_config)
                        evaluators.reset(task_config.cleanup_procedure)

        agent.close()
        logger.info(
            f"Average score: {sum(scores.values())}/{len(scores)}="
            f"{sum(scores.values()) / max(len(scores), 1)}"
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        if interface is not None:
            interface.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--agent", type=str, default="direct", help="Agent type")
    parser.add_argument("--task_configs_path", type=str, help="Path to the task config")
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Path to save the logs",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Run with rendering (should be enabled for visual tasks)",
    )
    parser.add_argument("--remote", action="store_true", help="Run in remote mode")
    parser.add_argument(
        "--window_width", type=int, default=800, help="Width of the window"
    )
    parser.add_argument(
        "--window_height", type=int, default=600, help="Height of the window"
    )
    parser.add_argument(
        "--need_human_confirmation",
        action="store_true",
        help="Need human confirmation for actions",
    )
    parser.add_argument(
        "--use_time_limit", action="store_true", help="Use time limit for tasks"
    )
    parser.add_argument(
        "--ignore_finished", action="store_true", help="Only evaluate unfinished tasks"
    )
    parser.add_argument("--no_log", action="store_true", help="Do not log the results")
    parser.add_argument("--env_server_addr", type=str, default="127.0.0.1", help="Environment server address")
    parser.add_argument("--env_server_port", type=int, default=8000, help="Environment server port")
    parser.add_argument("--vnc_port", type=int, default=5900, help="VNC port")
    parser.add_argument("--vnc_password", type=str, default="123456", help="VNC password")
    args = parser.parse_args()
    logger.info(f"Running with args: {args}")
    assert args.task_configs_path is not None, "Task config is not set."

    config.remote = args.remote
    config.headless = not args.render
    config.need_human_confirmation = args.need_human_confirmation
    if config.remote:
        config.env_server_addr = args.env_server_addr
        config.env_server_port = args.env_server_port
        config.vnc_port = args.vnc_port
        config.vnc_password = args.vnc_password

    # Ensure a second screen is available.
    app = QApplication(sys.argv)
    screens = QApplication.screens()
    if not args.remote and len(screens) < 2:
        raise RuntimeError("A second screen is required for local annotation.")

    if not args.render:
        interface = NonGUI(
            args=args,
            remote=args.remote,
            window_width=args.window_width,
            window_height=args.window_height,
        )
        eval(args, interface)
    else:
        try:
            # Create the main interface.
            interface = GUI(
                args=args,
                remote=args.remote,
                window_width=args.window_width,
                window_height=args.window_height,
            )
            interface.resize(args.window_width, args.window_height)

            if not args.remote:
                # Move window to the second screen
                second_screen = screens[1]
                geometry = second_screen.geometry()
                interface.move(geometry.topLeft())
            interface.show()

            sys.exit(app.exec())
        except asyncio.exceptions.CancelledError:
            sys.exit(0)


if __name__ == "__main__":
    main()
