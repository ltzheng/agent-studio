# Real-world online benchmarks

We provide 205 single-API, single-GUI, and compositional tasks for online benchmarking. We provide a Docker image for reproducible and reliable online benchmarks. Our tools also allow for convenient task customization.

**Before You Start:** You should note that agents may do some **non-reversible actions**, such as deleting files, creating files, running commands, and deleting Google Calendar events. Please make sure you have backups of your data. Some tasks may require you to provide API keys. Before running the tasks, **please make sure the account doesn't have important data.**

## Task Preparation

1. Download the task files (`benchmark_task_files.tar.gz`) from [this link](https://drive.google.com/drive/folders/1XKDXwdWODCB2e80gflAgZiiBICqbgdeB) and extract `files/*` to `eval_online_benchmarks/data/`.
2. The tasks are located in `eval_online_benchmarks/tasks` and categorized into `single_api`, `single_gui`, and `compositional`.

## Google Account Setup (Optional)

Google Workspace tasks require Google API usage. Kindly enable Google APIs, configure OAuth, download the credentials following instructions [here](https://developers.google.com/docs/api/quickstart/python#set_up_your_environment). In Google Cloud Console, go to `APIs & Services` -> `Credentials` -> `Create Credentials` -> `OAuth client ID` to create a new credential and download the `credentials.json` file to local disk. Then specify the credential path in `agent_studio/config/api_key.json`. When you run the benchmark for the first time, you should authorize all Google services before experiments by running `python scripts/setup_api_keys.py`. You will be prompted to visit several URLs to authorize Google Docs, Drives, etc. The corresponding token json files like `docs_token.json` will be saved in `agent_studio/config`.

### Google Calendar Setup

For a clean start, you should create a new calendar for the benchmark. You can do so by going to [Google Calendar](https://calendar.google.com/calendar) and create a new calendar. Click the three dots in the right side of the calendar name and select `Settings and sharing`. Scroll down to `Integrate calendar` and copy the `Calendar ID`. Specify the calendar ID in `agent_studio/config/api_key.json` as `google_calendar_id`.

### Gmail Setup

For a clean start, you should specify a temporary email address for the benchmark. You can do so by going to [Temp-Mail](https://temp-mail.org/en/) and get a temporary email address. Specify the email address in `agent_studio/config/api_key.json` as `gmail_recipient`.

> If you want to benchmark Google Workspace, you need to do the above steps before running the Evaluation scripts.

## Setup Docker Image

First, please follow the instructions in the [README](../README.md) to install the AgentStudio python package and setup API keys.

We provide a lightweight Dockerfile of Ubuntu 22.04 for reproducible and reliable online benchmarks.

```bash
docker build -f dockerfiles/Dockerfile.ubuntu22.04.amd64 . -t agent-studio:latest
```

Run Docker:

```bash
docker run -d -e RESOLUTION=1024x768 -p 6080:80 -p 5900:5900 -p 8000:8000 -e VNC_PASSWORD=123456 -v /dev/shm:/dev/shm -v ${PWD}/scripts/agent_server.py:/home/ubuntu/agent_studio/scripts/agent_server.py:ro -v ${PWD}/agent_studio/envs:/home/ubuntu/agent_studio/agent_studio/envs:ro -v ${PWD}/agent_studio/utils:/home/ubuntu/agent_studio/agent_studio/utils:ro -v ${PWD}/agent_studio/agent:/home/ubuntu/agent_studio/agent_studio/agent:ro -v ${PWD}/agent_studio/config:/home/ubuntu/agent_studio/agent_studio/config -v ${PWD}/eval_online_benchmarks/data:/home/ubuntu/agent_studio/data:ro agent-studio:latest
```

> You can also replace `-d` to `-it` to use interactive mode. If successful, you should see logs with a bunch of success followed by `INFO  Listening on http://localhost:6079` in the output.

You can browse `http://127.0.0.1:6080` to interact with the remote machine through a browser. The port `6080`, `5900`, and `8000` are exposed for noVNC, VNC server, and AgentStudio HTTP, respectively.

## Start Evaluation

### Single-API Tasks

Start benchmarking:

```bash
# Use docker environment, specify --remote
as-online-benchmark --task_configs_path eval_online_benchmarks/tasks --model gpt-4o-2024-08-06 --remote
as-online-benchmark --task_configs_path eval_online_benchmarks/tasks --model gemini-1.5-flash-001 --remote

# Run a specific task set, e.g., os
as-online-benchmark --task_configs_path eval_online_benchmarks/tasks/single_api/os --model gemini-1.5-flash-001 --remote
# Or google_docs
as-online-benchmark --task_configs_path eval_online_benchmarks/tasks/single_api/google_docs --model gemini-1.5-flash-001 --remote
```

> You can add `--use_time_limit` to run with time limit (specified in the task config) and `--ignore_finished` to recover from last run. You can also set `need_human_confirmation=True` in `agent_studio/config/config.py` to do safety check before each action execution. You can add `--help` to explore more args.

By default, you can check `logs` to see the full logs and result jsonl files.

### Single-GUI Tasks

```bash
as-online-benchmark --task_configs_path eval_online_benchmarks/tasks/single_gui/gimp --model gpt-4o-2024-08-06 --remote


as-online-benchmark --task_configs_path eval_online_benchmarks/tasks/basic/vscode/ --model gemini-1.5-flash-001 --remote --render
as-online-benchmark --task_configs_path eval_online_benchmarks/tasks/basic/vscode/ --model gemini-1.5-flash-001 --remote ...
```

We also provide more auto-evaluators on other applications in `agent_studio/envs/desktop_env/evaluators`, such as Telegram, Google Slides, etc.

### Compositional Tasks

```bash
as-online-benchmark --task_configs_path eval_online_benchmarks/tasks/compositional --model gemini-1.0-pro-001
```

## Human Validation & Evaluation

![](../docs/assets/onlinebenchmark_gui_1.png)

In order to 1) validate the correctness of task implementation or 2) evaluate human performance on these tasks, we provide this tool for human evaluation.

Step 1: Start the Agent Monitor GUI

Example command to start human evaluation on vscode tasks:

```bash
as-online-benchmark --task_configs_path eval_online_benchmarks/tasks/single_gui/vscode --agent human --remote --render --need_human_confirmation
```

Step 2: Open http://localhost:6080/ in the browser (or any VNC viewer)

Step 3: Double click a task in the Task Selection of the Agent Monitor, and click on the start button. You may be prompted to confirm some resetting actions. The script will popup "Confirm when you finish" after reset

Step 4: Complete the task in the browser (or VNC viewer). After finishing the task, you can confirm the popup message to see the evaluation result. If you find it hard to complete, click on the "reject" button

Step 5: Wait for the evaluation result on the right bottom of the Agent Monitor or the output from the terminal.

Step 6: Repeat step 3-5 for other tasks.

## Add more tasks

To add custom tasks for benchmarking agents in the wild, you can add a task.jsonl files according to ...

This guide provides instructions for creating a valid Task JSON file in accordance with the specified schema for task evaluation. The JSON file combines details about the environment and tasks, along with various parameters pertinent to the evaluation process.

### Task Structure

-   `task_id`: A unique identifier for the task.
-   `instruction`: The task instuction.
-   `visual`: (boolean) Whether the task requires visual output.
-   `max_steps`: (int) The maximum number of steps the agent can take.
-   `max_time`: (float) A time limit for the task.
-   `eval_procedure`: (list) Contains the evaluation procedure and the reference answers.
-   `reset_procedure`: (optional list) A list of actions to reset environment before the task.
-   `cleanup_procedure`: (optional list) A list of actions to clean up the environment after the task.

Example task:

```json
{
    "task_id": "uuid string",
    "instruction": "Task instruction for the agent to complete",
    "visual": false,
    "max_steps": 1,
    "max_time": 60,
    "eval_procedure": [
        {
            "evaluator": "evaluator1",
            "function": "function1",
            "params": {
                "param1": "value1"
            }
        }
    ],
    "reset_procedure": [
        {
            "evaluator": "evaluator2",
            "function": "function1",
            "params": {
                "param1": "value1",
                "param2": "value2"
            }
        }
    ],
    "cleanup_procedure": [
        {
            "evaluator": "evaluator3",
            "function": "function3",
            "params": {
                "param1": "value1"
            }
        }
    ]
}
```

## Debugging

### Do I need to rebuild the docker image after modifying the evaluation related code?

No, you don't need to rebuild the docker image. The agent studio code base is mounted into the docker image at `/home/ubuntu/agent_studio`. You can edit the code and restart the docker container to apply the changes. Or you can use `docker exec -it <container_id> /bin/bash` to enter the container and restart the server by executing `supervisordctl restart agent_server`.

### When should I rebuild the docker image?

You should rebuild the docker image when you modify the Dockerfile or the dependencies or `agent_studio/__init__.py`.

### After modifying the evaluation related code, how to apply the changes to the running docker container?

Since the agent studio code base is mounted into the docker image at `/home/ubuntu/agent_studio`. You can simply restart the docker container to apply the changes. Or if you don't want to restart the docker container, you can also use `docker exec -it <container_id> /bin/bash` to enter the container and restart the server by executing `supervisordctl restart agent_server`.
