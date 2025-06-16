<h1 align="center">
AgentStudio
</h1>

<p align="center">
<a href='https://arxiv.org/abs/2403.17918'><img src='https://img.shields.io/badge/arXiv-2403.17918-b31b1b.svg'></a>
<a href='https://ltzheng.github.io/agent-studio'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href="https://www.python.org/downloads/release/python-3117/"><img alt="Python 3.11" src="https://img.shields.io/badge/python-3.11-blue.svg"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<!-- <a href="https://mypy-lang.org/"><img src="https://www.mypy-lang.org/static/mypy_badge.svg" alt="Checked with mypy"></a> -->
<a href="https://www.gnu.org/licenses/agpl-3.0"><img src="https://img.shields.io/badge/License-AGPL%20v3-blue.svg" alt="License: AGPL v3"></a>
<a href="https://pre-commit.com/"><img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white" alt="pre-commit"></a>
</p>

![](docs/assets/overview.png)

AgentStudio is **a trinity of environments, tools, and benchmarks** for general virtual agents to interact with any computer software. AgentStudio targets the desiderata for robust, general, and open-ended virtual agents by providing:
1. **A lightweight, interactive environment** with highly **generic observation and action spaces**, e.g., video observations and GUI/API actions
2. **Tools for creating online benchmark tasks, annotating GUI elements, and labeling actions in videos**
3. **Online benchmark tasks** that evaluate both GUI interactions and function calling with **auto-evaluation** and language feedback
4. **Three benchmark datasets**: GroundUI, IDMBench, and CriticBench, for fundamental agent abilities, including GUI grounding, learning from videos, and success detection


Comparisons with existing work:

![](docs/assets/comparison.png)

## News

- **Oct 3, 2024**: Released the <a href='https://arxiv.org/abs/2403.17918'>arXiv paper v2</a> and a full version of AgentStudio, including comprehensive documentation, complete tasks, and datasets!!
- **Aug 18, 2024**: Major update to clean up the codebase and datasets.
- **Mar 30, 2024**: Released the beta version of AgentStudio.

## Installation

Install requirements:

```bash
git clone https://github.com/ltzheng/agent-studio.git
apt-get install gnome-screenshot xclip xdotool  # If using Ubuntu 22.04
conda create --name agent-studio python=3.11 -y
conda activate agent-studio
pip install -e '.[client]'
```

All confidential API keys should be stored in `agent_studio/config/api_key.json`, e.g., OpenAI API key, Claude API key, Gemini API key, etc. We have provided an example config in `agent_studio/config/api_key_template.json`.

## AgentStudio Overall Benchmark Tasks

![](docs/assets/agent_space.jpg)

AgentStudio provides the most generic observation and action spaces, which significantly expands the task space, allowing for developing and evaluating agents in real-world settings. We introduce a benchmark suite consisting of 205 tasks. These tasks span API usages such as terminal and Gmail and GUI software like VS Code. Please find more in [eval_online_benchmarks/README.md](eval_online_benchmarks/README.md). The task-related files are available at our <a href='https://ltzheng.github.io/agent-studio'>project page</a>.

## AgentStudio Datasets Decompose Agent Abilities

To gain deeper insights into agent capabilities beyond the overall performance measured by online benchmark tasks, we develop three datasets using AgentStudio: GroundUI, IDMBench, and CriticBench. These datasets target general UI grounding, learning from videos, and success detection. More details are provided in [eval_agent_desiderata/README.md](eval_agent_desiderata/README.md). All data are available at our <a href='https://ltzheng.github.io/agent-studio'>project page</a>.

## AgentStudio Tools

To facilitate the development and evaluation of agents within the AgentStudio environment, we provide three tools for:
- Benchmark task creation and validation
- Step-level GUI element annotation
- Trajectory-level video-action recording and refinement

These tools, combined with the realistic environment of AgentStudio, contribute to the generation of rich, structured data for training and evaluating agents. Please refer to [docs/annotate_ground_ui.md](docs/annotate_ground_ui.md) for the GUI annotation tool, [agent_studio/recorder/README.md](agent_studio/recorder/README.md) for the video-action annotation tool, and [eval_online_benchmarks/README.md](eval_online_benchmarks/README.md) for the task creation/validation.

## Contributing

Contributions and feedback from everyone on how to make this into a better tool are more than welcome. Please check out [CONTRIBUTING.md](CONTRIBUTING.md) for how to get involved.

## Acknowledgement

We would like to thank the following projects for their inspiration and contributions to the open-source community: [Open Interpreter](https://github.com/KillianLucas/open-interpreter), [WebArena](https://github.com/web-arena-x/webarena), [Cradle](https://baai-agents.github.io/Cradle/), [Synapse](https://ltzheng.github.io/Synapse/), [SeeClick](https://github.com/njucckevin/SeeClick), [ScreenAgent](https://github.com/niuzaisheng/ScreenAgent), [OSWorld](https://github.com/xlang-ai/OSWorld), etc.

## Citation

If you find AgentStudio useful, please cite our paper:

```bibtex
@inproceedings{zheng2024agentstudio,
  title={Agent{S}tudio: A Toolkit for Building General Virtual Agents},
  author={Zheng, Longtao and Huang, Zhiyuan and Xue, Zhenghai and Wang, Xinrun and An, Bo and Yan, Shuicheng},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2024}
}
```
