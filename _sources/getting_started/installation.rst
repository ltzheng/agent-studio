.. _installation:

Installation
============

Setup Environment
-----------------

Install requirements::

    apt-get install gnome-screenshot xclip xdotool  # If using Ubuntu 22.04
    conda create --name agent-studio python=3.11 -y
    conda activate agent-studio
    pip install -r requirements.txt
    pip install -e .

This command will download the task suite and agent trajectories from `Huggingface <https://huggingface.co/datasets/Skywork/agent-studio-data>`_ (you may need to `configure huggingface and git lfs <https://huggingface.co/docs/hub/en/repositories-getting-started#cloning-repositories>`_).

::

    git submodule update --init --remote --recursive

Setup API Keys
--------------

Please refer to the `doc <https://skyworkai.github.io/agent-studio/getting_started/setup_api_keys.html>`_ for detailed instructions.

Setup Docker
------------

This step is optional, only for running tasks with GUI in a docker container.

Build Docker image::

    docker build -f dockerfiles/Dockerfile.ubuntu.amd64 . -t agent-studio:latest
