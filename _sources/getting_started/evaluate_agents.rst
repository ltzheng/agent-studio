.. _evaluate_agents:

Evaluate Agents
===============

You may modify `config.py <agent_studio/config/config.py>`_ to configure the environment.

* ``headless``: Set to ``False`` for GUI mode or ``True`` for CLI mode.
* ``remote``: Set to ``True`` for running experiments in the docker or remote machines. Otherwise, experiments will run locally.
* ``task_config_paths``: The path to the task configuration file.

Local + Headless
----------------

Set ``headless = True`` and ``remote = False``. This setup is the simplest, and it is suitable for evaluating agents that do not require GUI (e.g., Google APIs).

Start benchmarking::

    python run.py --mode eval

Remote + GUI
------------

Set ``headless = False`` and ``remote = True``. This setup is suitable for evaluating agents in visual tasks. The remote machines can either be a docker container or a remote machine, connected via VNC remote desktop.

Run Docker (optional)
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    docker run -d -e RESOLUTION=1024x768 -p 5900:5900 -p 8000:8000 -e VNC_PASSWORD=123456 -v /dev/shm:/dev/shm -v ${PWD}/agent_studio/config/:/root/agent_studio/agent_studio/config/:ro agent_studio:latest

Start benchmarking::

    python run.py --mode eval
