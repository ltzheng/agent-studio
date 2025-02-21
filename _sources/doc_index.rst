Welcome to AgentStudio!
======================================

.. _Project Page: https://skyworkai.github.io/agent-studio/

.. image:: https://img.shields.io/badge/python-3.11-blue.svg
    :target: https://www.python.org/downloads/release/python-3117/
    :alt: Python 3.11

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: Code style: black

.. image:: https://img.shields.io/badge/License-AGPL%20v3-blue.svg
    :target: https://www.gnu.org/licenses/agpl-3.0
    :alt: License: AGPL v3

.. image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
    :target: https://pre-commit.com/
    :alt: pre-commit

This is the documentation of AgentStudio environments and toolkits. For more info about the benchmark suites and the leaderboard, please see our `Project Page`_.


AgentStudio environments and toolkits cover the entire lifespan of
building computer agents that can interact with everything on digital worlds. Here, we open-source the beta of environment implementations, benchmark suite, data collection pipeline, and graphical interfaces to promote research towards generalist computer agents of the future.

.. image:: ./assets/imgs/overview.png

AgentStudio provides unified observation and action spaces aligned with how humans interact with computers, allowing agent evaluation and data collection on any human-performed task. This feature drastically expands the potential task space. Therefore, AgentStudio can facilitate the development and benchmark of agents that generalize across diverse real-world use cases. In comparison, most previous environments tailored the observation and action spaces solely for specific domains, such as web operations or API calls.

.. image:: ./assets/imgs/agent_space.jpg

Contributing
------------

We plan to expand the collection of environments, tasks, and data over time. Contributions and feedback from everyone on how to make this into a better tool are more than welcome, no matter the scale. Please check out `CONTRIBUTING.md`_ for how to get involved.

.. image:: ./assets/imgs/annotation_example.jpg

.. _`CONTRIBUTING.md`: CONTRIBUTING.md

.. toctree::
   :maxdepth: 4

   getting_started/installation

   getting_started/setup_api_keys

   getting_started/evaluate_agents

   getting_started/create_agents

   getting_started/connect_model

   getting_started/upload_results

   getting_started/start_recording

   getting_started/annotation

   getting_started/troubleshooting
