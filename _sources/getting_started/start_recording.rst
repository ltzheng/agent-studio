.. _start_recording:

Record Dataset
==============

Start recording::

    python run.py --mode record

The first screen is a task configuration interface, where you can create a new task or load an existing task. The task configuration interface is shown below:

.. image:: ../assets/imgs/recorder_task_config.jpg

You can choose to record an existing task or create a new task. The following are the steps to record a task:

Record an existing task:

#. Choose task from the top right list.
#. Click the "Save Task Config/Start Recording" button to start recording.

.. image:: ../assets/imgs/recorder_choose_existing.jpg

Create & Record new task:

#. Input the task instruction.
#. Select the task type (whether is a visual task or not).
#. Select the evaluator from the dropdown list.
#. Select the evaluator methods from the list table. A single click will display the method description in "Docs" and a double click will show the method example JSON snippet in "JSON format preview".
#. Edit the "Evaluation Steps" input box, which should be a list of steps to evaluate the task. The format should match the "evals" field in the task configuration JSON format.
#. Click the "Save Task Config/Start Recording" button to start recording.

.. image:: ../assets/imgs/recorder_create_new.jpg

The recording interface is shown below:

.. image:: ../assets/imgs/recorder_record.jpg

The recording interface is divided into three parts: the left part is the VNC window, task configuration created in the previous step is displayed in the middle part, and the right part is the "Action" panel. To record a task, you need to perform the following steps:

1. Input actions (currently is Python code) in the "Action" panel.
2. Click the "Step Action" button to execute the actions.
3. See the result in the VNC window and "Runtime Response" panel.
4. Repeat the above steps until the task is completed.
5. Click the "Save" button to save the recording.
