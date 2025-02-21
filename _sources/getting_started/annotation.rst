.. _annotation:

Record GUI Grounding Dataset
==============

Using the annotator to record human demonstrations, you can easily create a dataset for imitation learning. The annotator **only supports recording remotely with GUI**, so remember to set Set ``headless=True`` and ``remote=True``.

Start the annotator:

.. code-block:: bash

    python run.py --mode annotate

The annotator is a simpler version of the Human Recorder. It only supports recording one mouse action for each task. The recorded data will be saved to the ``data/trajectories/annotate`` folder. Usage of the annotator is shown below:

1. Select an existing task or create a new task (**You must select a visual task**), and click the "Save Task Config/Start Recording" button to enter the annotator screen. (This step is the same as the recorder.)
2. The task and remote environment will be reset before the recording starts. After the task is reset, you can draw the bounding box on the VNC screen by mouse. Right click to give up drawing the bounding box.
3. Select the mouse action type on the "Annotation Panel". The mouse action types include "left click", "double click", "right click", and "middle click". After selecting the mouse action type, you **must** click the "Step Action" button to record the mouse action and annotation. When finished, you can click the "Save" button to save the trajectory.
