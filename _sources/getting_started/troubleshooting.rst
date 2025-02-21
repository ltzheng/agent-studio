.. _troubleshooting:

Troubleshooting
===============

``[Errno 10048]`` error while attempting to bind on address ('0.0.0.0', 8000): only one usage of each socket address (protocol/network address/port) is normally permitted

The port 8000 is already in use. Maybe there is another process running on the same port. Or the previous process is not closed properly. You can follow the following two solutions:

1. check the process running on the port by running the following command and kill it if necessary::

    sudo lsof -i:8000

2. If the port is used due to the previous process not closed properly, you can exit the GUI with ``Ctrl+C`` and run it again.

--------------------------------------------------------------------------------

If you enabled high DPI scaling, and the VNC window is beyond the screen, you may need to set the ``QT_AUTO_SCREEN_SCALE_FACTOR`` environment variable to ``0`` to disable high DPI scaling.
