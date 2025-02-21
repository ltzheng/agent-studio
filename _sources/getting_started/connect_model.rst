.. _connect_model:

Connect to Your Own Model
=========================

Run the Model Locally or via API
--------------------------------

Our platform allows you to connect to your own model. Models are stored in the ``agent_studio/llm`` directory. Each model is a Python class that inherits from the ``BaseModel`` class. You can integrate your model by creating a new model class that inherits from the ``agent_studio.llm.base_model.BaseModel``. The model class must implement the following methods:

1. ``name: str``:
    A unique name for the model. **The name should match the ``provider`` field in ``agent_studio/config/config.py``**.
2. ``compose_messages(self, intermedia_msg: list[dict[str, Any]]) -> Any:``:
    Convert the intermediate messages to the format that the model can accept. The returned value is only used for ``generate_response`` function.
3. ``generate_response(self, messages: list[dict[str, Any]], **kwargs) -> tuple[str, dict[str, int]]``:
    Generate the model's response to the given intermediate message. ``kwargs`` parameter is not used currently, but can be used in the future. The method should return two values:

    ``message: str``:
        The raw response message.
    ``info: dict[str, int]``:
        Additional information about the response. If you want to count token costs, you can store ``total_tokens`` in the ``info`` dictionary.


Run the Model on Remote Machine
--------------------------------

Maybe your model is too large to run on your local machine or there's no existing API to connect to it. In this case, you can run the model on a remote machine and connect to it via our model server. To do this, you need to use the ``agent_studio.llm.remote_model.RemoteProvider`` class and ``scripts/model_server`` script.

1. **Modify the model server script** so that it can return the raw message and info.
2. Run the model server on the remote machine with the following command:

    ```bash
    python scripts/model_server.py
    ```

3. The ``RemoteProvider`` sends the messages to the remote machine, runs the model, and returns the response. Modify the ``model_server`` parameter in ``agent_studio/config/config.py`` by setting the IP address and port of the remote machine.
