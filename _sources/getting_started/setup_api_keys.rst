.. _setup_api_keys:

Setup API Keys
==============

All confidential API keys should be stored in `api_key.json` <agent_studio/config/api_key_template.json>, e.g., OpenAI API key, Gemini API key, Google credentials, etc. First, you need to rename the ``api_key_template.json`` to ``api_key.json``:

.. code-block:: bash

   mv agent_studio/config/api_key_template.json agent_studio/config/api_key.json

You need to obtain the API keys and add them to `api_key.json` <agent_studio/config/api_key_template.json>.

Google Workspace
----------------

`Enable Google APIs, configure OAuth, download the credentials` <https://developers.google.com/docs/api/quickstart/python#set_up_your_environment>, the credentials should be saved as ``credentials.json`` in the `agent_studio/config` <agent_studio/config> directory. Google services need the user to log in manually. Run ``python scripts/setup_api_keys.py`` to finish setup.

This library may modify your Google Calendar. For safety, it is recommended to create a new calendar, `obtain the calendar ID` <https://it.umn.edu/services-technologies/how-tos/google-calendar-find-your-google>, and add it to the ``google_calendar_id`` field in `api_key.json` <agent_studio/config/api_key_template.json>.

Telegram
--------

The Telegram evaluator is based on `Pyrogram` <https://docs.pyrogram.org/>. To enable it, `obtain the Telegram API key` <https://core.telegram.org/api/obtaining_api_id>, and add ``api_id`` and ``api_hash`` to the ``telegram_api_id`` and ``telegram_api_hash`` fields in `api_key.json` <agent_studio/config/api_key_template.json>, respectively.
