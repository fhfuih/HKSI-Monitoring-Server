# Athletes' Physiological Monitoring (with HKSI): Backend Server

This is the simple backend server as a broker between the frontend (facial image stream) and the prediction models (several independent algorithms in Python)

## Set up and Run

### Two versions

There are two versions. `main.py` is an old version using websocket (the old "video tranmission protocol"). `webetc.py` is a new version using webrtc (the more efficient but complex video transmission protocol).

I use python 3.9. Not sure about other versions, but should work. Note that python 3.12 doesn't have certain optional WebRTC dependency (but any version between 3.8 and 3.11 has), so there will be a warning when running `webrtc.py`.
For `main.py`, 3.12 should also work.

**The `webrtc.py` version still requires some tuning. For ML developers, currently you can test using the `main.py` version.**

### Installing Dependencies

For `main.py`, install the following packages
* python-socketio (5.x, which is latest), aiohttp, aiodns, brotli
* pillow
* python-dotenv

For `webrtc.py`, install the packages listed in `pyproject.toml`. If you use the Poetry software to manage Python environments, you can directly call `poetry install` to load the `pyproject.toml` file.

### File structure

* The main file is `main.py` (or `webrtc.py`, which is still not finished and not plugged into ML models). Run `main.py` to start a WebSocket server that listens to frontend messages and passes them to ML models.
* `mock_model.py` is a mock of ML models, which is intended to be time consuming.
* `test.py` is a mock client (frontend). After the server is running, run this file in parallel to send five frames consecutively with random interval. The frames are simply black text on a solid color square. **You can change this `test.py` to actually read a sample video's frames and send them.**

## For ML Developers

* One can configure the host address and port of the WebSocket server.
* One can add concrete ML model implementations that resembles the mock ones
    * The function signature (parameters and return) should look like the mock ones. Docs are at `mock_model.py`
    * ML model function *do not* need to implement concurrency. They are handled in `main.py`.
    * ML model function **do** need to implement any behaviors when receiving a new image frame. For example, return nothing on the first few passes and meanwhile cache the images for those passes.
    * Add the functions to the `PREDICTIONS` variable in `main.py`. The order of this list matters in that: if two prediction functions contain a same key in the returning dict, the latter one will override the former one.
    * **You may return something in `end` function (like a final average data of the current prediction session)**, even if the current type notation says it returns None. I will handle the changes if you do so.
* When I finish `webrtc.py`, I will plug ML models into the new WebRTC version server, and there should be no changes for the actual ML implementations.

## For Frontend Developers

### Websocket Message Protocol

* Frontend -> server: event `session_start`
    * Event argument: raw byte array of 64 bits (8B) with the following components:
        * timestamp in milliseconds: 64 bits (8B), unsigned int, little endian
    * With ack: JSON string `{"success": true}` or `{"success": false, "error": "..."}`
* Frontend -> server: event `session_end`
    * Event argument: raw byte array of 64 bits (8B) with the following components:
        * timestamp in milliseconds: 64 bits (8B), unsigned int, little endian
    * With ack: JSON string `{"success": true}` or `{"success": false, "error": "..."}`
* Frontend -> server: event `frame`
    * Event argument: raw byte array with the following components:
        * timestamp in milliseconds: 64 bits (8B), unsigned int, little endian
        * image in PNG format: arbitrary length, PNG file, default to big endian (PNG standard requires big endian)
    * With ack: JSON string `{"success": true}` or `{"success": false, "error": "..."}`.
    **The ack happends immediately when the current frame is scheduled. It is not guaranteed whether the current frame is processed.**
* Server -> frontend: event `prediction`
    * Event argument: JSON string in the following format:
        * `{"some key in string": some value in arbitrary data type}`
    * Without ack
