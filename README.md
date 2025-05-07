# Athletes' Physiological Monitoring (with HKSI): Backend Server

This is the simple backend server as a broker between the frontend (facial image stream) and the prediction models (several independent algorithms in Python)

## Set up and Run

### Installing Dependencies

Install the packages listed in `pyproject.toml`.

If you use the [uv](https://docs.astral.sh/uv/guides/projects/) program to manage Python environments (I recommend), you can simply do:
```bash
cd <project root folder>
uv sync  # It auto-detects pyproject.toml
```

It will create an virtual environment at `.venv` inside the root folder. You can either
* manually activate this venv, and then run `python main.py`, or
* don't activate it, but run `uv run main.py` inside the project folder. (So that uv will use the venv for you.)

If you don't use uv, I recommend you to create a new venv first and activate it.
Then, `pip install <...>` all the packages listed in `pyproject.toml`.

### File structure

* The main file is `main.py`. Run `main.py` to start a WebRTC server that listens to frontend messages and passes them to ML models.
* `models` folder contains a `BaseModel` definition that ML developers should inherit. `mock_model_1.py` and `mock_model_2/` are two example implementations. (You can see that a model can be defined in a single file or in a folder with other helper files.) The two mock models are intended to be time consuming.
* `test.py` is a mock client (frontend). After the server is running, run this file in parallel to read `sample-video.mp4` and stream it to the server as if the camera view.

## For ML Developers

* One can add concrete ML model implementations that resembles the mock ones
    * The function signature (parameters and return) should look like the mock ones. Docs are at `models/base_model.py`.
    * ML model function *do not* need to consider concurrency/parallel programming. They are handled in `main.py`.
    * ML model function **do** need to implement any behaviors when receiving a new image frame. For example, return nothing on the first few passes and meanwhile cache the images for those passes.
    * Add your model subclasses to the `MODELS` list in `main.py`. The order of this list matters in that: if two model outputs contain a same key in the returning dict, the latter one will override the former one.
    * **You are suggested to return something in `end` function (like a final average data of the current prediction session)**.
    * **The `end` function may not contain a timestamp. It does not contain any frame data anyway.**

## For Frontend Developers

### WebRTC Protocol

* SDP
    * Format: JSON `{"sdp": "<The standard SDP string>", "type": "<SDP type name>"}`.
    * Protocol: HTTP(S) at `/offer`. Currently, only support `"offer"` type in the request payload, and `"answer"` in the response.
* Paradigm: The client (frontend) is the "offer" side of the WebRTC PeerConnection. The client makes the offer, connects to the server, and actively disconnects after 30 seconds.
    * Setting up a video track => ML prediction session start
    * Incoming frame => ML prediction frame input
    * Disconnecting => ML prediction session end

<!-- * Frontend -> server: event `session_start`
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
    * Without ack -->
