# Athletes' Physiological Monitoring (with HKSI): Backend Server

This is the simple backend server as a broker between the frontend (facial image stream) and the prediction models (several independent algorithms in Python)

## Set up and Run

I use python 3.12. Not sure about versions < 3.12.

### Dependencies

Install dependencies and (preferrably) create a new virtual environment

> The following example uses `conda`.
> If you use `venv` (for virtual environments) and `pip` (for dependency installation),
> there are similar commands.

```sh
conda create -n "your_env_name" --file requirements.txt
conda activate "your_env_name"
```

Or manually install the following dependencies, and don't care about the `requirements.txt` file
* python-socketio (5.x, which is latest), aiohttp, aiodns, brotli
* pillow
* python-dotenv

### Run

* Ask for the OneDrive share folder link with your collaborator. There, go to `Code/Server/` and copy everything there to the root folder of this code repository. For example, there are (including but not limited to)
    * `.env`
    * `ssl/cert.pem`
    * `ssl/key.pem`
* The main file is `main.py`. Run it to start a WebSocket server that listens to frontend messages and passes them to ML models.
* `mock_model.py` is a mock of ML models, which can be time consuming.
* `test.py` is a mock client (frontend). After the server is running, run this file in parallel to send five images consecutively with random interval.

## For ML Developers

* One can configure the host address and port of the WebSocket server.
* One can add concrete ML model implementations that resembles the mock ones
    * The function signature (parameters and return) should look like the mock ones. Docs are at `mock_model.py`
    * ML model function *do not* need to implement concurrency. They are handled in `main.py`.
    * ML model function **do** need to implement any behaviors when receiving a new image frame. For example, do not return on the first few passes and meanwhile cache the images for those passes.
    * Add the functions to the `PREDICTIONS` variable in `main.py`. The order of this list matters in that: if two prediction functions contain a same key in the returning dict, the latter one will override the former one.

## For Frontend Developers

### SocketIO Message Protocol

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
