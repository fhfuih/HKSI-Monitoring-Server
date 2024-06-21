from datetime import datetime
from io import BytesIO
import asyncio
import json
from typing import Any, List
from enum import Enum
import logging
import os.path
import ssl
import os

from dotenv import load_dotenv
from aiohttp import web
import socketio
from PIL import Image, UnidentifiedImageError

from models import BaseModel
import mock_model

# WebSocket server address
HOST = "0.0.0.0"
PORT = 8765

# list of prediction functions to run (e.g., HR is 1, Fatigue is 2)
MODELS: List[BaseModel] = [
    mock_model.MockModel1(),
    mock_model.MockModel2(),
]

# Logger setup. If file logger is needed, add a handler here.
logger = logging.getLogger("socketIO")
logging.basicConfig(level=logging.INFO)

# SSL keys for HTTPS
cert_path = os.path.join(os.path.dirname(__file__), "ssl", "cert.pem")
key_path = os.path.join(os.path.dirname(__file__), "ssl", "key.pem")

# ==========
# Server implemention starts here
# ==========

load_dotenv()
is_dev = os.getenv("DEPLOYMENT_ENV", "").lower().startswith("dev")

SessionState = Enum("SessionState", ["IDLE", "RUNNING"])

sio = socketio.AsyncServer(
    logger=logger,
    cors_allowed_origins=(
        "*" if is_dev else os.getenv("CORS_ALLOWED_ORIGINS").split(",")
    ),
)
app = web.Application()
sio.attach(app)

ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
ssl_context.load_cert_chain(certfile=cert_path, keyfile=key_path)


@sio.event
async def connect(sid: str, environ: dict, auth: dict):
    # username = authenticate_user(environ)
    key = auth.get("key", r"ye7B9xuvL%^%b@6C2RMZ")
    if key != os.getenv("SOCKETIO_PASSWORD"):
        raise ConnectionRefusedError("authentication failed")
    await sio.save_session(sid, {"state": SessionState.IDLE})
    logger.info("connect %s", sid)


@sio.event
async def disconnect(sid: str):
    state = await sio.get_session(sid)
    if state != SessionState.IDLE:
        timestamp = int(datetime.now().timestamp() * 1000)
        await end_all_models(sid, timestamp)
        logger.info("disconnect %s & end all running models for this session", sid)
    else:
        logger.info("disconnect %s", sid)


@sio.event
async def frame_start(sid: str, data: bytes = bytes()):
    if len(data) < 8:
        return error_ack("Data length is less than 8 bytes")

    session_data = await sio.get_session(sid)
    state = session_data["state"]
    if state != SessionState.IDLE:
        logger.debug(
            f"Session is already running: state {state} != {SessionState.IDLE}"
        )
        return error_ack(f"Session is already running: {state}")
    await sio.save_session(sid, {"state": SessionState.RUNNING})

    timestamp = get_timestamp(data)
    logger.debug(
        f"frame_start at {datetime.fromtimestamp(timestamp / 1000)}, sid {sid}"
    )

    await start_all_models(sid, timestamp)
    return '{"success": true}'


@sio.event
async def frame_end(sid: str, data: bytes = bytes()):
    if len(data) < 8:
        return error_ack("Data length is less than 8 bytes")

    session_data = await sio.get_session(sid)
    state = session_data["state"]
    if state != SessionState.RUNNING:
        return error_ack(f"Session is not running: {state}")
    await sio.save_session(sid, {"state": SessionState.IDLE})

    timestamp = get_timestamp(data)
    logger.debug(f"frame_end at {datetime.fromtimestamp(timestamp / 1000)}, sid {sid}")

    await end_all_models(sid, timestamp)
    return '{"success": true}'


@sio.event
async def frame(sid: str, data: bytes = bytes()) -> str:
    if len(data) < 8:
        return error_ack("Data length is less than 8 bytes")

    session_data = await sio.get_session(sid)
    state = session_data["state"]
    if state != SessionState.RUNNING:
        return error_ack(f"Session is not running: {state}")

    timestamp = get_timestamp(data)
    logger.debug(f"Received frame with length {len(data)} bytes")

    try:
        frame = get_image(data)
    except UnidentifiedImageError:
        return error_ack("Image is corrupted or not in PNG format")

    results = await process_frame_all_models(sid, timestamp, frame)
    result = {
        "frame": {"size": frame.size, "mode": frame.mode},
        "recv_ts": timestamp,
    }
    for r in results:
        if r is not None:
            result.update(r)

    logger.debug(
        f"Responding data at {datetime.fromtimestamp(timestamp / 1000)} with {result}"
    )
    return success_ack(result)


@sio.on("*")
async def any_event(event: str, sid: str, data: Any):
    logger.debug(f"Unregistered event {event} received with data {data} from {sid}")
    return error_ack("Unregistered event")


async def start_all_models(sid: str, timestamp: int):
    await asyncio.gather(
        *[asyncio.to_thread(model.start, sid, timestamp) for model in MODELS]
    )


async def end_all_models(sid: str, timestamp: int):
    await asyncio.gather(
        *[asyncio.to_thread(model.end, sid, timestamp) for model in MODELS]
    )


def process_frame_all_models(
    sid, timestamp, frame
) -> asyncio.Future[list[dict | None]]:
    return asyncio.gather(
        *[
            asyncio.to_thread(model.forward_single_frame, sid, frame, timestamp)
            for model in MODELS
        ]
    )


def get_timestamp(data: bytes) -> int:
    """extract the first 64 bits of `data` as an unsigned integer

    Parameters
    ----------
    data : bytes
        The binary uint64 data.

    Returns
    -------
    int
        Timestamp **in milliseconds**.
        This is *not* the same as python built-in libraries' default timestamp (which is in seconds).
    """
    timestamp = data[:8]
    return int.from_bytes(timestamp, byteorder="little", signed=False)


def get_image(data: bytes) -> Image.Image:
    """extract the remaining bytes of `data` as a png image

    Parameters
    ----------
    data : bytes
        The binary PNG image data.

    Returns
    -------
    Image.Image
        A PIL Image object.

    Raises
    ------
    UnidentifiedImageError
        If the image is corrupted or not in PNG format.
    """
    frame = data[8:]
    frameIO = BytesIO(frame)
    frame = Image.open(frameIO, formats=("PNG",))
    frameIO.close()
    return frame


def success_ack(data: dict) -> str:
    return json.dumps({"success": True, **data}, ensure_ascii=False)


def error_ack(msg: str) -> str:
    return '{"success": false, "error": "%s"}' % msg


if __name__ == "__main__":
    if is_dev:
        logger.setLevel(logging.DEBUG)

    print(
        f"Starting with logger level {logging.getLevelName(logger.getEffectiveLevel())}"
    )

    web.run_app(app, host=HOST, port=PORT, ssl_context=ssl_context)
