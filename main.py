"""
SocketIO server implementation for prediction models on the cloud.

Note: According to the official document,
SocketIO event handlers' first argument is called `sid` which stands for "session id",
and the "session" refers to the connection between the client and the server.
However, in this program, we call it `cid` which stands for "connection id".
We reuse the term "session" to represent the start and end of a prediction session ---
a consecutive series of frames of the same user that are processed by the models.
The variable `sid` is also used when the latter meaning is intended.
"""

from datetime import datetime
import asyncio
import json
from typing import Any
from enum import Enum
import logging
import os.path
import ssl
import os

from dotenv import load_dotenv
from aiohttp import web
import socketio
from PIL import UnidentifiedImageError

from data_structure import FrameData, get_timestamp
from models import BaseModel
import mock_model

# WebSocket server address
HOST = "0.0.0.0"
PORT = 8765

# list of prediction functions to run (e.g., HR is 1, Fatigue is 2)
MODELS: list[BaseModel] = [
    mock_model.MockModel1(),
    mock_model.MockModel2(),
]

# Logger setup. If file logger is needed, add a handler here.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HKSI")

# SSL keys for HTTPS
cert_path = os.path.join(os.path.dirname(__file__), "ssl", "cert.pem")
key_path = os.path.join(os.path.dirname(__file__), "ssl", "key.pem")

# ==========
# Server implemention starts here
# ==========

# Load environment variables
load_dotenv()
is_dev = os.getenv("DEPLOYMENT_ENV", "").lower().startswith("dev")

# Server setup
SessionState = Enum("SessionState", ["IDLE", "RUNNING"])

sio_logger = logging.getLogger("HKSI Socket.IO")
sio_logger.setLevel(logging.WARNING)

sio = socketio.AsyncServer(
    logger=sio_logger,
    cors_allowed_origins=(
        "*" if is_dev else os.getenv("CORS_ALLOWED_ORIGINS").split(",")
    ),
)
app = web.Application()
sio.attach(app)

ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
ssl_context.load_cert_chain(certfile=cert_path, keyfile=key_path)


# asyncio producer-consumer pattern
# queue_by_cid: Annotated[
#     dict[str, asyncio.Queue[FrameData]],
#     """
# A dictionary that maps connection id to a queue of FrameData objects.
# For a connection, the queue is *NOT* created at connection.
# Rather, it is created at the session_start event (i.e., when a *session* starts).
# Since all sessions via this connection share the same connection id,
# when latter sessions start, they can check if previous sessions' data is cleared.
# """,
# ] = {}


async def consume_frame(queue: asyncio.Queue[FrameData]):
    """
    This is the worker task that consumes the frame data from the queue.
    It should be cancelled somewhere at session_end event.
    In case the server disconnects while streaming frames (i.e., session_end is not emitted),
    it aborts, reads no more items in the queue, and the reference to the queue is released.
    """
    while True:
        frame_data = await queue.get()

        # Check if connection is still there
        # try:
        #     await sio.get_session(frame_data.cid)
        # except KeyError:
        #     logger.warning(
        #         "Connection %s lost while consuming its frames. Abort consumption.",
        #         frame_data.cid,
        #     )
        #     queue.task_done()
        #     return

        result_list = await process_frame_all_models(
            frame_data.sid, frame_data.timestamp, frame_data.frame
        )
        result = {
            "frame": {"size": frame_data.frame.size, "mode": frame_data.frame.mode},
            "recv_ts": frame_data.timestamp,
        }
        for r in result_list:
            if r is not None:
                result.update(r)

        await sio.emit("prediction", success_ack(result), to=frame_data.cid)
        queue.task_done()
        del frame_data


@sio.event
async def connect(cid: str, environ: dict, auth: dict):
    key = auth.get("key")
    if key != os.getenv("SOCKETIO_PASSWORD"):
        logger.warning("Authentication failed for %s. Auth payload is %s", cid, auth)
        raise ConnectionRefusedError("authentication failed")

    # if cid in queue_by_cid:
    #     logger.warning("Connection data already exists for %s. Removing them.", cid)
    #     del queue_by_cid[cid]
    # queue_by_cid[cid] = asyncio.Queue()

    connection_data = await sio.get_session(cid)
    if connection_data.get("sid") is not None:
        logger.warning("Connection data already exists for %s. Removing them.", cid)
    await sio.save_session(cid, {"sid": None, "queue": None, "consumer_task": None})

    logger.info("connect %s", cid)


@sio.event
async def disconnect(cid: str):
    # if cid not in queue_by_cid:
    #     logger.warning("Connection data not found for %s. Skip deletion.", cid)
    # else:
    #     del queue_by_cid[cid]

    connection_data = await sio.get_session(cid)
    sid = connection_data.get("sid")
    if sid is not None:
        ts = int(datetime.now().timestamp() * 1000)
        await session_end(cid, ts.to_bytes(8, byteorder="little", signed=False))
        # TODO: clear the queue
        logger.info("disconnect %s & end running session", cid)
    else:
        logger.info("disconnect %s", cid)


@sio.event
async def session_start(cid: str, data: bytes = bytes()):
    """
    Start a new session for the connection.
    A new session is started when a new user comes to use this service;
    therefore, the previous user has left.
    Thus, we can safely remove any incomplete tasks in the previous session.
    """
    if len(data) < 8:
        return error_ack("Data length is less than 8 bytes")

    # check if session_start is already called
    connection_data = await sio.get_session(cid)
    sid = connection_data.get("sid")
    if sid is not None:
        logger.debug(f"Session is already running: sid {sid}")
        return error_ack(f"Session is already running: starting at {sid[-14:]}")

    # get timestamp from the data
    timestamp = get_timestamp(data)
    now = datetime.fromtimestamp(timestamp / 1000)

    # create sid and task queue
    sid = cid + now.strftime("%Y%m%d%H%M%S")
    queue = asyncio.Queue()

    # call all model's start method
    await start_all_models(sid, timestamp)

    # create a consumer task
    consumer_task = asyncio.create_task(consume_frame(queue))

    # save session id, task queue, and the consumer task to this connection
    await sio.save_session(
        cid, {"sid": sid, "queue": queue, "consumer_task": consumer_task}
    )

    logger.debug(
        f"session_start at {datetime.fromtimestamp(timestamp / 1000)}, sid {sid}"
    )
    return success_ack()


@sio.event
async def session_end(cid: str, data: bytes = bytes()):
    if len(data) < 8:
        return error_ack("Data length is less than 8 bytes")

    # check if session_end is already called
    connection_data = await sio.get_session(cid)
    sid = connection_data.get("sid")
    if sid is None:
        return error_ack("Session is not running")

    # get timestamp from the data
    timestamp = get_timestamp(data)

    # abort the task if it's still processing any frame
    consumer_task: asyncio.Task | None = connection_data.get("consumer_task")
    if consumer_task is not None:
        consumer_task.cancel()  # schedule the cancellation
        try:
            await consumer_task  # wait for the task to be actually cancelled
        except asyncio.CancelledError:
            pass
        del consumer_task  # remove the reference to the task

    # call all model's end method
    await end_all_models(sid, timestamp)

    # clear remaining session data in the queue (if any)
    # TODO: do we actually need to clear the queue?
    # after clearing the reference to the task and the queue, all data will be GC-ed

    # clear the connection data
    # Note: Not sure if sio holds a strong or weak reference to connection_data,
    # so reset sio's reference first, before we del connection_data
    await sio.save_session(cid, {"sid": None, "queue": None, "consumer_task": None})
    del connection_data

    logger.debug(
        f"session_end at {datetime.fromtimestamp(timestamp / 1000)}, sid {sid}"
    )
    return success_ack()


@sio.event
async def frame(cid: str, data: bytes = bytes()) -> str:
    if len(data) < 8:
        return error_ack("Data length is less than 8 bytes")

    # check if session_end is already called (should not)
    connection_data = await sio.get_session(cid)
    sid = connection_data.get("sid")
    queue: asyncio.Queue[FrameData] = connection_data.get("queue")
    if sid is None:
        return error_ack("Session is not running")
    if queue is None:
        # This should not happen
        # Because the queue is deleted right at session_end
        return error_ack("Internal error: queue is not initialized")

    # create a FrameData and put it into the queue
    try:
        frame_data = FrameData(data, cid, sid)
    except UnidentifiedImageError:
        return error_ack("Image is corrupted or not in PNG format")

    try:
        queue.put_nowait(frame_data)
    except asyncio.QueueFull:
        return error_ack("Queue is full")

    return success_ack()


@sio.on("*")
async def any_event(event: str, cid: str, data: Any):
    logger.debug(f"Unregistered event {event} received with data {data} from {cid}")
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


def success_ack(data: dict = {}) -> str:
    return json.dumps({"success": True, **data}, ensure_ascii=False)


def error_ack(msg: str = "") -> str:
    return '{"success": false, "error": "%s"}' % msg


if __name__ == "__main__":
    if is_dev:
        logger.setLevel(logging.DEBUG)

    print(
        f"Starting with logger level {logging.getLevelName(logger.getEffectiveLevel())}"
    )

    web.run_app(app, host=HOST, port=PORT, ssl_context=ssl_context)
    # web.run_app(app, host=HOST, port=PORT)
