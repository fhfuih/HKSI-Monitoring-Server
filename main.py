import argparse
import asyncio
import json
import logging
import os
import ssl
import sys
import uuid
from collections.abc import AsyncIterable, Iterable
from pathlib import Path
from typing import Optional, cast

from dotenv import load_dotenv

load_dotenv()

# some ops are not available in macOS metal accelerator
# This allows fallback to CPU
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import websockets
from aiortc import RTCSessionDescription
from aiortc.sdp import candidate_from_sdp
from av import logging as av_logging

from broker import Broker
from models.base_model import BaseModel
from models.eye_bag_model import EyeBagModel
from models.face_rec_model import FaceRecognitionModel
from models.fatigue_model import FatigueModel
from models.mock_model_1 import MockModel1
from models.mock_model_2 import MockModel2
from models.Physiological import HeartRateAndHeartRateVariabilityModel
from models.pimple_model import PimpleModel
from network.webrtc_session_manager import WebRTCSessionManager
from utils.log import logger, set_console_log_level
from utils.monkey_patch import monkeypatch_method
from utils.network import ICE_SERVERS

ROOT = os.path.dirname(__file__)
# MODELS: list[type[BaseModel]] = [
#     FatigueModel,
#     EyeBagModel,
#     PimpleModel,
#     HeartRateAndHeartRateVariabilityModel,
#     FaceRecognitionModel,
# ]

MODELS: list[type[BaseModel]] = [MockModel1, MockModel2]
broker = Broker(MODELS)

# Command line arguments & globals to be initialized later
record_path = cast(Path, None)
webrtc_session_manager = cast(WebRTCSessionManager, None)


# NOTE: Monkey-patch a send method that also yields controls to the event loop
# See https://websockets.readthedocs.io/en/stable/faq/asyncio.html#why-does-my-program-never-receive-any-messages
# See https://github.com/python-websockets/websockets/issues/867
# @monkeypatch_method(websockets.ServerConnection)
# async def send_and_yield(
#     self: websockets.ServerConnection,
#     message: websockets.Data
#     | Iterable[websockets.Data]
#     | AsyncIterable[websockets.Data],
#     text: bool | None = None,
# ):
#     await self.send(message, text)
#     await asyncio.sleep(0)


# setattr(websockets.ServerConnection, "send_and_yield", send_and_yield)


async def websocket_process_message(
    websocket: websockets.ServerConnection, client_id: str, message: dict
):
    msg_type = message.get("type")

    if msg_type == "offer":
        try:
            ice_data: dict = message["data"]
            ice_sdp: str = ice_data["sdp"]
            if ice_data["type"] != "offer":
                raise ValueError("Only support offer type")
        except (KeyError, TypeError) as e:
            logger.error("Invalid offer format: %s", e)
            await websocket.send(
                json.dumps(
                    {
                        "type": "error",
                        "data": {
                            "message": "Invalid offer format",
                            "code": 400,
                        },
                    }
                )
            )
            return

        offer = RTCSessionDescription(sdp=ice_sdp, type="offer")
        pc_id, pc = await webrtc_session_manager.create_session(
            offer.sdp, client_id, "offer"
        )

        logger.info("PC(%s) created", pc_id)
        logger.debug("Receiving remote SDP %s", pc.remoteDescription)
        logger.debug("Responding with local SDP %s", pc.localDescription)

        await websocket.send(
            json.dumps(
                {
                    "type": "answer",
                    "data": {
                        "sdp": pc.localDescription.sdp,
                        "type": pc.localDescription.type,
                    },
                }
            )
        )

    elif msg_type == "ice-candidate":
        try:
            pc = webrtc_session_manager.get_session(client_id)
        except KeyError:
            logger.error("No peer connection found for client %s", client_id)
            await websocket.send(
                json.dumps(
                    {
                        "type": "error",
                        "data": {
                            "message": "No peer connection found",
                            "code": 400,
                        },
                    }
                )
            )
            return
        try:
            ice_data: dict = message["data"]
            ice_sdp: str = ice_data["sdp"]
            ice_sdpMLineIndex: str | int = ice_data["sdpMLineIndex"]
            ice_sdpMid: Optional[str] = ice_data.get("sdpMid", None)
            # ice_serverUrl: Optional[str] = ice_data.get("serverUrl", None)

            if ice_sdpMid is None and ice_sdpMLineIndex is None:
                raise ValueError("Either sdpMid or sdpMLineIndex must be set")
            if not ice_sdp.startswith("candidate:"):
                raise ValueError(f"Not starting from `candidate:` ({ice_sdp[:10]}…)")
            ice_sdp = ice_sdp[10:]  # len("candidate:")==10. strip "candidate:"
            ice_sdpMLineIndex = int(ice_sdpMLineIndex)

            ice_candidate = candidate_from_sdp(ice_sdp)
            ice_candidate.sdpMid = ice_sdpMid
            ice_candidate.sdpMLineIndex = ice_sdpMLineIndex

            await pc.addIceCandidate(ice_candidate)
            logger.info("PC(%s) added ICE candidate %s", client_id, ice_sdp)
        except (KeyError, TypeError, ValueError) as e:
            logger.error("Invalid ice candidate format: %s", e)
            await websocket.send(
                json.dumps(
                    {
                        "type": "error",
                        "data": {
                            "message": "Invalid ice candidate format",
                            "code": 400,
                        },
                    }
                )
            )
            return

        await websocket.send(json.dumps({"type": "success", "data": None}))

    elif msg_type == "ice-servers-request":
        await websocket.send(json.dumps({"type": "ice-servers", "data": ICE_SERVERS}))

    else:  # unknown msg_type
        logger.warning("Unknown message type: %s", msg_type)
        await websocket.send(
            json.dumps(
                {
                    "type": "error",
                    "data": {
                        "message": f"Unknown message type: {msg_type}",
                        "code": 400,
                    },
                }
            )
        )

    # NOTE: Since the websocket is in a request-response pattern,
    # Every type of message triggers a `send_and_yield`,
    # ending up with an `asyncio.sleep(0)` to force yield controls to the event loop.
    # So I don't need to call another one here.


async def websocket_handler(websocket: websockets.ServerConnection):
    """Handle WebSocket connections for WebRTC signaling."""
    client_id = str(uuid.uuid4())
    logger.info(
        "~~~~~~\n🛜 New WebSocket connection from %s (ID: %s)",
        websocket.remote_address,
        client_id,
    )

    try:  # Here catches WebSocket connection errors
        async for message in websocket:
            try:  # Here catches message handling errors
                message_json = json.loads(message)
                if not isinstance(message_json, dict):
                    raise ValueError()
                await websocket_process_message(websocket, client_id, message_json)
            except (json.JSONDecodeError, ValueError):
                logger.error("Invalid JSON object(dist) message: %s", message)
                await websocket.send(
                    json.dumps(
                        {
                            "type": "error",
                            "data": {"message": "Invalid JSON message", "code": 400},
                        }
                    )
                )
            except Exception as e:
                logger.error("Error processing WebSocket message: %s", e)
                await websocket.send(
                    json.dumps(
                        {
                            "type": "error",
                            "data": {"message": f"Server error: {str(e)}", "code": 500},
                        }
                    )
                )
            await asyncio.sleep(0)
    except websockets.exceptions.ConnectionClosed:
        logger.info("WebSocket connection closed for client %s", client_id)
    finally:
        logger.info("WebSocket connection ended for client %s", client_id)


if __name__ == "__main__":
    ssl_cert_file = os.environ.get("SSL_CERT_FILE")
    ssl_key_file = os.environ.get("SSL_KEY_FILE")
    ssl_cert_file = ssl_cert_file and ssl_cert_file.strip()
    ssl_key_file = ssl_key_file and ssl_key_file.strip()
    if ssl_cert_file and ssl_key_file:  # not None or empty string
        logger.info(
            "SSL enabled. Assuming listening 0.0.0.0 (unless otherwise specified)."
        )
        ssl_context = ssl.SSLContext(protocol=ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(ssl_cert_file, ssl_key_file)
    else:
        logger.info(
            "SSL disabled because %s is empty. "
            "Assuming listening localhost only (unless otherwise specified). "
            "But you can still configure SSL by putting this service behind a reverse proxy. ",
            "SSL_CERT_FILE" if not ssl_cert_file else "SSL_KEY_FILE",
        )
        ssl_context = None

    host = "127.0.0.1" if ssl_context else "0.0.0.0"
    port = int(port_str) if (port_str := os.environ.get("PORT")) else 8083

    parser = argparse.ArgumentParser(
        description="HKSI WebRTC server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--record-to",
        default="./recordings",
        help="Write received media to a folder. Give it an empty string to disable",
    )
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if record_to := args.record_to.strip():
        record_path = Path(record_to).resolve()
        if record_path.is_file():
            print(
                "The record-to path is a file. It should be a folder instead.",
                file=sys.stderr,
            )
            sys.exit(1)
        record_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving video recordings to {record_path}")
    else:
        record_path = None
        logger.info(f"Don't save video recordings")

    if args.verbose:
        set_console_log_level(logging.DEBUG)
    av_logging.set_level(av_logging.ERROR)  # Slient internal logging of the av package

    async def main():
        global webrtc_session_manager
        print("#=====🛜=====🛜")
        logger.info(f"Starting WebSocket signaling server on {host}:{port}")

        # Launch core service components
        loop = asyncio.get_running_loop()
        webrtc_session_manager = WebRTCSessionManager(loop, record_path, broker)

        async with websockets.serve(
            websocket_handler,
            host,
            port,
            ssl=ssl_context,
        ) as server:
            logger.info(
                f"WebSocket signaling server running on {'wss' if ssl_context else 'ws'}://{host}:{port}"
            )
            print("#=====🛜=====🛜")
            await server.serve_forever()

    asyncio.run(main())
