import argparse
import asyncio
import json
import logging
import os
import ssl
import time
from enum import Enum
from typing import Optional, Union

import aiohttp
import websockets
from aiortc import (
    RTCIceCandidate,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder
from av.video.frame import VideoFrame
from dotenv import load_dotenv

from utils.network import END_SESSION_MESSAGE

Role = Enum("Role", ["offer", "answer"])

# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"


async def run(
    pc: RTCPeerConnection,
    player: Optional[MediaPlayer],
    recorder: Union[MediaBlackhole, MediaRecorder],
    role: Role,
):
    # a future that only resolves when the player is done
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    def add_tracks():
        # Data channel
        data_channel = pc.createDataChannel("data")

        @data_channel.on("open")
        def on_data_channel_open():
            print("Data channel opened")

        @data_channel.on("message")
        async def on_data_channel_message(message):
            message_obj: dict = json.loads(message)

            # Create a copy for logging and replace the embedding with a placeholder
            log_obj = message_obj.copy()
            if "face_embedding" in log_obj:
                log_obj["face_embedding"] = (
                    f"[Embedding of size {len(log_obj.get('face_embedding', []))}]"
                )

            print("🤖", log_obj, time.time())  # Print the cleaned-up object

            if message_obj.get("final", False):
                print(
                    "The test client receives the `end` data ('final': true). Closing the connection in 3 seconds…"
                )
                await asyncio.sleep(3)

                await pc.close()
                future.set_result(None)

        # Media tracks
        if player and player.audio:
            pc.addTrack(player.audio)
            print("Adding audio track")

        if player and player.video:
            track = player.video

            @track.on("ended")
            async def on_ended():
                print("Video track end.")
                print(
                    "The test client will wait for the server to send an `end` data, and exit itself after that."
                )
                print(
                    "In the real client, the WebRTC connection will be closed after, but may not be immediately after, the `end` data."
                )

                # Telling the server to end ML models
                data_channel.send(END_SESSION_MESSAGE)

                # Removing the video track. This is actually not required.
                # And in standard WebRTC, this will not notify remote side of track removal
                # somehow in python aiortc, it can notify the remote
                # for transceiver in pc.getTransceivers():
                #     # print(f"Transceiver ${transceiver.kind}, ${transceiver.stopped}")
                #     if transceiver.kind == "video" or transceiver.kind == "audio":
                #         await transceiver.stop()
                #     # print(f"Transceiver ${transceiver.kind}, ${transceiver.stopped}")

            pc.addTrack(track)
            print("Adding video track")

    async def negotiate():
        await pc.setLocalDescription(await pc.createOffer())
        offer_sdp = pc.localDescription.sdp

        # Use HTTP if specified or WebSocket otherwise
        if args.use_http:
            print("Using HTTP signaling as requested")
            await negotiate_http(offer_sdp)
        else:
            # Connect to WebSocket signaling server
            try:
                if ssl_context and ssl_context is not True:
                    # Custom SSL context
                    async with websockets.connect(ws_url, ssl=ssl_context) as websocket:
                        await negotiate_websocket(websocket, offer_sdp)
                elif ssl_context is True:
                    # Default SSL context
                    async with websockets.connect(ws_url) as websocket:
                        await negotiate_websocket(websocket, offer_sdp)
                else:
                    # No SSL
                    async with websockets.connect(ws_url) as websocket:
                        await negotiate_websocket(websocket, offer_sdp)
            except Exception as e:
                print(f"WebSocket connection failed: {e}")
                print("Falling back to HTTP signaling...")
                await negotiate_http(offer_sdp)

    async def negotiate_websocket(websocket, offer_sdp):
        # Send offer through WebSocket
        offer_message = {"type": "offer", "data": {"sdp": offer_sdp, "type": "offer"}}
        await websocket.send(json.dumps(offer_message))
        print("Sent offer via WebSocket")

        # Wait for answer
        response = await websocket.recv()
        data = json.loads(response)

        if data.get("type") == "answer":
            answer_data = data.get("data", {})
            obj = RTCSessionDescription(
                sdp=answer_data["sdp"], type=answer_data["type"]
            )
            print("Received answer via WebSocket")
        elif data.get("type") == "error":
            error_data = data.get("data", {})
            print(f"Server error: {error_data.get('message', 'Unknown error')}")
            return
        else:
            print(f"Unknown message type: {data.get('type')}")
            return

        if isinstance(obj, RTCSessionDescription):
            await pc.setRemoteDescription(obj)
            await recorder.start()
        elif isinstance(obj, RTCIceCandidate):  # type: ignore
            await pc.addIceCandidate(obj)
        else:
            print("Unknown message", repr(obj))  # type: ignore

    async def negotiate_http(offer_sdp):
        # Fallback to HTTP signaling
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json={"sdp": offer_sdp, "type": "offer"},
                ssl=ssl_context,
            ) as response:
                if response.status != 200:
                    print("Failed to send offer")
                    return
                params = await response.json()
                obj = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        if isinstance(obj, RTCSessionDescription):
            await pc.setRemoteDescription(obj)
            await recorder.start()
        elif isinstance(obj, RTCIceCandidate):  # type: ignore
            await pc.addIceCandidate(obj)
        else:
            print("Unknown message", repr(obj))  # type: ignore

    @pc.on("track")
    def on_track(track):
        print("Receiving %s" % track.kind)
        recorder.addTrack(track)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("Connection state is", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            future.set_result(None)

    if role != "offer":
        raise NotImplementedError("The test only supports offer role for now")

    add_tracks()
    await negotiate()

    await future


if __name__ == "__main__":
    load_dotenv()

    host = os.environ.get("HOST", "localhost")
    # port = os.environ.get("PORT", "8080")
    port = os.environ.get("PORT", "8083")

    ssl_cert_file = (s := os.environ.get("SSL_CERT_FILE")) and s.strip()
    ssl_key_file = (s := os.environ.get("SSL_KEY_FILE")) and s.strip()
    protocol = "https" if ssl_cert_file and ssl_key_file else "http"

    is_self_signed = ssl_key_file == "./ssl/key.pem"
    print(
        "Using protocol:",
        protocol,
        "with SSL",
        "off (due to self-signed certificate)" if is_self_signed else "on",
    )
    if is_self_signed:
        ssl_context = ssl.create_default_context()
        ssl_context.load_cert_chain(str(ssl_cert_file), str(ssl_key_file))
    else:
        ssl_context = True

    url = f"{protocol}://{host}:{port}/offer"
    ws_protocol = "wss" if protocol == "https" else "ws"
    ws_port = int(port) + 1  # WebSocket server runs on the next port
    ws_url = f"{ws_protocol}://{host}:{ws_port}"
    print("HTTP URL:", url)
    print("WebSocket URL:", ws_url)

    parser = argparse.ArgumentParser(description="Video stream from the command line")
    parser.add_argument("--role", choices=["offer"], default="offer")
    parser.add_argument(
        "--play-from",
        default="sample-video.mp4",
        help="Read the media from a file and sent it.",
    )
    parser.add_argument(
        "--record-to", default=None, help="Write received media to a file."
    )
    parser.add_argument("--verbose", "-v", action="count")
    parser.add_argument(
        "--use-http",
        action="store_true",
        help="Use HTTP signaling instead of WebSocket (for backward compatibility)",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    # create peer connection
    pc = RTCPeerConnection()

    # create media source
    if args.play_from:
        player = MediaPlayer(args.play_from)
    else:
        player = None

    # create media sink
    if args.record_to:
        recorder = MediaRecorder(args.record_to)
    else:
        recorder = MediaBlackhole()

    # run event loop
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(
            run(
                pc=pc,
                player=player,
                recorder=recorder,
                role=args.role,
            )
        )
    except KeyboardInterrupt:
        pass
    finally:
        # cleanup
        loop.run_until_complete(recorder.stop())
        loop.run_until_complete(pc.close())
