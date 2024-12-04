import argparse
import asyncio
import json
import logging
import os
from enum import Enum
import ssl
from typing import Optional, Union

import aiohttp
from aiortc import (
    RTCIceCandidate,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder
from av.video.frame import VideoFrame
from dotenv import load_dotenv

Role = Enum("Role", ["offer", "answer"])

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"


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
            print("Data channel message:", message)
            message_obj: dict = json.loads(message)
            if message_obj.get("final", False):
                print("The test client receives the `end` data.", message_obj)
                print("Closing the connection in 3 seconds...")
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
                data_channel.send("end session")

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
    port = os.environ.get("PORT", "8080")

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
    print("Connecting to", url)

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
