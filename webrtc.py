import argparse
import asyncio
import json
import logging
import os
import ssl
from typing import Awaitable, cast
import uuid
import av
import cv2
import numpy as np
from aiohttp import web
from aiohttp_catcher import Catcher
from aiohttp_catcher.canned import AIOHTTP_SCENARIOS
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import (
    MediaBlackhole,
    MediaPlayer,
    MediaRecorder,
    MediaRelay,
    MediaStreamTrack,
)
from aiortc.contrib.signaling import TcpSocketSignaling, BYE
from av.video.frame import VideoFrame
from av import logging as av_logging

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()


# class MyVideoTransformTrack(VideoStreamTrack):
#     def __init__(self, track):
#         super().__init__()
#         self.track = track
#         self.frames = []

#     async def recv(self):
#         frame = await self.track.recv()
#         img = frame.to_ndarray(format="bgr24")

#         # Assuming the frame is already 240x240
#         self.frames.append(img)

#         return frame


# async def run(pc: RTCPeerConnection, signaling: TcpSocketSignaling):
#     await signaling.connect()

#     @pc.on("datachannel")
#     async def on_datachannel(channel):
#         @channel.on("message")
#         async def on_message(message):
#             if isinstance(message, str) and message.startswith("start"):
#                 print("Starting video capture")
#             elif isinstance(message, str) and message.startswith("stop"):
#                 print("Stopping video capture and saving")
#                 video_file = "output.avi"
#                 out = cv2.VideoWriter(
#                     video_file, cv2.VideoWriter_fourcc(*"XVID"), 20.0, (240, 240)
#                 )
#                 for frame in pc.video_track.frames:
#                     out.write(frame)
#                 out.release()
#                 print(f"Video saved as {video_file}")

#     @pc.on("iceconnectionstatechange")
#     async def on_iceconnectionstatechange():
#         if pc.iceConnectionState == "failed":
#             await pc.close()
#             await signaling.close()

#     @pc.on("track")
#     def on_track(track):
#         if track.kind == "video":
#             pc.video_track = MyVideoTransformTrack(track)
#             pc.addTrack(pc.video_track)

#     while True:
#         obj = await signaling.receive()
#         if isinstance(obj, RTCSessionDescription):
#             await pc.setRemoteDescription(obj)
#             if obj.type == "offer":
#                 await pc.setLocalDescription(await pc.createAnswer())
#                 await signaling.send(pc.localDescription)
#         elif obj is BYE:
#             print("Exiting")
#             break


class VideoTransformTrack(VideoStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self, track: MediaStreamTrack, reset_timestamp=False):
        super().__init__()
        self.track = cast(VideoStreamTrack, track)
        self.reset_timestamp = reset_timestamp

    async def recv(self):
        frame = cast(VideoFrame, await self.track.recv())

        logger.debug(frame)

        # This attempts to fix messed up frame order and speed
        if self.reset_timestamp:
            pts, time_base = await self.next_timestamp()
            frame.pts = pts
            frame.time_base = time_base

        return frame.reformat(format="bgr24")


async def offer(request: web.Request) -> web.Response:
    try:
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    except:  # noqa: E722
        try:
            text = await request.text()
            logger.debug(
                "Received invalid offer payload from %s: %s", request.remote, text
            )
        except UnicodeDecodeError:
            pass

        return web.Response(
            content_type="application/json",
            headers={"Acceppt": "application/json"},
            text=json.dumps({"sdp": None, "type": None}),
            status=400,
        )

    pc = RTCPeerConnection()
    pc_id = uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(f"PeerConnection({pc_id})" + " " + msg, *args)

    log_info("Created for %s", request.remote)

    # prepare local media
    # player = MediaPlayer(os.path.join(ROOT, "demo-instruct.wav"))
    if args.record_to:
        recorder = MediaRecorder(args.record_to)
    else:
        recorder = MediaBlackhole()

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track: MediaStreamTrack):
        log_info("Track %s received", track.kind)

        if track.kind == "audio":
            # This sends (forwards) the audio back to every client
            # pc.addTrack(player.audio)

            # This adds the audio track to the file writer
            recorder.addTrack(track)
        elif track.kind == "video":
            # This sends (forwards) the video back to every client
            # pc.addTrack(
            #     VideoTransformTrack(
            #         relay.subscribe(track), transform=params.get("video_transform")
            #     )
            # )

            # This adds the audio track to the file writer
            if args.record_to:
                recorder.addTrack(VideoTransformTrack(relay.subscribe(track)))

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            # TODO: Sometimes the line below throws an error "non monotonically increasing dts to muxer in stream 0"
            # Don't now why and how to fully fix it
            # The fix in https://github.com/aiortc/aiortc/issues/580 is already applied
            await recorder.stop()

    # handle offer
    await pc.setRemoteDescription(offer)
    await recorder.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    logger.debug("Receiving remote SDP %s", pc.remoteDescription)
    logger.debug("Responding with local SDP %s", pc.localDescription)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    # signaling = TcpSocketSignaling("localhost", 1234)
    # pc = RTCPeerConnection()
    # loop = asyncio.get_event_loop()
    # try:
    #     loop.run_until_complete(run(pc, signaling))
    # finally:
    #     loop.run_until_complete(pc.close())
    #     loop.run_until_complete(signaling.close())

    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels demo"
    )
    parser.add_argument("--ssl", type=bool, default=False, help="Use SSL")
    parser.add_argument(
        "--cert-file",
        default=os.path.join(ROOT, "ssl", "cert.pem"),
        help="SSL certificate file (for HTTPS). Default is ./ssl/cert.pem",
    )
    parser.add_argument(
        "--key-file",
        default=os.path.join(ROOT, "ssl", "key.pem"),
        help="SSL key file (for HTTPS). Default is ./ssl/key.pem",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--record-to", help="Write received media to a file.")
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    av_logging.set_level(av_logging.ERROR)

    if args.ssl:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None
    catcher = Catcher()
    app = web.Application(middlewares=[catcher.middleware])
    app.on_shutdown.append(on_shutdown)
    app.router.add_post("/offer", offer)
    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )
