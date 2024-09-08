import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid
from typing import Awaitable, cast

import av
import numpy as np
from aiohttp import web
from aiohttp_catcher import Catcher
from aiohttp_catcher.canned import AIOHTTP_SCENARIOS
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
)
from aiortc.contrib.media import (
    MediaBlackhole,
    MediaRecorder,
    MediaStreamTrack,
)
from aiortc.rtcrtpreceiver import RemoteStreamTrack
from av import logging as av_logging
from av.video.frame import VideoFrame

from broker import Broker
from models.mock_model_1 import MockModel1
from models.mock_model_2 import MockModel2

ROOT = os.path.dirname(__file__)
MODELS = [MockModel1, MockModel2]

# Logs
logger = logging.getLogger("HKSI WebRTC")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

file_handler = logging.FileHandler("webrtc.log")
file_handler.setLevel(logging.WARNING)
logger.addHandler(file_handler)

pcs = set()  # keep track of peer connections for cleanup

broker = Broker(MODELS)


class VideoTransformTrack(VideoStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self, track: MediaStreamTrack, sid: str, reset_timestamp=False):
        super().__init__()
        self.track = track
        self.sid = sid
        self.reset_timestamp = reset_timestamp

    async def recv(self):
        # The track is an aiortc.rtcrtpreceiver.RemoteStreamTrack, and the frame is av.video.frame.VideoFrame
        frame = cast(VideoFrame, await self.track.recv())

        # logger.debug(f"{type(frame)} {frame.format.name}, {frame.width}x{frame.height}")

        # This attempts to fix messed up frame order and speed
        if self.reset_timestamp:
            pts, time_base = await self.next_timestamp()
            frame.pts = pts
            frame.time_base = time_base

        # Feed the frame to ML models
        data = frame.to_ndarray(format="rgb24")
        timestamp = round(frame.time * 1000)  # frame.time is in seconds
        broker.frame(self.sid, data, timestamp)

        return frame  # frame.reformat(format="rgb24") # is reformat needed? The returned frame is to be saved to a file


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
    pc_id = str(uuid.uuid4())
    pcs.add(pc)

    logger.info("PeerConnection (%s) created for %s", pc_id, request.remote)

    # prepare local media
    if args.record_to:
        recorder = MediaRecorder(args.record_to)
    else:
        recorder = MediaBlackhole()

    @pc.on("datachannel")
    def on_datachannel(channel):  # type: ignore
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():  # type: ignore
        logger.info("PeerConnection (%s) state is %s", pc_id, pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track: MediaStreamTrack):  # type: ignore
        logger.info(
            "PeerConnection (%s) received %s track with id %s and type %s",
            pc_id,
            track.kind,
            track.id,
            type(track),
        )

        if track.kind == "audio":
            # We don't care about audio. Just save it for future convenience
            if args.record_to:
                recorder.addTrack(VideoTransformTrack(track, pc_id))

        elif track.kind == "video":
            # Should be this subclass according to log. Cast for IDE code completion.
            track = cast(RemoteStreamTrack, track)

            # Mark session start
            broker.start_session(pc_id, 0)

            # This sends (forwards) the video back to every client
            # pc.addTrack(relay.subscribe(track))

            # This adds the track to the file writer
            if args.record_to:
                recorder.addTrack(VideoTransformTrack(track, pc_id))

            # Determine the start/stop of recorder based on the track
            @track.on("ended")
            async def on_ended():
                logger.info(f"PeerConnection ({pc_id}) video track {track.id} ended")

                # Mark session end
                broker.end_session(pc_id)

                # TODO: Sometimes the line below throws an error "non monotonically increasing dts to muxer in stream 0"
                # Don't now why and how to fully fix it
                # The fix in https://github.com/aiortc/aiortc/issues/580 is already applied
                await recorder.stop()

    # handle offer
    await pc.setRemoteDescription(offer)
    await recorder.start()

    # send answer
    # `answer` will never be null if we look into the source code.
    # The type checking is wrong.
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)  # type: ignore

    logger.debug("Receiving remote SDP %s", pc.remoteDescription)
    logger.debug("Responding with local SDP %s", pc.localDescription)

    return web.json_response(
        {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HKSI WebRTC server")
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
        console_handler.setLevel(logging.DEBUG)
    av_logging.set_level(av_logging.ERROR)  # Internal logging of the av package

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
