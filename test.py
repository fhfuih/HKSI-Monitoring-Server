import argparse
import asyncio
import logging

import aiohttp
from aiortc import (
    RTCIceCandidate,
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
)
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder
from av.video.frame import VideoFrame


async def run(pc, player, recorder, role):
    # a future that only resolves when the player is done
    future = asyncio.get_running_loop().create_future()

    def add_tracks():
        if not player:
            return

        if player.audio:
            pc.addTrack(player.audio)
            print("Adding audio track")

        if player.video:
            track = player.video

            @track.on("ended")
            async def on_ended():
                await pc.close()
                future.set_result(None)

            pc.addTrack(track)
            print("Adding video track")

    @pc.on("track")
    def on_track(track):
        print("Receiving %s" % track.kind)
        recorder.addTrack(track)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            future.set_result(None)

    if role == "offer":
        # send offer
        add_tracks()
        await pc.setLocalDescription(await pc.createOffer())
        offer_sdp = pc.localDescription.sdp
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:8080/offer", json={"sdp": offer_sdp, "type": "offer"}
            ) as response:
                if response.status != 200:
                    print("Failed to send offer")
                    return
                params = await response.json()
                obj = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    # consume signaling
    if isinstance(obj, RTCSessionDescription):
        await pc.setRemoteDescription(obj)
        await recorder.start()
    elif isinstance(obj, RTCIceCandidate):  # type: ignore
        await pc.addIceCandidate(obj)
    else:
        print("Unknown message", repr(obj))  # type: ignore

    await future


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video stream from the command line")
    parser.add_argument("--role", choices=["offer", "answer"], default="offer")
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
