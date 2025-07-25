import asyncio
import json
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Optional, cast

import numpy.typing as npt
from aiortc import (
    MediaStreamTrack,
    RTCConfiguration,
    RTCDataChannel,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
)
from aiortc.contrib.media import (
    MediaBlackhole,
    MediaRecorder,
)
from aiortc.rtcrtpreceiver import RemoteStreamTrack
from av import VideoFrame

from broker.broker import Broker
from services.database import DatabaseService
from utils.log import logger
from utils.network import END_SESSION_MESSAGE, ICE_SERVERS


class WebRTCSessionManager:
    """
    Manages WebRTC peer connections.

    Attributes:
        loop (asyncio.AbstractEventLoop): An event loop. Used to send webrtc datachannel messages (because the `channel.send` function may be called from a different thread in a callback, need to specify the main thread's event loop when calling)
        record_path (Optional[Path]): Directory path for recording video files, None for no recording
        broker (Broker):

    Note:
        The class expects clients to send specific message formats via data channels:
        - Session end: "END_SESSION" string
        - Participant data: JSON with "ParticipantID" field
        - Wellness data: JSON with "surveyResult", "bodyDataDict", or "weightDataDict" fields
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        record_path: Optional[Path],
        broker: Broker,
    ) -> None:
        self.loop = loop
        self.record_path = record_path
        self.broker = broker

        self.session_pool: dict[str, RTCPeerConnection] = {}

    def get_session(self, pc_id: str):
        """
        Get the peer connection object for a given ID.

        Parameters:
            pc_id (str): The ID of the peer connection.

        Returns:
            RTCPeerConnection: The peer connection object.

        Raises:
            KeyError: If the peer connection ID is not found.
        """
        return self.session_pool[pc_id]

    async def delete_session(self, pc_id: str):
        """
        Delete the peer connection object for a given ID if it exists.

        Parameters:
            pc_id (str): The ID of the peer connection.
        """
        if pc_id not in self.session_pool:
            return
        pc = self.session_pool.pop(pc_id)
        await pc.close()

    async def cleanup(self):
        coros = [pc.close() for pc in self.session_pool.values()]
        await asyncio.gather(*coros)
        self.session_pool.clear()

    async def create_session(
        self,
        sdp: str,
        pc_id: str,
        peer_type: str = "offer",
    ) -> tuple[str, RTCPeerConnection]:
        remote_sdp = RTCSessionDescription(sdp=sdp, type=peer_type)

        pc = RTCPeerConnection(
            configuration=RTCConfiguration(
                iceServers=[RTCIceServer(**ice) for ice in ICE_SERVERS]
            )
        )

        pc_id = str(uuid.uuid4()) if pc_id is None else pc_id
        self.session_pool[pc_id] = pc

        if self.record_path:
            file_name = self.record_path / f"{pc_id}.mp4"
            logger.info("PC(%s) recorded to %s", pc_id, file_name)
            recorder = MediaRecorder(file_name)
        else:
            recorder = MediaBlackhole()

        @pc.on("datachannel")
        def on_datachannel(channel: RTCDataChannel):
            logger.info("PC(%s) remote created datachannel %s", pc_id, channel.id)

            # Handle session-ending messages in the data channel
            @channel.on("message")
            def on_message(message):
                logger.info("PC(%s) received message: %s", pc_id, message)

                if isinstance(message, str) and message.strip() == END_SESSION_MESSAGE:
                    # Mark session end. Ask ML models to clean up and end ASAP
                    self.broker.end_session(pc_id)
                else:
                    try:
                        data = json.loads(message)
                        # participant_id = None
                        timestamp = int(
                            time.time() * 1000
                        )  # Current timestamp in milliseconds

                        # Extract participant ID (After evaluation, we may not use participant id. Then, change here.)
                        if "ParticipantID" in data.keys():
                            participant_id = data["ParticipantID"]
                            # person_id = data["PersonID"]
                            del data["ParticipantID"]
                            # print(f"Received ParticipantID: {participant_id}")
                            if participant_id and not data:
                                self.broker.set_participantID(participant_id)
                                print(
                                    "broker.get_participantID(): ",
                                    self.broker.get_participantID(),
                                )

                            # Only process data if we have a participant ID (for evaluation, participant id and person id are different, so we must have not null participant id)
                            elif participant_id:
                                logger.info(
                                    f"Received ParticipantID and Related Messages: {participant_id} and {list(data.values())}"
                                )
                                person_id = data["PersonID"]
                                logger.info(
                                    f"Received PersonID and Related Messages: {person_id} and {list(data.values())}"
                                )
                                # Initialize database connection if needed
                                db = DatabaseService()

                                # Store wellness data or body data
                                if "surveyResult" in data:
                                    db.store_wellness_data_int(
                                        person_id,
                                        participant_id,
                                        data["surveyResult"],
                                        timestamp,
                                    )
                                    logger.info(
                                        f"Stored survey data for participant {participant_id}"
                                    )
                                elif "bodyDataDict" in data:
                                    db.store_wellness_data_float(
                                        person_id,
                                        participant_id,
                                        data["bodyDataDict"],
                                        timestamp,
                                    )
                                    logger.info(
                                        f"Stored body data for participant {participant_id}"
                                    )
                                elif "weightDataDict" in data:
                                    db.store_wellness_data_float(
                                        person_id,
                                        participant_id,
                                        data["weightDataDict"],
                                        timestamp,
                                    )
                                    logger.info(
                                        f"Stored weight data for participant {participant_id}"
                                    )

                                # # Send confirmation back to client
                                # channel.send(json.dumps({
                                #     "status": "success",
                                #     "message": "data stored successfully",
                                #     "timestamp": timestamp
                                # }))
                        else:
                            logger.info(
                                "Received data without ParticipantID, cannot store any metrics"
                            )

                    except json.JSONDecodeError:
                        print("Failed to decode message as JSON.")
                    except Exception as e:
                        logger.error(f"Error processing data: {str(e)}")
                        # Send error back to client
                        channel.send(
                            json.dumps(
                                {
                                    "status": "error",
                                    "message": f"Failed to process data: {str(e)}",
                                    "timestamp": int(time.time() * 1000),
                                }
                            )
                        )

            # Let broker emit prediction data through datachannel
            def send_data(data: dict):
                if self.broker.get_participantID():
                    data["participant_id"] = self.broker.get_participantID()

                # Create a copy of the data for logging purposes
                log_data = data.copy() if data else {}
                if "face_embedding" in log_data:
                    # Replace the long embedding list with a short placeholder string for logging
                    log_data["face_embedding"] = (
                        f"[Embedding of size {len(log_data.get('face_embedding', []))}]"
                    )

                logger.info("There is data sent from backend: %s", log_data)

                d = json.dumps(
                    data,  # Send the original data, not the modified log_data
                    ensure_ascii=False,
                    default=lambda o: logger.error(f"can't serialize {o}") or None,
                )
                self.loop.run_in_executor(None, lambda: channel.send(d))

            def on_prediction(data: Optional[dict]):
                if channel.readyState == "closed" or data is None:
                    return
                send_data(data)

            self.broker.set_data_handler(pc_id, on_prediction, on_prediction)

        @pc.on("track")
        def on_track(track: MediaStreamTrack):
            logger.info(
                "PC(%s) remote created %s track with id %s and type %s",
                pc_id,
                track.kind,
                track.id,
                type(track),
            )

            if track.kind == "audio":
                # We don't care about audio. Just save it for future convenience
                recorder.addTrack(track)

            elif track.kind == "video":
                # Should be this subclass according to my tests. Cast for IDE code completion.
                track = cast(RemoteStreamTrack, track)

                # Mark session start, ML models please get ready
                self.broker.start_session(pc_id)
                logger.info(f"PC({pc_id}) model session started")

                # Forward the video to ML models and the recorder
                # NOTE: As soon as creating a `transformed_track`, DO NOT USE `track` anymore!!!
                transformed_track = VideoTransformTrack(
                    track,
                    lambda frame, timestamp: self.broker.frame(pc_id, frame, timestamp),
                )
                recorder.addTrack(transformed_track)
                logger.debug(
                    f"PC({pc_id}) prediction and recording hooks are registered"
                )

                # Determine the start/stop of recorder based on the track
                @track.on("ended")
                async def on_ended():
                    logger.info(f"PC({pc_id}) video track {track.id} ended")

                    # TODO: Sometimes the line below throws an error "non monotonically increasing dts to muxer in stream 0"
                    # Don't now why and how to fully fix it
                    # The fix in https://github.com/aiortc/aiortc/issues/580 is already applied
                    await recorder.stop()

        # more event handlers and logging
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info("PC(%s) -> %s", pc_id, pc.connectionState)
            if pc.connectionState == "failed":
                await pc.close()
                del self.session_pool[pc_id]

        @pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            logger.debug("PC(%s) iceConnectionState->%s", pc_id, pc.iceConnectionState)

        @pc.on("signalingstatechange")
        async def on_signalingstatechange():
            logger.debug("PC(%s) signalingState->%s", pc_id, pc.signalingState)

        @pc.on("icegatheringstatechange")
        async def on_icegatheringstatechange():
            logger.debug("PC(%s) iceGatheringState->%s", pc_id, pc.iceGatheringState)

        # handle remote SDP (should be an offer)
        await pc.setRemoteDescription(remote_sdp)
        await recorder.start()

        # generate and send local SDP (should be an answer)
        local_sdp = await pc.createAnswer()
        await pc.setLocalDescription(local_sdp)

        return pc_id, pc


class VideoTransformTrack(VideoStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(
        self,
        track: MediaStreamTrack,
        on_frame: Callable[[npt.NDArray, int], None],
        reset_timestamp=False,
    ):
        # def __init__(self, track: MediaStreamTrack, sid: str, reset_timestamp=True):
        super().__init__()
        self.track = track
        self.on_frame = on_frame
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
        self.on_frame(data, timestamp)

        return frame  # frame.reformat(format="rgb24") # is reformat needed? The returned frame is to be saved to a file
