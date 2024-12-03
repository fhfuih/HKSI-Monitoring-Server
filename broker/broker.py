import logging
from collections.abc import Hashable
from typing import Optional, Type

import numpy.typing as npt

from models.base_model import BaseModel

from .model_manager_worker import ModelManagerWorker
from .types import *

logger = logging.getLogger("HKSI WebRTC")


class Broker:
    def __init__(
        self,
        models: list[Type[BaseModel]],
    ) -> None:
        self._sessions: dict[Hashable, SessionAsset] = {}
        self._models = models

        self.manager_thread = ModelManagerWorker(self._models, self)
        self.manager_thread.start()
        logger.debug("Broker started ModelManagerWorker")

    def start_session(self, sid: Hashable, timestamp: int):
        if sid in self._sessions:
            raise KeyError(f"A session with id {sid} already exists.")

        # Prepare the session asset.
        self._sessions[sid] = SessionAsset(sid=sid)
        logger.debug("Broker(%s) prepared session assets", sid)

        # Tell ML models to start
        self.manager_thread.add_start(sid, timestamp)
        logger.debug("Broker(%s) told ML models to process `start` action", sid)

    def end_session(self, sid: Hashable, timestamp: Optional[int] = None):
        session_asset = self._sessions.get(sid, None)
        if session_asset is None:
            # Cannot find such a session. All good.
            return

        if session_asset.ending:
            # Don't add more data into a session that is about to end.
            return

        # Record this session to be ending soon
        session_asset.ending = True

        # Tell ML models to end
        self.manager_thread.add_end(sid, timestamp)
        logger.debug("Broker(%s) told ML models to process `stop` action", sid)

    def frame(self, sid: Hashable, data: npt.NDArray, timestamp: int):
        session_asset = self._sessions.get(sid, None)
        if session_asset is None:
            # Cannot find such a session. All good.
            return

        if session_asset.ending:
            # Don't add more data into a session that is about to end.
            return

        # Tell ML models to queue this frame
        self.manager_thread.add_frame(sid, timestamp, data)

    def set_data_handler(
        self,
        sid: Hashable,
        on_intermediate_data: OnDataCallback,
        on_end_data: OnDataCallback,
    ):
        session_asset = self._sessions[sid]
        session_asset.on_intermediate_data = on_intermediate_data
        session_asset.on_end_data = on_end_data
