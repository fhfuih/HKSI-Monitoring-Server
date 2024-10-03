import logging
import time
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from enum import Enum
from queue import Queue, SimpleQueue
from threading import Barrier, Thread, Timer
from typing import NamedTuple, Optional, Type, Union

import numpy as np

from models.base_model import BaseModel

logger = logging.getLogger("HKSI WebRTC")

ThreadActionType = Enum("ThreadActionType", ["Start", "Frame", "End"])


class ThreadAction(NamedTuple):
    type: ThreadActionType
    sid: Hashable
    timestamp: Optional[int]
    data: Optional[np.ndarray]


ActionQueue = Queue[ThreadAction]
OnDataCallback = Callable[[Optional[dict]], None]

# Monkey-patch debug info
setattr(Queue, "__del__", lambda self: logger.debug("__del__ Queue"))


class ModelManagerWorker(Thread):
    """
    The class that manages all ML models.
    """

    def __init__(
        self,
        models: list[Type[BaseModel]],
        broker,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._broker = broker
        self.__queue = ActionQueue()
        self.__scheduled_end = False

        # barrier.wait() blocks until <count> threads all called wait,
        # and all threads are released simultaneously.
        # Here, we use barriers to ensure that
        # (1) all models have finished this frame
        # (2) the manager (main queue) has another item to process
        self.__barrier = Barrier(len(models) + 1)

        # Create model workers and start all of them
        self.__model_workers = [
            ModelWorker(M(), self.__barrier, *args, **kwargs) for M in models
        ]

        for mw in self.__model_workers:
            mw.start()

    def __del__(self):
        logger.debug("__del__ ModelManagerWorker")

    ### Public methods
    def add_start(self, sid: Hashable, timestamp: int):
        self.__queue.put(ThreadAction(ThreadActionType.Start, sid, timestamp, None))

    def add_end(self, sid: Hashable, timestamp: Optional[int]):
        # Remove all queued frames
        while not self.__queue.empty():
            self.__queue.get()

        # Insert the end frame
        self.__queue.put(ThreadAction(ThreadActionType.End, sid, timestamp, None))

    def add_frame(self, sid: Hashable, timestamp: int, data: np.ndarray):
        self.__queue.put(ThreadAction(ThreadActionType.Frame, sid, timestamp, data))

    ### Thread routine
    def run(self) -> None:
        while not self.__scheduled_end:
            # Waits until another action is received
            action = self.__queue.get()
            sid = action.sid

            # Detect if this is an "end" action, exit the loop after handling end
            if action.type == ThreadActionType.End:
                self.__scheduled_end = True

            # Feed the action to all models
            logger.debug(
                f"ModelManagerWorker assigning action {action.type} at {action.timestamp}"
            )
            for mw in self.__model_workers:
                mw.queue.put(action)

            # Waits until all ModelWorker's have finished
            logger.debug(
                f"Before ModelManagerWorker waits for barrier, there are threads {self.__barrier.n_waiting} waiting"
            )
            self.__barrier.wait()

            # Get results from all models
            result = None
            for mw in self.__model_workers:
                if mw.output is not None:
                    if result is None:
                        result = {}
                    if action.timestamp != mw.timestamp:
                        logger.warning(
                            "Mismatched timestamp: manager %d, model %s %s",
                            action.timestamp,
                            mw.name,
                            mw.timestamp,
                        )
                    result.update(mw.output)
            if result is not None:
                result["timestamp"] = action.timestamp
                result["final"] = action.type == ThreadActionType.End

            # Pass the results back to the external
            session_asset: Optional[SessionAsset] = self._broker._sessions.get(
                sid, None
            )
            if session_asset is None:
                on_intermediate_data = None
                on_end_data = None
            else:
                on_intermediate_data = session_asset.on_intermediate_data
                on_end_data = session_asset.on_end_data
            if (
                action.type == ThreadActionType.Frame
                and on_intermediate_data is not None
            ):
                on_intermediate_data(result)
            elif action.type == ThreadActionType.End and on_end_data is not None:
                on_end_data(result)

            # Indicate that the previous task is done. Only useful if calling `self.__queue.join()` in the future
            self.__queue.task_done()

        # Here exited the loop when assigning an "end" action to all models.
        # Stilll need to wait for all models to finish processing the "end" action and exit.
        for mw in self.__model_workers:
            mw.join()

        logger.debug("ModelManagerWorker exited")


@dataclass
class SessionAsset:
    sid: Hashable
    ending: bool = False
    on_intermediate_data: Optional[OnDataCallback] = None
    on_end_data: Optional[OnDataCallback] = None


class Broker:
    def __init__(
        self,
        models: list[Type[BaseModel]],
        on_intermediate_data: Optional[OnDataCallback],
        on_end_data: Optional[OnDataCallback],
    ) -> None:
        self._sessions: dict[Hashable, SessionAsset] = {}
        self._models = models
        self._on_intermediate_data = on_intermediate_data
        self._on_end_data = on_end_data

        self.manager_thread = ModelManagerWorker(
            self._models,
            self,
        )
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
        logger.debug("Broker(%s) told ML models to start", sid)

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

    def frame(self, sid: Hashable, data: np.ndarray, timestamp: int):
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


class ModelWorker(Thread):
    """A thread for a model"""

    def __init__(self, model: BaseModel, barrier: Barrier, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # This is the queue local to each model.
        # It should only contain one item, which is the item to be processed.
        # The other queue handled by ModelManagerWorker contains further
        #   incoming items, and they are passed here on demand.
        # Theoretically, we only need a public variable. But a queue allows us
        #   to NOT immediately start processing the next frame (so that the
        #   manager can safely get last frame's results first)
        self.queue = SimpleQueue[ThreadAction]()

        # The return value and timestamp of the current frame
        # The manager gets this value by simply accessing this variable.
        self.output = None
        self.timestamp = None

        self.__model = model
        self.__barrier = barrier

        self.__started = False  # Thread has an attr called _started. Careful not to override it haha!
        self.__scheduled_end = False

    def run(self) -> None:
        logger.debug(f"Worker started for model {self.__model.name}.")

        while not self.__scheduled_end:
            # Wait until a new action is assigned by the manager
            action = self.queue.get()
            sid = action.sid

            if action.type == ThreadActionType.Frame:
                if not self.__started:
                    logger.warning("In session (%s), skipping frame before start", sid)
                    continue
                if action.data is None or action.timestamp is None:
                    logger.warning(
                        "In session (%s), skipping frame with incomplete data %s %s",
                        sid,
                        action.data.shape if action.data is not None else None,
                        action.timestamp,
                    )
                    continue
                self.output = self.__model.frame(sid, action.data, action.timestamp)
                self.timestamp = action.timestamp

            elif action.type == ThreadActionType.Start:
                if self.__started:
                    logger.warning(
                        "In session (%s), skipping repetitive start action", sid
                    )
                    continue
                if action.timestamp is None:
                    logger.warning(
                        "In session (%s), skipping start action without timestamp",
                        sid,
                    )
                    continue
                self.output = self.__model.start(sid, action.timestamp)
                self.timestamp = action.timestamp
                self.__started = True

            elif action.type == ThreadActionType.End:
                if not self.__started:
                    logger.warning(
                        "In session (%s), skipping End action before Start", sid
                    )
                    continue
                self.__scheduled_end = True
                self.output = self.__model.end(sid, action.timestamp)
                self.timestamp = action.timestamp
                self.__started = False

            # Wait until other models finish as well
            logger.debug(
                f"Before Model {self.__model.name} waits for barrier, there are threads {self.__barrier.n_waiting} waiting"
            )
            self.__barrier.wait()

        logger.debug(f"Worker exited for model {self.__model.name}")


if __name__ == "__main__":
    from models.mock_model_1 import MockModel1
    from models.mock_model_2 import MockModel2

    logger.setLevel(logging.DEBUG)

    print("Running this file is a test.")

    ### Things defined by the external
    models = [
        MockModel1,
        MockModel2,
    ]

    def on_intermediate_data(result: Optional[dict]):
        print(f"The external received intermediate data, {result}")

    def on_end_data(result: Optional[dict]):
        print(f"The external received final data, {result}")

    broker = Broker(models, on_intermediate_data, on_end_data)

    mock_sid = "test connection"

    def make_mock_image(i=0):
        return np.full((24, 24, 3), i)

    def mock_network_thread():
        broker.start_session(mock_sid, time.time_ns() // 1_000_000)
        for i in range(5):
            time.sleep(0.1)
            broker.frame(mock_sid, make_mock_image(i), time.time_ns() // 1_000_000)
        time.sleep(2.5)
        broker.end_session(mock_sid, time.time_ns() // 1_000_000)
        time.sleep(1)

    network_thread = Thread(target=mock_network_thread, name="Mock Network Thread")
    network_thread.start()
    network_thread.join()

    print()
    print("start a sleep and see when threads get GC-ed")
    hi_thread = Timer(1.0, lambda: print("1 second passed..."))
    hi_thread.start()
    time.sleep(3)
    print(
        "After 3 seconds, broker's thread list contains",
        len(broker._sessions),
        "sessions",
    )
