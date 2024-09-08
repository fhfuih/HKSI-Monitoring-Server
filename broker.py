import logging
import time
from dataclasses import dataclass
from enum import Enum
from queue import Queue, SimpleQueue
from threading import Barrier, Thread, Timer
from typing import Callable, NamedTuple, Optional, Type

import numpy as np

from models.base_model import BaseModel

logger = logging.getLogger("BrokerDebug")
logging.basicConfig(level=logging.WARNING)


ThreadActionType = Enum("ThreadActionType", ["Start", "Frame", "End"])


class ThreadAction(NamedTuple):
    type: ThreadActionType
    timestamp: Optional[int]
    data: Optional[np.ndarray]


ActionQueue = Queue[ThreadAction]
if logger.level == logging.DEBUG:
    setattr(Queue, "__del__", lambda self: logger.debug("GC ActionQueue"))


class ModelManagerWorker(Thread):
    """
    The class that manages all ML models in one consecutive prediction session.
    It exits when receiving an 'end' action, forwarding it to all managed models, and waiting all models to exit.
    """

    def __init__(
        self,
        models: list[Type[BaseModel]],
        sid: str,
        on_exit: Callable,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.__sid = sid
        self.__on_exit = on_exit

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
            ModelWorker(M(), sid, self.__barrier, *args, **kwargs) for M in models
        ]

        for mw in self.__model_workers:
            mw.start()

    def __del__(self):
        logger.debug("GC ModelManagerWorker")

    def run(self) -> None:
        while not self.__scheduled_end:
            # Waits until another action is received
            action = self.__queue.get()

            # Detect if this is an "end" action
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

            # Indicate that the previous task is done. Only useful in conjunction with `queue.join()`
            self.__queue.task_done()

        # Exiting the loop when assigning the "end" action to all models.
        # Wait for them to actually end.
        for mw in self.__model_workers:
            mw.join()
        self.__on_exit()
        logger.debug("ModelManagerWorker exited")

    def add_start(self, timestamp: int):
        self.__queue.put(ThreadAction(ThreadActionType.Start, timestamp, None))

    def add_end(self, timestamp: Optional[int]):
        # Remove all queued frames
        while not self.__queue.empty():
            self.__queue.get()

        # Insert the end frame
        self.__queue.put(ThreadAction(ThreadActionType.End, timestamp, None))

    def add_frame(self, timestamp: int, data: np.ndarray):
        self.__queue.put(ThreadAction(ThreadActionType.Frame, timestamp, data))


@dataclass
class SessionAsset:
    sid: str
    manager_thread: ModelManagerWorker
    ending: bool = False


class Broker:
    def __init__(
        self,
        models: list[Type[BaseModel]],
    ) -> None:
        self._sessions: dict[str, SessionAsset] = {}
        self._models = models

    def start_session(self, sid: str, timestamp: int):
        if sid in self._sessions:
            raise KeyError(f"A session with id {sid} already exists.")

        # Clear broker's reference to the session's assets after the thread ends.
        # Done via a callback, because we don't want to `manager_thread.join(); del` in end_session
        # The latter would block the end_session function and the thread that calls it
        # Maybe(?) that will block the network event handling things.
        def on_thread_stop():
            try:
                del self._sessions[sid]
            except KeyError:
                pass

        # Prepare the session asset.
        manager_thread = ModelManagerWorker(self._models, sid, on_thread_stop)
        self._sessions[sid] = SessionAsset(sid=sid, manager_thread=manager_thread)

        # Tell ML models to start
        manager_thread.add_start(timestamp)

        # Start the manager thread
        manager_thread.start()

    def end_session(self, sid: str, timestamp: Optional[int] = None):
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
        session_asset.manager_thread.add_end(timestamp)

    def frame(self, sid: str, data: np.ndarray, timestamp: int):
        session_asset = self._sessions.get(sid, None)
        if session_asset is None:
            # Cannot find such a session. All good.
            return

        if session_asset.ending:
            # Don't add more data into a session that is about to end.
            return

        # Tell ML models to queue this frame
        session_asset.manager_thread.add_frame(timestamp, data)


class ModelWorker(Thread):
    def __init__(
        self, model: BaseModel, sid: str, barrier: Barrier, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        # This is the queue local to each model worker
        self.queue = SimpleQueue[ThreadAction]()

        self.__model = model
        self.__sid = sid
        self.__barrier = barrier

        self.__started = False  # Thread has an attr called _started. Careful not to override it haha!
        self.__scheduled_end = False

    def run(self) -> None:
        logger.debug(f"Worker started for model {self.__model.name}")
        while not self.__scheduled_end:
            # Wait until a new action is assigned by the manager
            action = self.queue.get()
            if action.type == ThreadActionType.Frame:
                if not self.__started:
                    # TODO: log something: should run a Start action first
                    continue
                if action.data is None or action.timestamp is None:
                    # TODO: log someting: should contain data
                    continue
                self.__model.frame(self.__sid, action.data, action.timestamp)

            elif action.type == ThreadActionType.Start:
                if self.__started:
                    logger.warning(
                        "In session (%s), skipping repetitive start action", self.__sid
                    )
                    continue
                if action.timestamp is None:
                    logger.warning(
                        "In session (%s), skipping start action without timestamp",
                        self.__sid,
                    )
                    continue
                self.__model.start(self.__sid, action.timestamp)
                self.__started = True

            elif action.type == ThreadActionType.End:
                if not self.__started:
                    logger.warning(
                        "In session (%s), skipping End action before Start", self.__sid
                    )
                    continue
                self.__scheduled_end = True
                self.__model.end(self.__sid, action.timestamp)
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
    models = [
        MockModel1,
        MockModel2,
    ]
    broker = Broker(models=models)

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
