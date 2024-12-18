import logging
import time
import weakref
from collections import deque
from collections.abc import Hashable
from queue import SimpleQueue
from threading import Semaphore, Thread
from typing import Optional, Tuple, Type

import numpy.typing as npt

from models.base_model import BaseModel

from .model_worker import ModelWorker
from .types import *

logger = logging.getLogger("HKSI WebRTC")


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
        # A reference to the broker
        # Note that ModelManagerWorker should expect to exit before the broker object is destroyed
        # This is expected for now because ModelManagerWorker is a thread object created by broker itself
        # But if it no longer is, remember to change to weakref to allow python GC broker object
        self.__broker = weakref.ref(broker)

        # Store start and end actions (sid -> action)
        self.__start_actions: dict[Hashable, ModelAction] = {}
        self.__end_actions: dict[Hashable, ModelAction] = {}

        # Store all incoming frame actions (sid -> frame index -> action)
        self.__frames: dict[Hashable, deque[ModelAction]] = {}

        # For model workers to set the latest results and periodically report results to the broker (sid -> model index -> result)
        self.__model_results: dict[Hashable, list[Optional[ModelResultReport]]] = {}
        self.__last_report_time: float = time.time()  # (in seconds)

        # Record the progress of each model (sid -> model index -> deque index)
        # -3 means end, -2 means resting (created but not started), -1 means start, 0 means the first frame, etc.
        self.__model_progress: dict[Hashable, list[int]] = {}

        # A queue for models to submit previous results and get next actions
        self.__models_report_queue = SimpleQueue[ModelResultReport]()

        # Create model workers and start all of them
        self.__n_models = len(models)
        self.__model_workers = [
            ModelWorker(M(), i, self.__models_report_queue)
            for (i, M) in enumerate(models)
        ]

        # Start all model workers
        for mw in self.__model_workers:
            mw.start()

    def __del__(self):
        logger.debug("__del__ ModelManagerWorker")

    ### Public methods
    def add_start(self, sid: Hashable, timestamp: Optional[int]):
        # If already started, do nothing
        if sid in self.__start_actions:
            logger.debug(
                "ModelManagerWorker: Session %s already started and trying to start again",
                sid,
            )
            return

        # add to start action
        start_action = ModelAction(ModelActionType.Start, sid, timestamp, None)
        self.__start_actions[sid] = start_action

        # Create a deque to store frames of this connection
        self.__frames[sid] = deque[ModelAction]()

        # Create model progress record for this connection
        self.__model_progress[sid] = [-1] * self.__n_models

        # Create result containers
        self.__model_results[sid] = [None] * self.__n_models

        # Put the start action to model workers
        for mw in self.__model_workers:
            mw.queue.put(ModelActionWithProgress(start_action, -1))

    def add_end(self, sid: Hashable, timestamp: Optional[int]):
        # if already ended, do nothing
        if sid in self.__end_actions:
            return

        # add to end action
        self.__end_actions[sid] = ModelAction(ModelActionType.End, sid, timestamp, None)

    def add_frame(self, sid: Hashable, timestamp: int, data: npt.NDArray):
        # If there is no deque created for this session (meaning it has not started), do nothing
        if (q := self.__frames.get(sid, None)) is None:
            return

        # If this session is already ended, do nothing
        if sid in self.__end_actions:
            return

        q.append(ModelAction(ModelActionType.Frame, sid, timestamp, data))

    ### Private fuctions
    def __set_result(self, result_report: ModelResultReport) -> None:
        sid = result_report.sid
        model_index = result_report.model_index

        # Get model's current progress
        if sid not in self.__model_progress:
            return
        progress = self.__model_progress[sid][model_index]

        # No result to save for the start action, or the model is not started yet.
        if progress == -1 or progress == -2:
            return

        # Otherwise, he model is processing end or frame. Store the result
        raw_result = (
            result_report.result.copy() if result_report.result is not None else {}
        )
        if (
            previous_result_report := self.__model_results[sid][model_index]
        ) is not None and previous_result_report.result is not None:
            for k, v in previous_result_report.result.items():
                if k not in raw_result or raw_result[k] is None:
                    raw_result[k] = v
        self.__model_results[sid][model_index] = ModelResultReport(
            sid=result_report.sid,
            model_index=result_report.model_index,
            progress=result_report.progress,
            result=raw_result,
            is_final=result_report.is_final,
        )

    def __report_results(self, sid: Hashable) -> None:
        # Combine all results into a single dict
        combined_result = None
        is_final = True
        for model_report in self.__model_results[sid]:
            if (
                model_report is not None
                and (raw_result := model_report.result) is not None
            ):
                if combined_result is None:
                    combined_result = {}
                combined_result.update(raw_result)
                is_final = is_final and model_report.is_final

        # If there is no result, do nothing
        if combined_result is None:
            return

        combined_result["final"] = is_final

        # Report the result to the broker
        broker = self.__broker()
        if broker is None:
            return
        session_asset = broker._sessions.get(sid, None)
        if session_asset is None:
            on_intermediate_data = None
            on_end_data = None
        else:
            on_intermediate_data = session_asset.on_intermediate_data
            on_end_data = session_asset.on_end_data
        if (not is_final) and on_intermediate_data is not None:
            on_intermediate_data(combined_result)
        elif is_final and on_end_data is not None:
            on_end_data(combined_result)

    def __progress(
        self, sid: Hashable, model_index: int
    ) -> tuple[Optional[ModelAction], Optional[int], bool]:
        """
        Get the next action for the model. Also update the model's progress.

        The second return value indicates the next progress of the model.

        The third return value indicates whether None action is because new frames haven't arrived yet.
        If it is True, it is suggested to retry after a while.
        """
        # Get model current progress
        if sid not in self.__model_progress:
            return (None, None, False)
        progress = self.__model_progress[sid][model_index]

        # Whenever an end action is received and the model is running & not ended, end the model
        if sid in self.__end_actions and progress != -3 and progress != -2:
            self.__model_progress[sid][model_index] = -3
            return (self.__end_actions[sid], -3, False)

        # If the model was already ended, do nothing
        if progress == -3:
            return (None, None, False)

        # If the model was in rest, feed the starting action
        if progress == -2:
            self.__model_progress[sid][model_index] = -1
            return (self.__start_actions[sid], -1, False)

        # If the model was processing the start action, give the first frame (-1 -> 0)
        # If the model was processing a frame, give the next frame (n -> n+1)
        # But need to check whether the next frame has been added
        if (frames := self.__frames.get(sid, None)) is None:
            return (None, None, False)
        next_progress = progress + 1
        if next_progress >= len(frames):
            return (None, None, True)
        self.__model_progress[sid][model_index] = next_progress
        return (frames[next_progress], next_progress, False)

    ### Thread routine
    def run(self) -> None:
        while True:
            # Wait until a new result is available
            result_report = self.__models_report_queue.get()

            # Save current action's result
            self.__set_result(result_report)

            # Throttle the result reporting such that
            # - It reports every 0.5 second
            # - It reports when the result is final
            current_time = time.time()
            if current_time - self.__last_report_time > 0.5 or result_report.is_final:
                self.__report_results(result_report.sid)
                self.__last_report_time = current_time

            # Get next action
            action, new_progress, should_retry = self.__progress(
                result_report.sid, result_report.model_index
            )

            if should_retry:
                # Frame is not available, but because new frames haven't arrived. Retry later.
                ## A hacky way to retry is to wait 1/30 second
                time.sleep(1 / 30)
                self.__models_report_queue.put(result_report)
            elif action is not None and new_progress is not None:
                # When the next action is available, send it to the corresponding model worker
                action_with_progress = ModelActionWithProgress(action, new_progress)
                self.__model_workers[result_report.model_index].queue.put(
                    action_with_progress
                )
