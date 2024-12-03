import logging
import weakref
from collections.abc import Hashable
from queue import SimpleQueue
from threading import Thread
from typing import NamedTuple

from models.base_model import BaseModel

from .types import *

logger = logging.getLogger("HKSI WebRTC")


class ModelWorker(Thread):
    """A thread for a model"""

    def __init__(
        self,
        model: BaseModel,
        model_id: int,
        report_queue: SimpleQueue,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        # This is the queue local to each model.
        # It should only contain one item, which is the item to be processed.
        # The other queue handled by ModelManagerWorker contains further
        #   incoming items, and they are passed here on demand.
        # Theoretically, we only need a public variable. But a queue allows us
        #   to NOT immediately start processing the next frame (so that the
        #   manager can safely get last frame's results first)
        self.queue = SimpleQueue[ModelActionWithProgress]()

        self.__model = model
        self.__model_id = model_id
        self.__report_queue = weakref.ref(report_queue)

    def __del__(self):
        logger.debug(
            "__del__ ModelWorker for %s %d", self.__model.name, self.__model_id
        )

    def run(self) -> None:
        logger.debug(f"Worker started for model {self.__model.name}.")

        while True:
            # Wait until a new action is assigned by the manager
            action_with_progress = self.queue.get()
            action = action_with_progress.action
            progress = action_with_progress.progress
            sid = action.sid

            # Actually run the model
            if action.type == ModelActionType.Frame:
                if action.data is None or action.timestamp is None:
                    logger.warning(
                        "In session (%s), skipping frame with incomplete data %s %s",
                        sid,
                        action.data.shape if action.data is not None else None,
                        action.timestamp,
                    )
                    continue
                raw_result = self.__model.frame(sid, action.data, action.timestamp)

            elif action.type == ModelActionType.Start:
                if action.timestamp is None:
                    logger.warning(
                        "In session (%s), skipping start action without timestamp",
                        sid,
                    )
                    continue
                raw_result = self.__model.start(sid, action.timestamp)

            elif action.type == ModelActionType.End:
                raw_result = self.__model.end(sid, action.timestamp)

            # Report the result
            report = ModelResultReport(
                sid=sid,
                model_index=self.__model_id,
                progress=progress,
                result=raw_result,
                is_final=action.type == ModelActionType.End,
            )
            report_queue = self.__report_queue()
            if report_queue is not None:
                report_queue.put(report)
            else:
                logger.warning(
                    "ModelWorker: The report queue is already deleted while processing action %s at timestamp %s and progress %d. Discarding the result and exiting the thread.",
                    action.type,
                    action.timestamp,
                    progress,
                )
                break
