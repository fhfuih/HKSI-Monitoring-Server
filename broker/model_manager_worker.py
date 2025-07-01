import logging
import time
import weakref
from collections import deque
from collections.abc import Hashable
from queue import SimpleQueue
from threading import Semaphore, Thread
from typing import Optional, Tuple, Type, Dict, Any

import numpy.typing as npt

from models.base_model import BaseModel
from models.face_rec_model import FaceRecognitionModel
from services.database import DatabaseService

from .model_worker import ModelWorker
from .types import *

logger = logging.getLogger("HKSI WebRTC")


def compress_list_by_timestamp(
        timestamps: list[int],
        values: list[Optional[float]],
        threshold_ms: int = 1000
) -> Tuple[list[int], list[Optional[float]]]:
    """将时间戳间隔在 threshold_ms 以内的合并为一个，保留第一个非 None 的值"""
    compressed_ts = []
    compressed_vals = []

    prev_ts = None
    for ts, val in zip(timestamps, values):
        if prev_ts is None or ts - prev_ts > threshold_ms:
            compressed_ts.append(ts)
            compressed_vals.append(val)
            prev_ts = ts
        else:
            # 已在前一项中合并；可选择更新规则：
            if compressed_vals[-1] is None and val is not None:
                compressed_vals[-1] = val
            # prev_ts 不变
    return compressed_ts, compressed_vals


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
        self.broker = broker # no sure

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

        # Initialize database connection
        self.db = DatabaseService()

        # Create model workers and start all of them
        self.__n_models = len(models)
        self.__model_workers = [
            ModelWorker(M(self.db) if isinstance(M(), FaceRecognitionModel) else M(), i, self.__models_report_queue)
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

        # Otherwise, the model is processing an `end` or a `frame`. Store the result.
        raw_result = (
            result_report.result.copy() if result_report.result is not None else {}
        )
        # - Inherit last frame's values if this frame doesn't have
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
        person_id = None
        face_embedding = None  # Initialize here
        # participant_id = None
        report_timestamp = int(time.time() * 1000)  # Get timestamp in milliseconds

        for model_report in self.__model_results[sid]:
            if (model_report is not None and
                (raw_result := model_report.result) is not None):
                # Get person_id from face recognition model if available
                if isinstance(self.__model_workers[model_report.model_index]._ModelWorker__model,
                            FaceRecognitionModel):
                    person_id = raw_result.get('person_id')
                    face_embedding = raw_result.get('face_embedding')  # Get the embedding
                    # participant_id = raw_result.get('participant_id')
                # participant_id = raw_result.get('participant_id')
                if combined_result is None:
                    combined_result = {}
                combined_result.update(raw_result)
                is_final = is_final and model_report.is_final

        # If there is no result, do nothing
        if combined_result is None:
            return

        combined_result["final"] = is_final
        combined_result["timestamp"] = report_timestamp  # Add the report timestamp

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

        if not is_final and on_intermediate_data is not None:
            # Store measurements for intermediate results

            # if person_id:
            #     self._store_measurements(person_id, combined_result, combined_result.get("timestamp"))
            on_intermediate_data(combined_result)

        elif is_final and on_end_data is not None:
            logger.info("combined_result:" + str(combined_result))
            participant_id = self.broker.get_participantID()
            if participant_id:
                # first persist this session's values, then reload history including this session
                self._store_measurements(
                    person_id, participant_id,
                    face_embedding,  # Pass the face embedding
                    combined_result,
                    combined_result.get("timestamp"),
                    combined_result.get("final")
                )
                historical_data = self._get_historical_data(person_id, participant_id)
                combined_result['historical_data'] = historical_data
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

    def _store_measurements(self, person_id: str, participant_id: str, face_embedding: Optional[list[float]], results: Dict[str, Any], timestamp: int, is_final: bool = False):
        """Store relevant measurements in the database"""
        # if not person_id:
        #     return
        if not participant_id:
            return

        # Store the face embedding if it exists
        if face_embedding is not None:
            self.db.store_measurement(
                person_id=person_id,
                participant_id=participant_id,
                measurement_type='face_embedding',
                value=face_embedding,
                timestamp=timestamp,
                is_final=is_final
            )

        measurements = {
            'heart_rate': results.get('hr'),
            'heart_rate_variability': results.get('hrv'),
            'fatigue': results.get('fatigue'),
            'darkCircleLeft': results.get('darkCircleLeft'),
            'darkCircleRight': results.get('darkCircleRight'),
            'pimpleCount': results.get('pimpleCount')
        }

        # Store non-null measurements
        for measurement_type, value in measurements.items():
            if value is not None:
                self.db.store_measurement(
                    person_id=person_id,
                    participant_id=participant_id,
                    measurement_type=measurement_type,
                    value=value,
                    timestamp=timestamp,
                    is_final=is_final
                )

    def _get_historical_data(self, person_id: str, participant_id: str) -> Dict[str, Any]:
        """Gather six aligned lists of historical final measurements."""
        if not participant_id:
            return {}

        # Pull raw final measurements per type from MongoDB
        raw_history = self.db.get_person_measurements_summary(person_id, participant_id)
        print("raw_history:", raw_history)

        # Build a sorted list of all distinct session timestamps
        timestamps = sorted({
            m['timestamp']
            for measurements in raw_history.values()
            for m in measurements
        })

        # Map each measurement type to a {timestamp→value} dict
        history_map: Dict[str, Dict[int, Any]] = {}
        for mtype in [
            'heart_rate', 'fatigue',
            'darkCircleLeft', 'darkCircleRight', 'pimpleCount',
            'weight', 'body_fat'
        ]:
            history_map[mtype] = {
                m['timestamp']: m['value']
                for m in raw_history.get(mtype, [])
            }

        # Now assemble each list, in chronological order, inserting None if a session never recorded that metric
        hr_list          = [ history_map['heart_rate'].get(ts)       for ts in timestamps ]
        fatigue_list     = [ history_map['fatigue'].get(ts)          for ts in timestamps ]
        pimple_list      = [ history_map['pimpleCount'].get(ts)      for ts in timestamps ]

        # for dark-circle count, only if both left+right exist, else None
        dark_circle_list: list[Any] = []
        for ts in timestamps:
            left  = history_map['darkCircleLeft'].get(ts)
            right = history_map['darkCircleRight'].get(ts)
            if left is None or right is None:
                dark_circle_list.append(None)
                # dark_circle_list.append(0)
            else:
                dark_circle_list.append(int(bool(left)) + int(bool(right)))

        weight_list      = [ history_map['weight'].get(ts)           for ts in timestamps ]
        body_fat_list    = [ history_map['body_fat'].get(ts)         for ts in timestamps ]

        timestamps_hr, hr_list = compress_list_by_timestamp(timestamps, hr_list, threshold_ms=1000)
        timestamps_fatigue, fatigue_list = compress_list_by_timestamp(timestamps, fatigue_list, threshold_ms=1000)
        timestamps_pimple, pimple_list = compress_list_by_timestamp(timestamps, pimple_list, threshold_ms=1000)
        timestamps_dark_circle, dark_circle_list = compress_list_by_timestamp(timestamps, dark_circle_list, threshold_ms=1000)
        timestamps_weight, weight_list = compress_list_by_timestamp(timestamps, weight_list, threshold_ms=1000)
        timestamps_body_fat, body_fat_list = compress_list_by_timestamp(timestamps, body_fat_list, threshold_ms=1000)


        return {
            'hrList'          : hr_list,
            'fatigueList'     : fatigue_list,
            'darkCircleList'  : dark_circle_list,
            'pimpleCountList' : pimple_list,
            'weightList'      : weight_list,
            'bodyFatList'     : body_fat_list
        }
