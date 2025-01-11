import logging
import time
from typing import Hashable, Optional

import numpy as np

from models.base_model import BaseModel

from .pimple_detection import PimpleDetection

logger = logging.getLogger("HKSI WebRTC")


class PimpleModel(BaseModel):
    name = "Pimple"

    def __init__(self):
        super().__init__()
        # ckpt_path = "shape_predictor_81_face_landmarks.dat"
        # pimple_detector = PimpleDetection(ckpt_path)
        self.pimple_detector = PimpleDetection("shape_predictor_81_face_landmarks.dat")

        self.pimple_num = 0
        # self.pimple_bboxes = []
        self.second_records_pimple = []

    def start(self, sid: Hashable, timestamp: Optional[int], *args, **kwargs) -> None:
        logger.debug(
            f"{self.name} started at {timestamp or 'unknown time'} with sid {sid}"
        )
        self.pimple_num = 0
        # self.pimple_bboxes = []
        self.second_records_pimple = []

    def end(self, sid: Hashable, timestamp: Optional[int], *args, **kwargs) -> dict:
        logger.debug(
            f"{self.name} ended at {timestamp or 'unknown time'} with sid {sid}"
        )

        # Example: return a final conclusive value (e.g., the average over the 30 seconds)
        return {
            # "pimples": {
            #     "count": self.pimple_num,
            #     "coordinates": self.pimple_bboxes,
            # },
            "pimpleCount": self.pimple_num
        }

    def frame(
        self, sid: Hashable, frame: np.ndarray, timestamp: int, *args, **kwargs
    ) -> Optional[dict]:
        logger.debug(f"{self.name} start processing sid({sid})'s frame@{timestamp}")

        # sleep_time = 1  # random.uniform(0.5, 2)
        # time.sleep(sleep_time)

        # Demonstrate the usage of helper functions/classes in another file.
        face_exist, pimple_num, pimple_bboxes = self.pimple_detector.run(frame)
        if not face_exist:
            return {"pimpleCount": None}

        # self.pimple_num = pimple_num
        # self.pimple_bboxes = pimple_bboxes
        self.second_records_pimple.append(pimple_num)
        # logger.debug(f"{self.second_records_pimple} self.second_records_pimple")

        if len(self.second_records_pimple) == 30:
            self.pimple_num = int(np.round(np.mean(self.second_records_pimple)))
            self.second_records_pimple = []
            # logger.debug(f"{self.pimple_num} 30 frames pimple")

        logger.debug(f"{self.name} finish processing sid({sid})'s frame@{timestamp}")

        return {
            "sid": sid,
            # "pimples": {
            #     "count": self.pimple_num,
            #     "coordinates": self.pimple_bboxes,
            # },
            "pimpleCount": self.pimple_num
        }
