import time
from datetime import datetime
from typing import Hashable, Optional

from models.base_model import *
from .pimple_detection import PimpleDetection

import numpy as np

class PimpleModel(BaseModel):
    name = "PimpleDetection"

    def __init__(self):
        super().__init__()
        # ckpt_path = "shape_predictor_81_face_landmarks.dat"
        # pimple_detector = PimpleDetection(ckpt_path)
        self.pimple_detector = PimpleDetection("shape_predictor_81_face_landmarks.dat")

        self.pimple_num = 0
        self.pimple_bboxes = []

    def start(self, sid: Hashable, timestamp: int, *args, **kwargs) -> None:
        print(
            f"{self.name} started at {datetime.fromtimestamp(timestamp/1000)} with sid {sid}"
        )
        self.pimple_num = 0
        self.pimple_bboxes = []

    def end(self, sid: Hashable, timestamp: Optional[int], *args, **kwargs) -> dict:
        print(
            f"{self.name} ended at {datetime.fromtimestamp(timestamp/1000) if timestamp else 'unknown time'} with sid {sid}"
        )

        # Example: return a final conclusive value (e.g., the average over the 30 seconds)
        print(f"pimple_num: {self.pimple_num}, pimple_bboxes: {self.pimple_bboxes}")
        return {
            "pimple_num": self.pimple_num,
            "pimple_bboxes": self.pimple_bboxes
        }

    def frame(
        self, sid: Hashable, frame: np.ndarray, timestamp: int, *args, **kwargs
    ) -> Optional[dict]:
        print(
            f"{self.name} start processing sid({sid})'s frame@{datetime.fromtimestamp(timestamp/1000)}"
        )

        sleep_time = 1  # random.uniform(0.5, 2)
        time.sleep(sleep_time)

        # Demonstrate the usage of helper functions/classes in another file.
        pimple_num, pimple_bboxes = self.pimple_detector.run(frame)
        
        self.pimple_num = pimple_num
        self.pimple_bboxes = pimple_bboxes

        print(
            f"{self.name} finish processing sid({sid})'s frame@{datetime.fromtimestamp(timestamp/1000)}"
        )
        
        return {
            "pimple_num": self.pimple_num,
            "pimple_bboxes": self.pimple_bboxes,
            "sid": sid,
            "pimple_resp_ts": time.time(),
            "pimple_process_time": sleep_time,
        }