"""
An example model that is self-contained in a single .py file.
"""

import random
import time
from datetime import datetime
from typing import Hashable, Optional

from models.base_model import *
from .pimple_detection import PimpleDetection

class MockModel1(BaseModel):
    name = "PimpleDetection"

    ckpt_path = "shape_predictor_81_face_landmarks.dat"
    pimple_detector = PimpleDetection(ckpt_path)
    def start(self, sid: Hashable, timestamp: int, *args, **kwargs) -> None:
        print("Timestamp: ", timestamp)
        print(
            f"111 {self.name} started at {datetime.fromtimestamp(timestamp/1000)} with sid {sid}"
        )

    def end(self, sid: Hashable, timestamp: Optional[int], *args, **kwargs) -> dict:
        print(
            f"111 {self.name} ended at {datetime.fromtimestamp(timestamp/1000) if timestamp else 'unknown time'} with sid {sid}"
        )

        # Example: return a final conclusive value (e.g., the average over the 30 seconds)
        return {
            "HR": "1",
        }

    # def frame(
    #     self, sid: Hashable, frame: np.ndarray, timestamp: int, *args, **kwargs
    # ) -> Optional[dict]:
    #     print(
    #         f"111 {self.name} start processing sid({sid})'s frame@{datetime.fromtimestamp(timestamp/1000)}"
    #     )

    #     sleep_time = 0.3  # random.uniform(0.5, 2)
    #     time.sleep(sleep_time)
    #     not_none = random.choice((None, True, True, True, True))

    #     print(
    #         f"111 {self.name} finish processing sid({sid})'s frame@{datetime.fromtimestamp(timestamp/1000)}"
    #     )

    #     # Example: return a value when finishing a certain frame
    #     return not_none and {
    #         "HR": "1",
    #         "sid": sid,
    #         "HR_resp_ts": time.time(),
    #         "HR_process_time": sleep_time,
    #     }

    def frame(
        self, sid: Hashable, frame: np.ndarray, timestamp: int, *args, **kwargs
    ) -> Optional[dict]:
        print(
            f"111 {self.name} start processing sid({sid})'s frame@{datetime.fromtimestamp(timestamp/1000)}"
        )

        sleep_time = 1  # random.uniform(0.5, 2)
        time.sleep(sleep_time)
        # Demonstrate the usage of helper functions/classes in another file.
        pimple_num, pimple_bboxes = self.pimple_detector.run(frame)
        
        result = {
            "pimple_num": pimple_num,
            "pimple_bboxes": pimple_bboxes
        }
        
        print(
            f"111 {self.name} finish processing sid({sid})'s frame@{datetime.fromtimestamp(timestamp/1000)}"
        )
        
        return {
            **result,
            "sid": sid,
            "pimple_resp_ts": time.time(),
            "pimple_process_time": sleep_time,
        }