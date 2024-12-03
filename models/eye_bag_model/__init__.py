import time
from datetime import datetime
from typing import Hashable, Optional

import numpy as np

from models.base_model import BaseModel

from .eye_bag_detection import EyeBagDetection


class EyeBagModel(BaseModel):
    name = "EyeBagDetection"

    def __init__(self):
        super().__init__()

        # ckpt_path = "shape_predictor_81_face_landmarks.dat"
        # eye_bag_detector = EyeBagDetection(ckpt_path)
        self.eye_bag_detector = EyeBagDetection("shape_predictor_81_face_landmarks.dat")

        self.left_eye_has_bag = False
        self.right_eye_has_bag = False

        self.left_eye_bag_region = None
        self.right_eye_bag_region = None

    def start(self, sid: Hashable, timestamp: int, *args, **kwargs) -> None:
        print(
            f"{self.name} started at {datetime.fromtimestamp(timestamp/1000)} with sid {sid}"
        )
        self.left_eye_has_bag = False
        self.right_eye_has_bag = False

        self.left_eye_bag_region = None
        self.right_eye_bag_region = None

    def end(self, sid: Hashable, timestamp: Optional[int], *args, **kwargs) -> dict:
        print(
            f"{self.name} ended at {datetime.fromtimestamp(timestamp/1000) if timestamp else 'unknown time'} with sid {sid}"
        )

        return {
            "darkCircles": {
                "left": self.left_eye_bag_region,
                "right": self.right_eye_bag_region,
            }
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
        (
            left_eye_has_bag,
            right_eye_has_bag,
            left_eye_bag_region,
            right_eye_bag_region,
        ) = self.eye_bag_detector.run(frame)

        self.left_eye_has_bag = bool(left_eye_has_bag)
        self.right_eye_has_bag = bool(right_eye_has_bag)

        self.left_eye_bag_region = (
            left_eye_bag_region if self.left_eye_has_bag else None
        )
        self.right_eye_bag_region = (
            right_eye_bag_region if self.right_eye_has_bag else None
        )

        print(
            f"{self.name} finish processing sid({sid})'s frame@{datetime.fromtimestamp(timestamp/1000)}"
        )

        return {
            "darkCircles": {
                "left": self.left_eye_bag_region,
                "right": self.right_eye_bag_region,
            }
        }
