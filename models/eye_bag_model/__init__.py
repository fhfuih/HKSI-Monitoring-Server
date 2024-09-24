import time
from datetime import datetime
from typing import Hashable, Optional

from models.base_model import *
from .eye_bag_detection import EyeBagDetection

class EyeBagModel(BaseModel):
    name = "EyeBagDetection"

    ckpt_path = "shape_predictor_81_face_landmarks.dat"
    eye_bag_detector = EyeBagDetection(ckpt_path)

    left_eye_has_bag = False
    right_eye_has_bag = False

    left_eye_bag_region = None
    right_eye_bag_region = None

    def start(self, sid: Hashable, timestamp: int, *args, **kwargs) -> None:
        print(
            f"{self.name} started at {datetime.fromtimestamp(timestamp/1000)} with sid {sid}"
        )

    def end(self, sid: Hashable, timestamp: Optional[int], *args, **kwargs) -> dict:
        print(
            f"{self.name} ended at {datetime.fromtimestamp(timestamp/1000) if timestamp else 'unknown time'} with sid {sid}"
        )

        # Example: return a final conclusive value (e.g., the average over the 30 seconds)
        print(f"left_eye_has_bag: {self.left_eye_has_bag}, right_eye_has_bag: {self.right_eye_has_bag}")
        return {
            "left_eye_has_bag": self.left_eye_has_bag,
            "right_eye_has_bag": self.right_eye_has_bag,
            "left_eye_bag_region": self.left_eye_bag_region,
            "right_eye_bag_region": self.right_eye_bag_region
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
        left_eye_has_bag, right_eye_has_bag, left_eye_bag_region, right_eye_bag_region = self.eye_bag_detector.run(frame)
        
        self.left_eye_has_bag = bool(left_eye_has_bag)
        self.right_eye_has_bag = bool(right_eye_has_bag)

        self.left_eye_bag_region = left_eye_bag_region if self.left_eye_has_bag else None
        self.right_eye_bag_region = right_eye_bag_region if self.right_eye_has_bag else None

        print(
            f"{self.name} finish processing sid({sid})'s frame@{datetime.fromtimestamp(timestamp/1000)}"
        )
        
        return {
            "left_eye_has_bag": self.left_eye_has_bag,
            "right_eye_has_bag": self.right_eye_has_bag,
            "left_eye_bag_region": self.left_eye_bag_region,
            "right_eye_bag_region": self.right_eye_bag_region,
            "sid": sid,
            "eye_bag_resp_ts": time.time(),
            "eye_bag_process_time": sleep_time,
        }