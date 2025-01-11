import logging
import time
from typing import Hashable, Optional

import numpy as np

from models.base_model import BaseModel

from .eye_bag_detection import EyeBagDetection

logger = logging.getLogger("HKSI WebRTC")


class EyeBagModel(BaseModel):
    name = "DarkCircle"

    def __init__(self):
        super().__init__()

        # ckpt_path = "shape_predictor_81_face_landmarks.dat"
        # eye_bag_detector = EyeBagDetection(ckpt_path)
        self.eye_bag_detector = EyeBagDetection("shape_predictor_81_face_landmarks.dat")

        self.left_eye_has_bag = False
        self.right_eye_has_bag = False

        self.one_second = 0
        self.second_records_right = []
        self.second_records_left = []

        # self.left_eye_bag_region = None
        # self.right_eye_bag_region = None

        # self.__reset_state_for_second()

    def start(self, sid: Hashable, timestamp: Optional[int], *args, **kwargs) -> None:
        logger.debug(
            f"{self.name} started at {timestamp or 'unknown time'} with sid {sid}"
        )
        self.left_eye_has_bag = False
        self.right_eye_has_bag = False

        # self.left_eye_bag_region = None
        # self.right_eye_bag_region = None

        # self.__reset_state_for_second()
        self.one_second = 0
        self.second_records_right = []
        self.second_records_left = []

    def end(self, sid: Hashable, timestamp: Optional[int], *args, **kwargs) -> dict:
        logger.debug(
            f"{self.name} ended at {timestamp or 'unknown time'} with sid {sid}"
        )

        return {
            # "darkCircles": {
            #     "left": self.left_eye_bag_region,
            #     "right": self.right_eye_bag_region,
            # },
            "darkCircleLeft": self.left_eye_has_bag,
            "darkCircleRight": self.right_eye_has_bag
        }

    def frame(
        self, sid: Hashable, frame: np.ndarray, timestamp: int, *args, **kwargs
    ) -> Optional[dict]:
        logger.debug(f"{self.name} start processing sid({sid})'s frame@{timestamp}")

        # sleep_time = 1  # random.uniform(0.5, 2)
        # time.sleep(sleep_time)
        # Demonstrate the usage of helper functions/classes in another file.
        # try:
        #     (
        #         face_exist,
        #         left_eye_has_bag,
        #         right_eye_has_bag,
        #         left_eye_bag_region,
        #         right_eye_bag_region,
        #     ) = self.eye_bag_detector.run(frame)
        # except:
        #     return {
        #         # "darkCircles": {
        #         #     "left": self.left_eye_bag_region,
        #         #     "right": self.right_eye_bag_region,
        #         # },
        #         "sid": sid,
        #         "darkCircleLeft": self.left_eye_has_bag,
        #         "darkCircleRight": self.right_eye_has_bag
        #     }

        (
            face_exist,
            left_eye_has_bag,
            right_eye_has_bag,
            left_eye_bag_region,
            right_eye_bag_region,
        ) = self.eye_bag_detector.run(frame)

        self.one_second += 1

        # logger.debug(f"{self.one_second}{face_exist}{left_eye_has_bag}{right_eye_has_bag} -- self.one_second, face_exist, left_eye_has_bag, right_eye_has_bag")

        if not face_exist:
            # return {"darkCircles": None}
            return {
                # "darkCircles": {
                #     "left": self.left_eye_bag_region,
                #     "right": self.right_eye_bag_region,
                # },
                "sid": sid,
                "darkCircleLeft": self.left_eye_has_bag,
                "darkCircleRight": self.right_eye_has_bag
            }

        current_left_eye_has_bag = int(left_eye_has_bag)
        current_right_eye_has_bag = int(right_eye_has_bag)

        # current_left_eye_bag_region = (
        #     left_eye_bag_region if current_left_eye_has_bag else None
        # )
        # current_right_eye_bag_region = (
        #     right_eye_bag_region if current_right_eye_has_bag else None
        # )

        self.second_records_left.append(current_left_eye_has_bag)
        self.second_records_right.append(current_right_eye_has_bag)
        # logger.debug(f"{self.second_records_left} self.second_records_left")
        # logger.debug(f"{self.second_records_right} self.second_records_right")

        if self.one_second == 30:
            # self.left_eye_has_bag = (np.sum(self.second_records_left == True) >= np.sum(self.second_records_left == False))
            # self.right_eye_has_bag = (np.sum(self.second_records_right == True) >= np.sum(self.second_records_right == False))

            self.left_eye_has_bag = bool(np.sum(self.second_records_left[-30:]) >= 15)
            self.right_eye_has_bag = bool(np.sum(self.second_records_right[-30:]) >= 15)
            # self.left_eye_has_bag = True
            # self.right_eye_has_bag = True

            # logger.debug(f"{self.left_eye_has_bag} 30 frames left")
            # logger.debug(f"{self.right_eye_has_bag} 30 frames right")
            # logger.debug(f"{self.second_records_left[-30:]} self.second_records_left 30")
            # logger.debug(f"{self.second_records_right[-30:]} self.second_records_right 30")

            self.one_second = 0

            # self.__reset_state_for_second()

        logger.debug(f"{self.name} finish processing sid({sid})'s frame@{timestamp}")

        return {
            # "darkCircles": {
            #     "left": self.left_eye_bag_region,
            #     "right": self.right_eye_bag_region,
            # },
            "sid": sid,
            "darkCircleLeft": self.left_eye_has_bag,
            "darkCircleRight": self.right_eye_has_bag
        }

    # def __reset_state_for_second(self):
    #     self.second_records_right = []
    #     self.second_records_left = []