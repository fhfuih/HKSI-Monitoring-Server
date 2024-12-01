import time
from collections import deque
from datetime import datetime
from typing import Hashable, Optional

import numpy as np
import torch

from models.base_model import BaseModel
from models.utils import GPU, datetime_from_ms

from . import utils


class FatigueModel(BaseModel):
    name = "FatigueModel"

    def __init__(self):
        super().__init__()

        self.model, self.tokenizer = utils.load_model()
        self.generation_config = utils.get_generation_config()
        self.frame_buffer = deque(maxlen=24)
        self.frame_count = 0
        self.skip_frames = 10
        self.rating = -1

    def start(self, sid: Hashable, timestamp: int, *args, **kwargs) -> None:
        print(
            f"{self.name} started at {datetime.fromtimestamp(timestamp/1000)} with sid {sid}"
        )
        self.frame_buffer.clear()
        self.frame_count = 0
        self.rating = -1

    def end(self, sid: Hashable, timestamp: Optional[int], *args, **kwargs) -> dict:
        print(
            f"{self.name} ended at {datetime.fromtimestamp(timestamp/1000) if timestamp else 'unknown time'} with sid {sid}"
        )
        print("Fatigue rate: ", self.rating)

        # # Free up memory
        # del self.model
        # del self.tokenizer
        # del self.generation_config
        # self.frame_buffer.clear()

        return {"status": "completed"}

    def frame(
        self, sid: Hashable, frame: np.ndarray, timestamp: int, *args, **kwargs
    ) -> Optional[dict]:
        # self.frame_count += 1

        if self.frame_count % self.skip_frames == 0:
            self.frame_buffer.append(frame)

        self.frame_count += 1

        # if len(self.frame_buffer) < 16:
        #     return None

        print(
            f"{self.name} start processing sid({sid})'s frames@{timestamp} at {datetime.now()}"
        )

        start_time = time.time()

        # Process the frames
        pixel_values, num_patches_list = utils.process_sample(list(self.frame_buffer))

        pixel_values = pixel_values.to(GPU, dtype=torch.bfloat16)

        video_prefix = "".join(
            [f"Frame{i*10+1}: <image>\n" for i in range(len(num_patches_list))]
        )
        question = "Rate the fatigue level of the person in this video segment on a scale from 1 to 5, where 1 is completely fresh and 5 is extremely exhausted. Answer with only a number."
        full_question = video_prefix + question

        with torch.no_grad():
            response, processed_scores = utils.chat(
                self.model,
                self.tokenizer,
                pixel_values,
                full_question,
                self.generation_config,
                num_patches_list=num_patches_list,
                history=None,
                return_history=False,
            )

        # with torch.no_grad():
        #     response, processed_scores = utils.chat(self.model, self.tokenizer, pixel_values, question,
        #                                             self.generation_config, num_patches_list=num_patches_list)

        rating = float(response.strip())
        self.rating = rating

        confidence = utils.get_highest_prob(processed_scores)

        # print("------", rating, confidence, "--------")

        process_time = time.time() - start_time

        print(
            f"{self.name} finish processing sid({sid})'s frames@{timestamp} at {datetime.now()}"
        )

        # return {
        #     "sid": sid,
        #     "fatigue_rating": rating,
        #     "confidence": confidence,
        #     "fatigue_resp_ts": time.time(),
        #     "fatigue_process_time": process_time,
        # }
        return {
            "sid": sid,
            "fatigue": rating,
            "confidence": confidence,
            "fatigue_resp_ts": time.time(),
            "fatigue_process_time": process_time,
        }

