import logging
import time
from collections import deque
from datetime import datetime
from typing import Hashable, Optional

import numpy as np
import torch

from models.base_model import BaseModel
from models.utils import GPU, datetime_from_ms

from . import utils


logger = logging.getLogger("HKSI WebRTC")


class FatigueModel(BaseModel):
    name = "Fatigue"

    def __init__(self):
        super().__init__()

        self.model, self.tokenizer = utils.load_model()
        self.generation_config = utils.get_generation_config()
        self.frame_buffer = deque(maxlen=12)
        self.frame_count = 0  # counts all frames
        self.stored_count = 0  # counts stored frames
        self.skip_frames = 10  # store every 10th frame (2.5 fps from 25 fps)
        self.rating = -1
        self.confidence = -1
        self.previous_result = None

    def start(self, sid: Hashable, timestamp: Optional[int], *args, **kwargs) -> None:
        logger.debug(
            f"{self.name} started at {timestamp or 'unknown time'} with sid {sid}"
        )
        self.frame_buffer.clear()
        self.frame_count = 0
        self.stored_count = 0
        self.rating = -1
        self.confidence = -1
        self.previous_result = None

    def end(self, sid: Hashable, timestamp: Optional[int], *args, **kwargs) -> dict:
        logger.debug(
            f"{self.name} ended at {timestamp or 'unknown time'} with sid {sid}"
        )
        logger.debug(f"Fatigue rate: {self.rating}")

        # # Free up memory
        # del self.model
        # del self.tokenizer
        # del self.generation_config
        # self.frame_buffer.clear()

        # return {"status": "completed"}
        return {"fatigue": self.rating}


    def frame(
        self, sid: Hashable, frame: np.ndarray, timestamp: int, *args, **kwargs
    ) -> Optional[dict]:
        
        self.frame_count += 1
        
        # Store frame if it meets the skip_frames criteria
        if self.frame_count % self.skip_frames == 1:
            self.frame_buffer.append(frame)
            self.stored_count += 1
            
            # Process only after storing every 3 frames
            if self.stored_count % 12 == 0:
                logger.debug(
                    f"{self.name} start processing sid({sid})'s frames@{timestamp} at {datetime.now()}"
                )

                start_time = time.time()

                # Process the frames
                pixel_values, num_patches_list = utils.process_sample(list(self.frame_buffer))
                pixel_values = pixel_values.to(GPU, dtype=torch.bfloat16)

                video_prefix = "".join(
                    [f"Frame{i*self.skip_frames+1}: <image>\n" for i in range(len(num_patches_list))]
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

                rating = float(response.strip())
                self.rating = (rating - 1) * 0.25

                confidence = utils.get_highest_prob(processed_scores)
                self.confidence = confidence

                process_time = time.time() - start_time

                logger.debug(
                    f"{self.name} finish processing sid({sid})'s frames@{timestamp} at {datetime.now()}"
                )

                result = {
                    "sid": sid,
                    "fatigue": self.rating,
                    "confidence": confidence,
                    "fatigue_resp_ts": time.time(),
                    "fatigue_process_time": process_time,
                }
                self.previous_result = result
                return result

        return self.previous_result
