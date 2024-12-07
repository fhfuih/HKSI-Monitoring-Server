"""
An example model that is self-contained in a single .py file.
"""

import logging
import random
import time
from typing import Hashable, Optional

from models.base_model import *

logger = logging.getLogger("HKSI WebRTC")


class MockModel1(BaseModel):
    name = "Model1"

    def start(self, sid: Hashable, timestamp: Optional[int], *args, **kwargs) -> None:
        logger.debug(
            f"{self.name} started at {timestamp or 'unknown time'} with sid {sid}"
        )

    def end(self, sid: Hashable, timestamp: Optional[int], *args, **kwargs) -> dict:
        logger.debug(
            f"{self.name} ended at {timestamp or 'unknown time'} with sid {sid}"
        )

        # Example: return a final conclusive value (e.g., the average over the 30 seconds)
        return {
            "hr": timestamp,
            "hrv": timestamp,
        }

    def frame(
        self, sid: Hashable, frame: np.ndarray, timestamp: int, *args, **kwargs
    ) -> Optional[dict]:
        logger.debug(f"{self.name} start processing sid({sid})'s frame@{timestamp}")

        sleep_time = 0.2
        time.sleep(sleep_time)

        logger.debug(f"{self.name} finish processing sid({sid})'s frame@{timestamp}")

        # Example: return a value when finishing a certain frame
        return {
            "hr": timestamp / 1000,
            "hrv": timestamp,
        }
