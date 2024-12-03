"""
An example model that is self-contained in a single .py file.
"""

import random
import time
from datetime import datetime
from typing import Hashable, Optional

from models.base_model import *


class MockModel1(BaseModel):
    name = "Model1"

    def start(self, sid: Hashable, timestamp: int, *args, **kwargs) -> None:
        print(
            f"111 {self.name} started at {datetime.fromtimestamp(timestamp/1000)} with sid {sid}"
        )

    def end(self, sid: Hashable, timestamp: Optional[int], *args, **kwargs) -> dict:
        print(
            f"111 {self.name} ended at {datetime.fromtimestamp(timestamp/1000) if timestamp else 'unknown time'} with sid {sid}"
        )

        # Example: return a final conclusive value (e.g., the average over the 30 seconds)
        return {
            "hr": timestamp,
            "hrv": timestamp,
        }

    def frame(
        self, sid: Hashable, frame: np.ndarray, timestamp: int, *args, **kwargs
    ) -> Optional[dict]:
        print(
            f"111 {self.name} start processing sid({sid})'s frame@{datetime.fromtimestamp(timestamp/1000)}"
        )

        sleep_time = 0.2
        time.sleep(sleep_time)

        print(
            f"111 {self.name} finish processing sid({sid})'s frame@{datetime.fromtimestamp(timestamp/1000)}"
        )

        # Example: return a value when finishing a certain frame
        return {
            "hr": timestamp / 1000,
            "hrv": timestamp,
        }
