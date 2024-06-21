import time
from datetime import datetime
import random
from typing import Optional, Hashable

from PIL import Image

from models import BaseModel


class MockModel1(BaseModel):
    name = "Model1"

    def start(self, sid: Hashable, timestamp: int) -> None:
        print(
            f"$$$ {self.name} started at {datetime.fromtimestamp(timestamp/1000)} with sid {sid}"
        )

    def end(self, sid: Hashable, timestamp: int) -> None:
        print(
            f"$$$ {self.name} ended at {datetime.fromtimestamp(timestamp/1000)} with sid {sid}"
        )

    def forward_single_frame(
        self, sid: Hashable, frame: Image.Image, timestamp: int
    ) -> Optional[dict]:
        print(
            f"{self.name} start processing sid({sid})'s frame@{datetime.fromtimestamp(timestamp/1000)}"
        )

        # sleep a random time between 0.5 second and 2 second, don't use asyncio
        sleep_time = random.uniform(0.5, 2)
        time.sleep(sleep_time)
        not_none = random.choice((None, True, True, True, True))

        print(
            f"{self.name} finish processing sid({sid})'s frame@{datetime.fromtimestamp(timestamp/1000)}"
        )

        return not_none and {
            "HR": "1",
            # The following fields are not required. They are added for demonstration purposes.
            "sid": sid,
            "HR_resp_ts": time.time(),
            "HR_process_time": sleep_time,
        }


class MockModel2(BaseModel):
    name = "Model2"

    def start(self, sid: Hashable, timestamp: int) -> None:
        print(f"@@@ {self.name} started at {timestamp} with sid {sid}")

    def end(self, sid: Hashable, timestamp: int) -> None:
        print(f"@@@ {self.name} ended at {timestamp} with sid {sid}")

    def forward_single_frame(
        self, sid: Hashable, frame: Image.Image, timestamp: int
    ) -> Optional[dict]:
        # sleep a random time between 0.5 second and 2 second, don't use asyncio
        sleep_time = random.uniform(0.5, 2)
        time.sleep(sleep_time)
        not_none = True
        return not_none and {
            "fatigue": "2",
            # The following fields are not required. They are added for demonstration purposes.
            "sid": sid,
            "fatigue_resp_ts": time.time(),
            "fatigue_process_time": sleep_time,
        }
