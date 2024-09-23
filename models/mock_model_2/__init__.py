"""
An example model that needs to import some utility files,
so that the model definition itself is in __init__.py of a subfolder.
This way, the external can `import mock_model_2`
"""

import time
from datetime import datetime
from typing import Hashable, Optional

from models.base_model import *

from . import lib


class MockModel2(BaseModel):
    name = "Model2"

    def start(self, sid: Hashable, timestamp: int, *args, **kwargs) -> None:
        print(
            f"222 {self.name} started at {datetime.fromtimestamp(timestamp/1000)} with sid {sid}"
        )

    def end(self, sid: Hashable, timestamp: Optional[int], *args, **kwargs) -> dict:
        print(
            f"222 {self.name} ended at {datetime.fromtimestamp(timestamp/1000) if timestamp else 'unknown time'} with sid {sid}"
        )
        a = lib.get_result()
        a["fatigue"] = 0.999
        return a

    def frame(
        self, sid: Hashable, frame: np.ndarray, timestamp: int, *args, **kwargs
    ) -> Optional[dict]:
        print(
            f"222 {self.name} start processing sid({sid})'s frame@{datetime.fromtimestamp(timestamp/1000)}"
        )

        sleep_time = 1  # random.uniform(0.5, 2)
        time.sleep(sleep_time)
        # Demonstrate the usage of helper functions/classes in another file.
        a = lib.get_result()

        print(
            f"222 {self.name} finish processing sid({sid})'s frame@{datetime.fromtimestamp(timestamp/1000)}"
        )
        return {
            **a,
            "sid": sid,
            "fatigue_resp_ts": time.time(),
            "fatigue_process_time": sleep_time,
        }
