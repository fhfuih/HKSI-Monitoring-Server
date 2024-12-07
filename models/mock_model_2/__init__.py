"""
An example model that needs to import some utility files,
so that the model definition itself is in __init__.py of a subfolder.
This way, the external can `import mock_model_2`
"""

import logging
import time
from typing import Hashable, Optional

from models.base_model import *

from . import lib

logger = logging.getLogger("HKSI WebRTC")


class MockModel2(BaseModel):
    name = "Model2"

    def start(self, sid: Hashable, timestamp: Optional[int], *args, **kwargs) -> None:
        logger.debug(
            f"{self.name} started at {timestamp or 'unknown time'} with sid {sid}"
        )

    def end(self, sid: Hashable, timestamp: Optional[int], *args, **kwargs) -> dict:
        logger.debug(
            f"{self.name} ended at {timestamp or 'unknown time'} with sid {sid}"
        )
        a = lib.get_result(timestamp)
        return a

    def frame(
        self, sid: Hashable, frame: np.ndarray, timestamp: int, *args, **kwargs
    ) -> Optional[dict]:
        logger.debug(f"{self.name} start processing sid({sid})'s frame@{timestamp}")

        sleep_time = 1  # random.uniform(0.5, 2)
        time.sleep(sleep_time)
        # Demonstrate the usage of helper functions/classes in another file.
        a = lib.get_result(timestamp)

        logger.debug(f"{self.name} finish processing sid({sid})'s frame@{timestamp}")
        return a
