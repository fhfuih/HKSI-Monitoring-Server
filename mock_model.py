import time
import random
from typing import Optional

from PIL import Image


def foward_single_frame_1(frame: Image.Image, timestamp: int) -> Optional[dict]:
    """Mock the first model's forward pass.

    Parameters
    ----------
    frame : Image.Image
        The image frame to process.
    timestamp : int
        The timestamp **in milliseconds** when the frame was received.
        This is NOT the same as Python's default timestamp behavior (which is in seconds).

    Returns
    -------
    Optional[dict]
        The prediction result. If the model does not output a prediction at this step, return None.
    """
    # sleep a random time between 0.5 second and 2 second, don't use asyncio
    sleep_time = random.uniform(0.5, 2)
    time.sleep(sleep_time)
    not_none = random.choice((None, True, True, True, True))
    return not_none and {
        "HR": "1",
        # The following fields are not required. They are added for demonstration purposes.
        "HR_resp_ts": time.time(),
        "HR_process_time": sleep_time,
    }


def foward_single_frame_2(frame: Image.Image, timestamp: int) -> Optional[dict]:
    """Mock the second model's forward pass.

    Parameters
    ----------
    frame : Image.Image
        The image frame to process.
    timestamp : int
        The timestamp **in milliseconds** when the frame was received.
        This is NOT the same as Python's default timestamp behavior (which is in seconds).

    Returns
    -------
    Optional[dict]
        The prediction result. If the model does not output a prediction at this step, return None.
    """
    # sleep a random time between 0.5 second and 2 second, don't use asyncio
    sleep_time = random.uniform(0.5, 2)
    time.sleep(sleep_time)
    not_none = True
    return not_none and {
        "fatigue": "2",
        # The following fields are not required. They are added for demonstration purposes.
        "fatigue_resp_ts": time.time(),
        "fatigue_process_time": sleep_time,
    }
