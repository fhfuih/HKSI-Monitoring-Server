from abc import ABC, abstractmethod
from typing import Hashable, Optional

import numpy as np


class BaseModel(ABC):
    name = "Unnamed model"

    @abstractmethod
    def start(self, sid: Hashable, timestamp: Optional[int], *args, **kwargs) -> None:
        """Start a prediciton session (i.e., a new video stream).

        Parameters
        ----------
        sid : Hashable
            An identifier for the prediction session. Common Hashables can be a string or an integer.
        timestamp : int
            The timestamp **in milliseconds** when the model is started.
            This is NOT the same as Python's default timestamp behavior (which is in seconds).
            NOTE: Currently, it is not used and ALWAYS return None. Because after all the frame timestamp is relative.
        """
        raise NotImplementedError

    @abstractmethod
    def end(self, sid: Hashable, timestamp: Optional[int], *args, **kwargs) -> dict:
        """End a prediction session (i.e., a video stream).

        Parameters
        ----------
        sid : Hashable
            An identifier for the prediction session.
        timestamp : Optional[int]
            The timestamp **in milliseconds** when the model is ended.
            This is NOT the same as Python's default timestamp behavior (which is in seconds).
            NOTE: Currently, it is not used and ALWAYS return None. Because after all the frame timestamp is relative.

        Returns
        -------
        dict
            The overall prediction result of the session.
            For example, the averaged value over the 30 seconds.
        """
        raise NotImplementedError

    @abstractmethod
    def frame(
        self, sid: Hashable, frame: np.ndarray, timestamp: int, *args, **kwargs
    ) -> Optional[dict]:
        """Mock the model's forward pass.

        Parameters
        ----------
        sid : Hashable
            An identifier for the prediction session.
        frame : np.ndarray
            The image frame to process.
        timestamp : int
            The timestamp **in milliseconds** when the frame was received.
            This is NOT the same as Python's default timestamp behavior (which is in seconds).
            NOTE: This is a relative timestamp (e.g., 0, 33, 66, 100, 133, ...). Not an absolute timestamp.

        Returns
        -------
        Optional[dict]
            The prediction result. If the model does not output a prediction at this step, return None.
        """
        raise NotImplementedError
