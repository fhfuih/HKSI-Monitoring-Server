from abc import ABC, abstractmethod
from typing import Hashable, Optional
from PIL import Image


class BaseModel(ABC):
    name = "Unnamed model"

    @abstractmethod
    def start(self, sid: Hashable, timestamp: int, *args, **kwargs) -> None:
        """Start a prediciton session (i.e., a new video stream)

        Parameters
        ----------
        sid : Hashable
            An identifier for the prediction session. Common Hashables can be a string or an integer.
        timestamp : int
            The timestamp **in milliseconds** when the model is started.
            This is NOT the same as Python's default timestamp behavior (which is in seconds).
        """
        raise NotImplementedError

    @abstractmethod
    def end(self, sid: Hashable, timestamp: int, *args, **kwargs) -> None:
        """End a prediction session (i.e., a video stream)

        Parameters
        ----------
        sid : Hashable
            An identifier for the prediction session.
        timestamp : int
            The timestamp **in milliseconds** when the model is ended.
            This is NOT the same as Python's default timestamp behavior (which is in seconds).
        """
        raise NotImplementedError

    @abstractmethod
    def forward_single_frame(
        self, sid: Hashable, frame: Image.Image, timestamp: int, *args, **kwargs
    ) -> Optional[dict]:
        """Mock the model's forward pass.

        Parameters
        ----------
        sid : Hashable
            An identifier for the prediction session.
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
        raise NotImplementedError
