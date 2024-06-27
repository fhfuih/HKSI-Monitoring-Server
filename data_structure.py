from io import BytesIO
from datetime import datetime
import logging

from PIL import Image, UnidentifiedImageError


logger = logging.getLogger("HKSI")


class FrameData:
    def __init__(self, raw_data: bytes, connection_id: str, session_id: str) -> None:
        self.timestamp = get_timestamp(raw_data[:8])
        self.frame = get_image(raw_data[8:])
        self.sid = session_id
        self.cid = connection_id

    def __del__(self) -> None:
        logger.debug(
            f"FrameData at {datetime.fromtimestamp(self.timestamp / 1000)} is deleted."
        )


def get_timestamp(data: bytes) -> int:
    """Interpret `data` as an unsigned integer.
    The data is expected to be 64 bits (8 bytes).

    Parameters
    ----------
    data : bytes
        The binary uint64 data.

    Returns
    -------
    int
        Timestamp **in milliseconds**.
        This is *not* the same as python built-in libraries' default timestamp (which is in seconds).
    """
    return int.from_bytes(data, byteorder="little", signed=False)


def get_image(data: bytes) -> Image.Image | None:
    """Interpret `data` as a png image.

    Parameters
    ----------
    data : bytes
        The binary PNG image data.

    Returns
    -------
    Image.Image
        A PIL Image object.

    Raises
    ------
    UnidentifiedImageError
        If the image is corrupted or not in PNG format.
    """
    frameIO = BytesIO(data)
    try:
        frame = Image.open(frameIO, formats=("PNG",))
    except UnidentifiedImageError:
        raise
    frameIO.close()
    return frame
