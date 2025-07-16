import logging

logger = logging.getLogger("HKSI WebRTC")
logger.setLevel(logging.DEBUG)
logger.propagate = False

__console_handler = logging.StreamHandler()
__console_handler.setLevel(logging.INFO)
logger.addHandler(__console_handler)

__file_handler = logging.FileHandler("webrtc.log")
__file_handler.setLevel(logging.WARNING)
logger.addHandler(__file_handler)


def set_console_log_level(level: int):
    __console_handler.setLevel(level)
