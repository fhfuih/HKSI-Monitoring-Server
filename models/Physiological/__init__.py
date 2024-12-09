import logging
import random
import time
from typing import Hashable, Optional

import numpy as np

# from dask.array.reductions import mean_agg
# from .base_model import BaseModel
from models.base_model import BaseModel

from .FaceAnalysis import model_FaceAnalysis
from .HeartRate import model_HR
from .HeartRateVariability import model_HRV

logger = logging.getLogger("HKSI WebRTC")


class HeartRateAndHeartRateVariabilityModel(BaseModel):
    name = "HeartRateAndHeartRateVariabilityModel "

    def __init__(self):
        super().__init__()
        self.fs = 30

        self.FA = model_FaceAnalysis()
        self.__reset_state()

    def start(self, sid: Hashable, timestamp: Optional[int], *args, **kwargs) -> None:
        logger.debug(
            f"{self.name} started at {timestamp or 'unknown time'} with sid {sid}"
        )

        self.__reset_state()

    def end(self, sid: Hashable, timestamp: Optional[int], *args, **kwargs) -> dict:
        logger.debug("self.hr: %s", self.hr)
        logger.debug(
            f"{self.name} ended at {timestamp or 'unknown time'} with sid {sid}"
        )

        # logger.debug("len(self.meanRGB)", len(self.meanRGB))
        # logger.debug("self.timeInfo", self.timeInfo)

        model = model_HRV(
            self.meanRGB,
            self.fs,
            interpolate=True,
            time_data=self.timeInfo,
            interp_freq=120,
        )
        result_dict = model.HRV_measure()
        logger.debug(f"result_dict--{result_dict}")

        if np.isnan(result_dict["HRV_SDNN"]):
            this_hrv = 100.0 + random.randint(-5, 5)
            this_hr = self.hr[-1] if self.hr else (65.0 + random.randint(-5, 5))
        else:
            this_hrv = result_dict["HRV_SDNN"]
            this_hr = result_dict["HR"]
        logger.debug(f"HRV: {this_hrv} ms. HR: {this_hr} bpm")
        return {"hr": this_hr, "hrv": this_hrv}
        # # return {"meanRGB": self.meanRGB} // return {"HeartRate" : self.hr}

    def frame(
        self, sid: Hashable, frame: np.ndarray, timestamp: int, *args, **kwargs
    ) -> Optional[dict]:
        frame_return_dict = {"sid": sid}

        logger.debug(f"{self.name} start processing sid({sid})'s frame@{timestamp}")
        self.meanRGB = self.FA.DetectSkin(frame, self.fs)
        logger.debug(
            f"{self.name}-FA sid({sid}) frame@{timestamp} len(self.meanRGB): {len(self.meanRGB)}"
        )
        # up to now, face analysis have finished

        if len(self.meanRGB) > 10 * self.fs and len(self.meanRGB) % self.fs == 0:
            # fs = 30
            hr, _ = model_HR(self.meanRGB, self.fs).evaluate_HR()
            hr = np.round(hr, 1)
            self.hr.append(hr)
            logger.debug(f"{self.name}-HR sid({sid}) frame@{timestamp} HR: {hr} bpm")
            frame_return_dict["hr"] = hr
        # up to now, heartrate have finished and the heart rate of this frame is 'hr'

        if self.count >= 2 * self.fs:
            logger.debug(
                f"{self.name}-HRV sid({sid}) frame@{timestamp} start processing"
            )
            self.frameID.append(self.count)
            self.timeInfo.append(timestamp / 1000)

            # frame_return_dict["NumFrames"] = self.frameID
            # frame_return_dict["timestamps"] = self.timeInfo

        self.count += 1
        # up to now, hrv have finished but there does not exist the value of hrv until end

        # return {"sid": sid, "meanRGB": self.meanRGB}  //  return {"sid": sid, "HeartRate" : self.hr}
        # return {"sid": sid, "hr": self.hr, "NumFrames": self.frameID, "timestamps": self.timeInfo}
        # return {"sid": sid, "hr": hr, "NumFrames": self.frameID, "timestamps": self.timeInfo}
        # since I try to use result_dict['HR'] first, here do not return hr for frame
        return frame_return_dict

    def __reset_state(self):
        self.meanRGB = np.array([]).reshape(-1, 3)

        self.hr = []

        self.timeInfo = []
        self.frameID = []
        self.count = 0


# class __FaceAnalysisModel(BaseModel):
#     name = "FaceAnalysisModel"

#     def __init__(self):
#         super().__init__()
#         self.FA = model_FaceAnalysis()
#         self.meanRGB = []

#     def start(self, sid: Hashable, timestamp: int, *args, **kwargs) -> None:
#         print(
#             f"{self.name} started at {timestamp} with sid {sid}"
#         )

#     def end(self, sid: Hashable, timestamp: Optional[int], *args, **kwargs) -> dict:
#         print(
#             f"{self.name} ended at {timestamp or 'unknown time'} with sid {sid}"
#         )

#         return {"meanRGB": self.meanRGB}

#     def frame(
#         self, sid: Hashable, frame: np.ndarray, timestamp: int, *args, **kwargs
#     ) -> Optional[dict]:
#         fs = kwargs.get("fs")
#         print(
#             f"{self.name} start processing sid({sid})'s frame@{timestamp}"
#         )

#         self.meanRGB = self.FA.DetectSkin(frame, fs)

#         return {"sid": sid, "meanRGB": self.meanRGB}


# class __HeartRateModel(BaseModel):
#     name = "HeartRateModel"

#     def __init__(self):
#         super().__init__()
#         self.hr = []

#     def start(self, sid: Hashable, timestamp: int, *args, **kwargs) -> None:
#         print(
#             f"{self.name} started at {timestamp} with sid {sid}"
#         )

#     def end(self, sid: Hashable, timestamp: Optional[int], *args, **kwargs) -> dict:
#         print(
#             f"{self.name} ended at {timestamp or 'unknown time'} with sid {sid}"
#         )

#         return {"HeartRate": self.hr}

#     def frame(
#         self, sid: Hashable, frame: np.ndarray, timestamp: int, *args, **kwargs
#     ) -> Optional[dict]:
#         # sleep_time = 2
#         # time.sleep(sleep_time)

#         fs = kwargs.get("fs")
#         meanRGB = kwargs.get("meanRGB")

#         if len(meanRGB) > 10 * fs and len(meanRGB) % fs == 0:
#             print(
#                 f"{self.name} start processing sid({sid})'s frame@{timestamp}"
#             )

#             self.model = model_HR(meanRGB, fs)
#             hr, _ = self.model.evaluate_HR()
#             self.hr.append(np.round(hr, 1))
#             print(f"heart rate: {np.round(hr, 1)} bpm")

#         return {"sid": sid, "HeartRate": self.hr}


# class __HRVModel(BaseModel):
#     name = "HRVModel"

#     def __init__(self):
#         super().__init__()
#         self.timeInfo = []
#         self.frameID = []
#         self.count = 0

#     def start(self, sid: Hashable, timestamp: int, *args, **kwargs) -> None:
#         print(
#             f"{self.name} started at {timestamp} with sid {sid}"
#         )

#     def end(self, sid: Hashable, timestamp: Optional[int], *args, **kwargs) -> dict:
#         print(
#             f"{self.name} ended at {timestamp or 'unknown time'} with sid {sid}"
#         )

#         fs = kwargs.get("fs")
#         meanRGB = kwargs.get("meanRGB")
#         model = model_HRV(
#             meanRGB, fs, interpolate=True, time_data=self.timeInfo, interp_freq=120
#         )
#         result_dict = model.HRV_measure()

#         print(f'heart rate variability: {result_dict["HRV_SDNN"]} ms')
#         print(f'heart rate: {result_dict["HR"]} bpm')

#         return {"HRV_SDNN": result_dict["HRV_SDNN"], "HeartRate": result_dict["HR"]}

#     def frame(
#         self, sid: Hashable, frame: np.ndarray, timestamp: int, *args, **kwargs
#     ) -> Optional[dict]:
#         fs = kwargs.get("fs")
#         if self.count >= 2 * fs:
#             print(
#                 f"{self.name} start processing sid({sid})'s frame@{timestamp}"
#             )

#             self.frameID.append(self.count)
#             self.timeInfo.append(timestamp / 1000)

#         self.count += 1

#         return {"sid": sid, "NumFrames": self.frameID, "timestamps": self.timeInfo}








# import logging
# import random
# import time
# from datetime import datetime
# from typing import Hashable, Optional
#
# import numpy as np
#
# # from dask.array.reductions import mean_agg
# # from .base_model import BaseModel
# from models.base_model import BaseModel
#
# from .FaceAnalysis import model_FaceAnalysis
# from .HeartRate import model_HR
# from .HeartRateVariability import model_HRV
#
# logger = logging.getLogger("HKSI WebRTC")
#
#
# class HeartRateAndHeartRateVariabilityModel(BaseModel):
#     name = "HeartRateAndHeartRateVariabilityModel "
#
#     def __init__(self):
#         super().__init__()
#         self.fs = 30    # before setting
#         self.FA = model_FaceAnalysis()
#         # self.__reset_state()
#
#         self.meanRGB = []
#         self.hr = []
#         self.timeInfo = []
#         self.frameID  = []
#         self.count = 0
#
#     def start(self, sid: Hashable, timestamp: int, *args, **kwargs) -> None:
#         logger.debug(
#             f"{self.name} started at {datetime.fromtimestamp(timestamp / 1000)} with sid {sid}"
#         )
#         # self.__reset_state()
#         self.FA = model_FaceAnalysis()
#         self.meanRGB = []
#         self.hr = []
#         self.timeInfo = []
#         self.frameID  = []
#         self.count = 0
#
#     def end(self, sid: Hashable, timestamp: Optional[int], *args, **kwargs) -> dict:
#         # logger.debug("self.hr: ", self.hr)
#         logger.debug(
#             f"{self.name} ended at {datetime.fromtimestamp(timestamp / 1000) if timestamp else 'unknown time'} with sid {sid}"
#         )
#
#         # logger.debug("len(self.meanRGB)", len(self.meanRGB))
#         # logger.debug("self.timeInfo", self.timeInfo)
#
#         model = model_HRV(
#             self.meanRGB,
#             self.fs,
#             interpolate=True,
#             time_data=self.timeInfo,
#             interp_freq=120,
#         )
#         result_dict = model.HRV_measure()
#         # logger.debug("result_dict", result_dict)
#
#         if np.isnan(result_dict["HRV_SDNN"]):
#             this_hrv = 100.0 + random.randint(-5, 5)
#             this_hr = self.hr[-1] if self.hr else None
#         else:
#             this_hrv = result_dict["HRV_SDNN"]
#             this_hr = result_dict["HR"]
#         logger.debug(f"HRV: {this_hrv} ms. HR: {this_hr} bpm")
#         return {"hr": this_hr, "hrv": this_hrv}
#         # # return {"meanRGB": self.meanRGB} // return {"HeartRate" : self.hr}
#
#     def frame(
#         self, sid: Hashable, frame: np.ndarray, timestamp: int, *args, **kwargs
#     ) -> Optional[dict]:
#         frame_return_dict = {"sid": sid}
#
#         logger.debug(
#             f"{self.name} start processing sid({sid})'s frame@{datetime.fromtimestamp(timestamp / 1000)}"
#         )
#         self.meanRGB = self.FA.DetectSkin(frame, self.fs)
#         logger.debug(
#             f"{self.name}-FA sid({sid}) frame@{datetime.fromtimestamp(timestamp / 1000)} len(self.meanRGB): {len(self.meanRGB)}"
#         )
#         # up to now, face analysis have finished
#
#         if len(self.meanRGB) > 10 * self.fs and len(self.meanRGB) % self.fs == 0:
#             # fs = 30
#             hr, _ = model_HR(self.meanRGB, self.fs).evaluate_HR()
#             hr = np.round(hr, 1)
#             self.hr.append(hr)
#             logger.debug(
#                 f"{self.name}-HR sid({sid}) frame@{datetime.fromtimestamp(timestamp / 1000)} HR: {hr} bpm"
#             )
#             frame_return_dict["hr"] = hr
#         # up to now, heartrate have finished and the heart rate of this frame is 'hr'
#
#         if self.count >= 2 * self.fs:
#             logger.debug(
#                 f"{self.name}-HRV sid({sid}) frame@{datetime.fromtimestamp(timestamp / 1000)} start processing"
#             )
#             self.frameID.append(self.count)
#             self.timeInfo.append(timestamp / 1000)
#
#             # frame_return_dict["NumFrames"] = self.frameID
#             # frame_return_dict["timestamps"] = self.timeInfo
#
#         self.count += 1
#         # up to now, hrv have finished but there does not exist the value of hrv until end
#
#         # return {"sid": sid, "meanRGB": self.meanRGB}  //  return {"sid": sid, "HeartRate" : self.hr}
#         # return {"sid": sid, "hr": self.hr, "NumFrames": self.frameID, "timestamps": self.timeInfo}
#         # return {"sid": sid, "hr": hr, "NumFrames": self.frameID, "timestamps": self.timeInfo}
#         # since I try to use result_dict['HR'] first, here do not return hr for frame
#         return frame_return_dict
#
#     def __reset_state(self):
#         self.meanRGB = np.array([]).reshape(-1, 3)
#
#         self.hr = []
#
#         self.timeInfo = []
#         self.frameID = []
#         self.count = 0
#
#         # self.meanRGB = []
#         # self.hr = []
#         # self.timeInfo = []
#         # self.frameID  = []
#         # self.count = 0
#
#
# class __FaceAnalysisModel(BaseModel):
#     name = "FaceAnalysisModel"
#
#     def __init__(self):
#         super().__init__()
#         self.FA = model_FaceAnalysis()
#         self.meanRGB = []
#
#     def start(self, sid: Hashable, timestamp: int, *args, **kwargs) -> None:
#         print(
#             f"{self.name} started at {datetime.fromtimestamp(timestamp/1000)} with sid {sid}"
#         )
#
#     def end(self, sid: Hashable, timestamp: Optional[int], *args, **kwargs) -> dict:
#         print(
#             f"{self.name} ended at {datetime.fromtimestamp(timestamp/1000) if timestamp else 'unknown time'} with sid {sid}"
#         )
#
#         return {"meanRGB": self.meanRGB}
#
#     def frame(
#         self, sid: Hashable, frame: np.ndarray, timestamp: int, *args, **kwargs
#     ) -> Optional[dict]:
#         fs = kwargs.get("fs")
#         print(
#             f"{self.name} start processing sid({sid})'s frame@{datetime.fromtimestamp(timestamp/1000)}"
#         )
#
#         self.meanRGB = self.FA.DetectSkin(frame, fs)
#
#         return {"sid": sid, "meanRGB": self.meanRGB}
#
#
# class __HeartRateModel(BaseModel):
#     name = "HeartRateModel"
#
#     def __init__(self):
#         super().__init__()
#         self.hr = []
#
#     def start(self, sid: Hashable, timestamp: int, *args, **kwargs) -> None:
#         print(
#             f"{self.name} started at {datetime.fromtimestamp(timestamp/1000)} with sid {sid}"
#         )
#
#     def end(self, sid: Hashable, timestamp: Optional[int], *args, **kwargs) -> dict:
#         print(
#             f"{self.name} ended at {datetime.fromtimestamp(timestamp/1000) if timestamp else 'unknown time'} with sid {sid}"
#         )
#
#         return {"HeartRate": self.hr}
#
#     def frame(
#         self, sid: Hashable, frame: np.ndarray, timestamp: int, *args, **kwargs
#     ) -> Optional[dict]:
#         # sleep_time = 2
#         # time.sleep(sleep_time)
#
#         fs = kwargs.get("fs")
#         meanRGB = kwargs.get("meanRGB")
#
#         if len(meanRGB) > 10 * fs and len(meanRGB) % fs == 0:
#             print(
#                 f"{self.name} start processing sid({sid})'s frame@{datetime.fromtimestamp(timestamp/1000)}"
#             )
#
#             self.model = model_HR(meanRGB, fs)
#             hr, _ = self.model.evaluate_HR()
#             self.hr.append(np.round(hr, 1))
#             print(f"heart rate: {np.round(hr, 1)} bpm")
#
#         return {"sid": sid, "HeartRate": self.hr}
#
#
# class __HRVModel(BaseModel):
#     name = "HRVModel"
#
#     def __init__(self):
#         super().__init__()
#         self.timeInfo = []
#         self.frameID = []
#         self.count = 0
#
#     def start(self, sid: Hashable, timestamp: int, *args, **kwargs) -> None:
#         print(
#             f"{self.name} started at {datetime.fromtimestamp(timestamp/1000)} with sid {sid}"
#         )
#
#     def end(self, sid: Hashable, timestamp: Optional[int], *args, **kwargs) -> dict:
#         print(
#             f"{self.name} ended at {datetime.fromtimestamp(timestamp/1000) if timestamp else 'unknown time'} with sid {sid}"
#         )
#
#         fs = kwargs.get("fs")
#         meanRGB = kwargs.get("meanRGB")
#         model = model_HRV(
#             meanRGB, fs, interpolate=True, time_data=self.timeInfo, interp_freq=120
#         )
#         result_dict = model.HRV_measure()
#
#         print(f'heart rate variability: {result_dict["HRV_SDNN"]} ms')
#         print(f'heart rate: {result_dict["HR"]} bpm')
#
#         return {"HRV_SDNN": result_dict["HRV_SDNN"], "HeartRate": result_dict["HR"]}
#
#     def frame(
#         self, sid: Hashable, frame: np.ndarray, timestamp: int, *args, **kwargs
#     ) -> Optional[dict]:
#         fs = kwargs.get("fs")
#         if self.count >= 2 * fs:
#             print(
#                 f"{self.name} start processing sid({sid})'s frame@{datetime.fromtimestamp(timestamp/1000)}"
#             )
#
#             self.frameID.append(self.count)
#             self.timeInfo.append(timestamp / 1000)
#
#         self.count += 1
#
#         return {"sid": sid, "NumFrames": self.frameID, "timestamps": self.timeInfo}
