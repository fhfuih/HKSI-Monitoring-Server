import time
from datetime import datetime
from typing import Hashable, Optional
import numpy as np

#from .base_model import BaseModel
from models.base_model import BaseModel
from .FaceAnalysis import model_FaceAnalysis
from .HeartRate import model_HR
from .HeartRateVariability import model_HRV


class FaceAnalysisModel(BaseModel):
    name = 'FaceAnalysisModel'
    
    def __init__(self):
        super().__init__()
        self.FA = model_FaceAnalysis()
        self.meanRGB = []
        
    def start(self, sid: Hashable, timestamp: int, *args, **kwargs) -> None:
        print(
            f"{self.name} started at {datetime.fromtimestamp(timestamp/1000)} with sid {sid}"
        )
        
    
    def end(self, sid: Hashable, timestamp: Optional[int], *args, **kwargs) -> dict:
        print(
            f"{self.name} ended at {datetime.fromtimestamp(timestamp/1000) if timestamp else 'unknown time'} with sid {sid}"
        )
        
        return {"meanRGB" : self.meanRGB}
        
        
    def frame(
        self, sid: Hashable, frame: np.ndarray, timestamp: int, *args, **kwargs
    ) -> Optional[dict]:
        

        fs = kwargs.get('fs')
        print(
            f"{self.name} start processing sid({sid})'s frame@{datetime.fromtimestamp(timestamp/1000)}"
        )
        
        self.meanRGB = self.FA.DetectSkin(frame, fs)
        
        return {"sid": sid, "meanRGB" : self.meanRGB}
        

class HeartRateModel(BaseModel):
    name = 'HeartRateModel'
    
    def __init__(self):
        super().__init__()
        self.hr = []
        
        
    def start(self, sid: Hashable, timestamp: int, *args, **kwargs) -> None:
        print(
            f"{self.name} started at {datetime.fromtimestamp(timestamp/1000)} with sid {sid}"
        )
        
    
    def end(self, sid: Hashable, timestamp: Optional[int], *args, **kwargs) -> dict:
        print(
            f"{self.name} ended at {datetime.fromtimestamp(timestamp/1000) if timestamp else 'unknown time'} with sid {sid}"
        )
        
        return {"HeartRate" : self.hr}
        
        
    def frame(
        self, sid: Hashable, frame: np.ndarray, timestamp: int, *args, **kwargs
    ) -> Optional[dict]:
        
        #sleep_time = 2
        #time.sleep(sleep_time)
        

        fs = kwargs.get('fs')
        meanRGB = kwargs.get('meanRGB')
        
        if len(meanRGB) > 10 * fs and len(meanRGB) % fs == 0:
            
            print(
                f"{self.name} start processing sid({sid})'s frame@{datetime.fromtimestamp(timestamp/1000)}"
            )
            
            self.model = model_HR(meanRGB, fs)
            hr, _ = self.model.evaluate_HR()
            self.hr.append(np.round(hr, 1))
            print(f'heart rate: {np.round(hr, 1)} bpm')
                
        return {"sid": sid, "HeartRate" : self.hr}

        
class HRVModel(BaseModel):
    name = 'HRVModel'
    
    def __init__(self):
        super().__init__()
        self.timeInfo = []
        self.frameID  = []
        self.count = 0
        
        
    def start(self, sid: Hashable, timestamp: int, *args, **kwargs) -> None:
        print(
            f"{self.name} started at {datetime.fromtimestamp(timestamp/1000)} with sid {sid}"
        )
        
    
    def end(self, sid: Hashable, timestamp: Optional[int], *args, **kwargs) -> dict:
        print(
            f"{self.name} ended at {datetime.fromtimestamp(timestamp/1000) if timestamp else 'unknown time'} with sid {sid}"
        )
        
        fs = kwargs.get('fs')
        meanRGB = kwargs.get('meanRGB')
        model = model_HRV(meanRGB, fs, interpolate=True, time_data=self.timeInfo, interp_freq=120)
        result_dict = model.HRV_measure()
        
        print(f'heart rate variability: {result_dict["HRV_SDNN"]} ms')
        print(f'heart rate: {result_dict["HR"]} bpm')
        
        return {"HRV_SDNN": result_dict['HRV_SDNN'], 
                "HeartRate": result_dict['HR']}
        
        
    def frame(
        self, sid: Hashable, frame: np.ndarray, timestamp: int, *args, **kwargs
    ) -> Optional[dict]:
        
        
        fs = kwargs.get('fs')
        if self.count >= 2 * fs:
            
            print(
                f"{self.name} start processing sid({sid})'s frame@{datetime.fromtimestamp(timestamp/1000)}"
            )
            
            self.frameID.append(self.count)
            self.timeInfo.append(timestamp/1000)
            
        self.count += 1
                
        return {"sid": sid, "NumFrames": self.frameID, "timestamps": self.timeInfo}
