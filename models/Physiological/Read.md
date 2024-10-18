This folder is for physological measurement research, i.e. for HR and HRV related analysis. Please refer to Mr. Ismoild Odinaev prior to mmaking any changes. The example below shows how to use the code:

from FaceAnalysis import model_FaceAnalysis
from HeartRate import model_HR
from HeartRateVariability import model_HRV



### trasnform each frame into time-series data point
FA = model_FaceAnalysis() # this only have to be instantiated once

for each frame:
  FA.DetectSkin(frame, fs) ### analyze each frame; inputs are frame, and framerate (frames per second)

### evaluate Heart rate at least after 10 seconds of data
meanRGB = FA.meanRGB
model1 = model_HR(meanRGB, fs) #### inputs are meanRGB and framerate (fs)
hr, _ = model1.evaluate_HR() ### hr is heart rate


#### evaluate HRV after scanning is complete
meanRGB = FA.meanRGB
model2 = model_HRV(meanRGB, fs, interpolate=True, time_data=timeInfo, interp_freq=120) ### inputs are meanRGB, framerate (fs) and timestamps (timeInfo)
dictionary = model2.HRV_measure()
dictionary["HR"] ### final HR value
dictionary["HRV_SDNN"] ### final HRV value

