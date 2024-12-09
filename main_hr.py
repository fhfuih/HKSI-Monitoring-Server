"""
@author: I.O.

This example file to run algorithms to obtain HR and HRV
HKUST 2023-
"""

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from models.Physiological.FaceAnalysis import model_FaceAnalysis
from models.Physiological.HeartRate import model_HR
from models.Physiological.HeartRateVariability import model_HRV

video_path = Path(__file__, "..", "sample-video.mp4").resolve()


############### Part 1: Video reading

cap = cv2.VideoCapture(str(video_path))
fs = cap.get(cv2.CAP_PROP_FPS)
count = 0
FA = model_FaceAnalysis()  # initialize time-series data
timeInfo = []

while True:
    ret, frame = cap.read()
    timeInfo.append(1 / fs * count)

    if not ret:
        print("End of video reached.")
        break

    # cv2.imshow('Video Frame', frame)

    # generate time-series data
    meanRGB = FA.DetectSkin(frame, fs)

    # calculate instantaneous heart rate
    ### Note heart rate can be computed after 10 sec of video + 2 sec for stabilization
    ### evaluting heart rate each second is sufficient. No need for every frame
    if len(meanRGB) > 12 * fs and len(meanRGB) % fs == 0:
        model1 = model_HR(meanRGB, fs)
        hr, _ = model1.evaluate_HR()
        print(f"heart rate: {np.round(hr,1)} bpm")

    count += 1
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


############ Part 2: Analysis

############ Evaluate Heart Rate and Heart Rate Variability AFTER video recording finished
model2 = model_HRV(meanRGB, fs, interpolate=True, time_data=timeInfo, interp_freq=120)
hrv_dict = model2.HRV_measure()
print(f'heart rate variability: {hrv_dict["HRV_SDNN"]} ms')
print(f'heart rate: {hrv_dict["HR"]} bpm')


"""
########## import ground truth data

gt_path = "D:/Data_Ismoil/Experimental_data/GT/P1_S1.txt"
HR_gt = []
ibi_gt = []
i = 0
with open(gt_path, 'r') as text_file:
    for line in text_file:
        if i > 6:
            ibi_gt.append(np.float64(line.split(" ")[1]))
            HR_gt.append(np.float64(line.split(" ")[0]))
    
        i += 1
"""
