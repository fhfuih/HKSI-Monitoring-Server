"""
@author: I.O.

This example file to run algorithms to obtain HR and HRV
HKUST 2023-
"""
import cv2
import numpy as np
from HKSI_physiological_functions import POS, band_pass_filter, Signal_Postprocessing, Face_Algorithm, rgb_mean, HRV
from HKSI_SkinDetect import SkinDetect
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


video_path = "D:/Data_Ismoil/Experimental_data/P1_S1.avi"


############### Part 1: Video reading

cap = cv2.VideoCapture(video_path)
fs = cap.get(cv2.CAP_PROP_FPS)
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
meanRGB = np.zeros([num_frames-90, 3])


sd = SkinDetect(strength=0.2)
count = 0
while True:
    # Capture the next frame
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        print("End of video reached.")
        break


    #cv2.imshow('Video Frame', frame)
    
    if count >= 90:
        _, skinFace = Face_Algorithm(frame, sd, count-90)
        b, g, r = rgb_mean(skinFace)
        meanRGB[count-90, 0], meanRGB[count-90, 1], meanRGB[count-90, 2] = r,g,b 
    
    count += 1
    # Press 'q' to quit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()



############ Part 2: Analysis

timestamp_path = "D:/Data_Ismoil/Experimental_data/P1_S1_frames.csv"
times = np.array(pd.read_csv(timestamp_path))
timeInfo = np.zeros(len(times))
for i in range(len(times)):
    times[i, 1] = datetime.strptime(times[i, 1], "%d/%m/%Y, %H:%M:%S.%f")
    if i>0:
        timeInfo[i] = (times[i, 1] - times[0, 1]).total_seconds()

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
file_path = "D:/Data_Ismoil/Experimental_data/P1_S1.avi.npz"
file = np.load(file_path, allow_pickle=True)
dictionary = file['arr_0'].item()
meanRGB2 = dictionary['meanRGB']


rPPG_cand = POS(meanRGB2, fs)
rPPG, det = band_pass_filter(rPPG_cand, 5, fs = fs, need_details=True)

SignalAnalysis = Signal_Postprocessing(rPPG_cand, fs, interpolate=True, interp_freq=120, time_data = timeInfo[60:])
rppg2, ibi = SignalAnalysis.HRV_measure(band=0.2)
hrv = HRV(ibi)
SDNN = hrv.sdnn()
