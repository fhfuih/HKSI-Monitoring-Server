"""
@author: I.O.

This file contain functions for HR and HRV. It is intended to be used 
in HKSI project, HKUST, 2023-
"""

import numpy as np
from pyampd.ampd import find_peaks
from scipy.signal import butter, sosfiltfilt
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d
import mediapipe as mp
import cv2


def POS(meanRGB, fps, step=10):
    """
    This function calculates rPPG.
    Input: meanRGB array(Nx3) which is mean RGB of skinFace pixels over time; N = # of frames; 3 = RGB channels
           fps = frame rate in units of frame/s

    Output: rPPG array (N,)
    """

    eps = 10**-9
    X = meanRGB.T
    c, f = X.shape
    w = int(1.6*fps)

    Q = np.array([[0, 1, -1], [-2, 1, 1]])

    # Initialize (1)
    final_signal = np.zeros(f)
    for window_end in np.arange(w, f, step):

        window_start = window_end - w + 1

        Cn = X[:, window_start:(window_end + 1)]
        M = 1.0 / (np.mean(Cn, axis=1)+eps)
        M = np.expand_dims(M, axis=1)
        Cn = M*Cn

        S = np.dot(Q, Cn)

        S1 = S[0, :]
        S2 = S[1, :]
        alpha = np.std(S1) / (eps + np.std(S2))

        pos_signal = S1 + alpha * S2
        pos_mean = pos_signal - np.mean(pos_signal)

        final_signal[window_start:(
            window_end + 1)] = final_signal[window_start:(window_end + 1)] + pos_mean

    rPPG = final_signal
    return rPPG



def DetectFace(frame, face_box = None, ROI_list=['face', 'forehead', 'left cheek', 'right cheek']):

    mp_face_mesh = mp.solutions.face_mesh

    forehead_pos = [66, 69, 299, 296]
    left_cheek_pos = [116, 187, 118, 216]
    right_cheek_pos = [347, 345, 436, 411]
    
    height, width, channels = frame.shape
    with mp_face_mesh.FaceMesh(static_image_mode=True,
                               max_num_faces=1, min_detection_confidence=0.01) as face_mesh:

        # Convert the BGR image to RGB before processing.
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Print and draw face mesh landmarks on the image.
        if not results.multi_face_landmarks:
            if face_box != None:
                face = frame[face_box[0]:face_box[1], face_box[2]:face_box[3], :]
            else:
                return 0
        else:
            face_box = []
            x_list = []
            y_list = []
            for face in results.multi_face_landmarks:
                for landmark in face.landmark:
                    x = landmark.x
                    y = landmark.y
    
                    relative_x = int(x * width)
                    relative_y = int(y * height)
    
                    x_list.append(relative_x)
                    y_list.append(relative_y)
    
            max_x = max(x_list)
            min_x = min(x_list)

    
            max_y = max(y_list)
            min_y = min(y_list)
            if min_y < 0:
                min_y = 0
    
            face = frame[min_y:max_y, min_x:max_x, :]
            face_box.append(min_y)
            face_box.append(max_y)
            face_box.append(min_x)
            face_box.append(max_x)
                                          



    result_list = []

    for ROI in ROI_list:
        if ROI == 'face':

            result_list.append(face)
        else:
            if ROI == 'forehead':
                ROI_pos = forehead_pos
            elif ROI == 'left cheek':
                ROI_pos = left_cheek_pos
            elif ROI == 'right cheek':
                ROI_pos = right_cheek_pos
            else:
                return None

            ROI_x = [x_list[i] - min_x for i in ROI_pos]
            ROI_y = [y_list[i] - min_y for i in ROI_pos]

            result = face[min(ROI_y): max(
                ROI_y), min(ROI_x): max(ROI_x), :]

            result_list.append(result)

    return result_list


