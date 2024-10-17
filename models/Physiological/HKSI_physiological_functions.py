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


def rgb_mean(skinFace):
    ''' This function calculates mean of the each channel for the 
    given image
    '''
    # Count how many colored pixel
    non_black_pixels_mask = np.any(skinFace != [0, 0, 0], axis=-1)
    num_pixel = np.count_nonzero(non_black_pixels_mask)

    # Count the sum for all pixels
    RGB_sum = np.sum(skinFace, axis=(0, 1))

    return RGB_sum/num_pixel


def band_pass_filter(signal, order, fs, min_freq=0.6, max_freq=4.5, apply_hanning=False, need_details=False):
    '''This function applies bandpass filter and returns filtered signal. Reccommended values are useful for Heart Rate.



    Input: 
        signal (N,) = (array) signal of size N
        order = (int) order of butterworth signal
        fs = (int) framerate
        min_freq = (float) min frequnecy range for HR in units of Hz
        max_freq = (float) max frequnecy range for HR in units of Hz
        apply_hanning = (bool) whether to apply hanning windowing
        need_details = (bool) whether to return details from FFT 

    Output:
        y (N,) = (array) bandpass filtered signal
        fft_details = (dictionary) FFT details such as FFT freqs, amplitudes and maximum freq in FFT range (only returned if need_details = True)

    '''

    # Step 1:  Bandpass filter
    sos = butter(order, [min_freq, max_freq],
                 btype='bandpass', output='sos', fs=fs)
    result = sosfiltfilt(sos, np.double(signal))
    y = result

    if apply_hanning:
        hanning = np.hanning(len(result))
        y = hanning * result

    fft_details = {}
    if need_details:

        N = len(signal)
        T = 1 / fs

        yf = fft(y)

        freqs = fftfreq(N, T)[:N//2]  # get real valuesof freqs

        yf = 2.0/N * np.abs(yf[0:N//2])  # get real parts of amplitudes

        max_idx = np.where(yf == np.amax(yf))
        highest_freq = float(freqs[max_idx])

        fft_details['frequencies'] = freqs
        fft_details['amplitudes'] = yf
        fft_details['max_freq'] = highest_freq

        return y, fft_details

    return y

def Face_Algorithm(frame, sd, count, face_box = None, ROI_list=['face', 'forehead', 'left cheek', 'right cheek']):

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
                                          

    if count == 0:
        sd.compute_stats(face)

    skinFace, Threshold = sd.get_skin(
        face, filt_kern_size=0, verbose=False, plot=False)

    result_list = []

    for ROI in ROI_list:
        if ROI == 'face':

            result_list.append(skinFace)
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

            result = skinFace[min(ROI_y): max(
                ROI_y), min(ROI_x): max(ROI_x), :]

            result_list.append(result)

    return result_list, skinFace

class Signal_Postprocessing:
    def __init__(self, rPPG, fs, interpolate = False, time_data = None, interp_freq = None):
        '''
        This class applies post-processing steps on rPPG signal to get 
        refined IBIs and cleaned rPPG signal.
        
        Input:
            rPPG (N,) = (array) rPPG signal of size N
            fs = (int) framerate of the video/signal
            interpolate = (bool) whether to do interpolation
            time_data (N,) = (array) time data of size N for interpolation. Time data should start from time 0sec and 
                            continue in units of seconds. It is None when interpolate = False
            interp_freq = (int) interpolation frequency. None if interpolate = False
            
        '''
        if interpolate:  
            interp_function = interp1d(time_data, rPPG, 'cubic')
            x_new = np.arange(time_data[0], time_data[-1], 1/interp_freq)
            rPPG = interp_function(x_new)
            fs = interp_freq
        
        self.fs = fs
        #self.rPPG = rPPG
        self.rPPG = self.bandpass_filter(rPPG, 5, 0.6, 4)
    
    def bandpass_filter(self, signal, order, min_freq=0.8, max_freq=4.5, apply_hanning=False, need_details=False):
        '''

        Input: 
            signal (N,) = (array) signal of size N
            order = (int) order of butterworth signal
            min_freq = (float) min frequnecy range for HR in units of Hz
            max_freq = (float) max frequnecy range for HR in units of Hz
            apply_hanning = (bool) whether to apply hanning windowing
            need_details = (bool) whether to return details from FFT 

        Output:
            y (N,) = (array) bandpass filtered signal
            fft_details = (dictionary) FFT details such as FFT freqs, amplitudes and maximum freq in FFT range (only returned if need_details = True)

        '''
        # Step 1:  Bandpass filter
        fs = self.fs
        
        sos = butter(order, [min_freq, max_freq],
                     btype='bandpass', output='sos', fs=fs)
        result = sosfiltfilt(sos, np.double(signal))
        y = result

        if apply_hanning:
            hanning = np.hanning(len(result))
            y = hanning * result

        fft_details = {}
        if need_details:

            N = len(signal)
            T = 1 / fs

            yf = fft(y)

            freqs = fftfreq(N, T)[:N//2]  # get real valuesof freqs

            yf = 2.0/N * np.abs(yf[0:N//2])  # get real parts of amplitudes

            max_idx = np.where(yf == np.amax(yf))
            highest_freq = float(freqs[max_idx])

            fft_details['frequencies'] = freqs
            fft_details['amplitudes'] = yf
            fft_details['max_freq'] = highest_freq

            return y, fft_details

        return y
    
    def peak_analysis(self, signal, window_size=10):
        '''
        This function calculates instantenious IBIs from given rPPG/PPG/ECG signal.

        Input:
            signal (N,) = (array) signal of size N
            window_size = (float) size of the windowing in units of number of peaks
  
        Output:
            final_ibi (M,) = (list) IBIs in units of ms
        '''
        
        fs = self.fs
        pyamd_win = int(100/60*fs)
        peak_locations = find_peaks(signal, pyamd_win)
        ibi = np.diff(peak_locations)*1000/fs
        # reject physically impossible IBIs
        ibi = ibi[ibi < 1750]
        ibi = ibi[ibi > 330]

        # reject impossible jumps in IBIs
        mean_ibi = np.mean(ibi)
        min_val = mean_ibi*0.4
        max_val = mean_ibi*1.6

        L = len(ibi)
        final_ibi = []
        for w in np.arange(0, L, window_size):
            if w + window_size >= len(ibi):
                ibi_window = ibi[w:]

            else:
                ibi_window = ibi[w: w + window_size]

            
            ibi_window = ibi_window[ibi_window > min_val]
            ibi_window = ibi_window[ibi_window < max_val]
            mean_ibi_window = np.mean(ibi_window)

            min_val_window = mean_ibi_window*0.8
            max_val_window = mean_ibi_window*1.2
            
            ibi_window = ibi_window[ibi_window > min_val_window]
            ibi_window = ibi_window[ibi_window < max_val_window]

            final_ibi = final_ibi + ibi_window.tolist()

        return final_ibi
    
    def inter_cleaning(self, signal, band):
        ''' 
        Input:
            signal (N,) = (array) signal of size N
            band  = (float) freq band for butterworth bandpass filtering

        Output:
            finalPPG (N,) = (array) cleaned signal
        '''

    
        ppg1, mydict = self.bandpass_filter(signal, 5, 0.6, 3, apply_hanning=True, need_details=True)
        highest_freq = mydict['max_freq']
 

        mean_ibi = 1/highest_freq
        lower_band = 1/(mean_ibi*(1.1 + band))
        upper_band = 1/(mean_ibi*(1 - band))

        if lower_band < 0.6:
            ppg3 = self.bandpass_filter(ppg1, 4, 0.6, upper_band)
        elif upper_band > 3:
            ppg3 = self.bandpass_filter(ppg1, 4, lower_band, 3)
        else:
            ppg3 = self.bandpass_filter(ppg1, 4, lower_band, upper_band)
        

        finalPPG = ppg3
        return finalPPG
    

    def HRV_measure(self, window_size=14.5, band=0.2, step=2.82):
        '''
        Input:
            window_size = (float) size of the windowing in units of seconds
            band  = (float) freq band for butterworth bandpass filtering in units of Hz
            step = (float) stepsize of the windowing in units of sec           


        Output:
            finalPPG (N,) = (array) cleaned signal
            ibi(m, ) = (array) refined IBIs
        '''
        
        
        fs = self.fs
        signal = self.rPPG
            
        N = len(signal)
        size = int(fs*window_size)
        step = int(fs*step) 
        win_ends = np.arange(size, N, step)
        L = len(win_ends)
        
        final_signal = np.zeros(N)
        coef = np.ones(L)

        for i in range(L+1):
            if i == L:
                win_end = len(signal)
            else:
                win_end = win_ends[i]

            running_window = win_end - size
            
                
            result = self.inter_cleaning(signal[running_window : win_end+1], band)
            cleaned_signal = result - np.mean(result)
            
            final_signal[running_window : win_end
                         +1] = final_signal[running_window : win_end+1] + cleaned_signal

        if L >= 2*int(size/step):
            for j in range(int(size/step)):
                coef[j] = (int(size/step) + 1) / (j + 1)
                coef[-1*j - 1] = (int(size/step) + 1) / (j + 1)
            
                if j == 0:
                    final_signal[j*step:(j+1)*step] *= coef[j]*2
                    final_signal[(-1)*step:] *= coef[j]*2
                else:
                    final_signal[j*step:(j+1)*step] *= coef[j]*1.5
                    final_signal[(-1*j-1)*step:(-1*j)*step] *= coef[j]*1.5
        else:
            pass
        
        ibi = self.peak_analysis(final_signal, window_size = 10)
        finalPPG = final_signal
        return finalPPG, np.array(ibi)
        

class HRV:
    def __init__(self, ibi):

        ibi = np.array(ibi)
        self.RR = ibi
        self.SD = np.diff(self.RR)
        
    def sdnn(self):
        return np.std(self.RR)

    def rmssd(self):
        rmssd = np.sqrt(np.mean(self.SD**2))
        return rmssd
       
