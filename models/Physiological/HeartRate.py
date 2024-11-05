"""

@author: I.O.
"""
from HKSI_functions import POS
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.fft import fft, fftfreq



class model_HR:
    def __init__(self, meanRGB, fs):
        self.meanRGB = np.array(meanRGB)
        self.fs = fs
        if len(self.meanRGB) == 0:
            self.rPPG_cand = meanRGB
        else:
            self.rPPG_cand = POS(self.meanRGB, self.fs)
    
    def evaluate_HR(self, order = 5, min_freq=0.6, max_freq=4, apply_hanning=False):
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
        

        signal = self.rPPG_cand
        fs = self.fs
        fft_details = {}
        HeartRate = 0
        
        if len(signal) > 10 * fs:
            
            sos = butter(order, [min_freq, max_freq],
                         btype='bandpass', output='sos', fs=fs)
            rPPG = sosfiltfilt(sos, np.double(signal))
    
    
            if apply_hanning:
                hanning = np.hanning(len(rPPG))
                rPPG = hanning * rPPG
    
            
            N = len(rPPG)
            T = 1 / fs
    
            yf = fft(rPPG)
    
            freqs = fftfreq(N, T)[:N//2]  # get real valuesof freqs
    
            yf = 2.0/N * np.abs(yf[0:N//2])  # get real parts of amplitudes
    
            max_idx = np.where(yf == np.amax(yf))
            highest_freq = float(freqs[max_idx])
            HeartRate = highest_freq*60
    
            fft_details['frequencies'] = freqs
            fft_details['amplitudes'] = yf
            fft_details['rPPG'] = rPPG
    
        
        return HeartRate, fft_details
        
    
    
    
    