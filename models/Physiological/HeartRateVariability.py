"""
@author: I.O.
"""

import numpy as np
from pyampd.ampd import find_peaks
from scipy.signal import butter, sosfiltfilt
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d
from .HKSI_functions import POS


class model_HRV:
    def __init__(self, meanRGB, fs, interpolate = False, time_data = None, interp_freq = None):
        '''
        This class applies post-processing steps on rPPG signal to get 
        refined IBIs and cleaned rPPG signal.
        
        Input:
            rPPG (N,3) = (array) meanRGB signal of size (N,3)
            fs = (int) framerate of the video/signal
            interpolate = (bool) whether to do interpolation
            time_data (N,) = (array) time data of size N for interpolation. Time data should start from time 0sec and 
                            continue in units of seconds. It is None when interpolate = False
            interp_freq = (int) interpolation frequency. None if interpolate = False
            
        '''
        
        self.meanRGB = np.array(meanRGB)
        rPPG_cand = POS(self.meanRGB, fs)
        fs = int(fs)
        
        if interpolate:  
            interp_function = interp1d(time_data, rPPG_cand, 'cubic') #discard 2 sec of initial signal
            x_new = np.arange(time_data[0], time_data[-1], 1/interp_freq) 
            rPPG_cand = interp_function(x_new)
            fs = interp_freq
        
        self.fs = fs
        self.rPPG = self.bandpass_filter(rPPG_cand, 5, 0.75, 4)
    
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
        ibi = ibi[ibi < 1450]
        ibi = ibi[ibi > 350]

        # reject impossible jumps in IBIs
        mean_ibi = np.median(ibi)
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

            min_val_window = mean_ibi_window*0.75
            max_val_window = mean_ibi_window*1.25
            
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

    
        ppg1, mydict = self.bandpass_filter(signal, 5, 0.75, 3, apply_hanning=True, need_details=True)
        highest_freq = mydict['max_freq']
 

        mean_ibi = 1/highest_freq
        lower_band = 1/(mean_ibi*(1.1 + band))
        upper_band = 1/(mean_ibi*(1 - band))

        if lower_band < 0.75:
            ppg3 = self.bandpass_filter(ppg1, 4, 0.75, upper_band)
        elif upper_band > 3:
            ppg3 = self.bandpass_filter(ppg1, 4, lower_band, 3)
        else:
            ppg3 = self.bandpass_filter(ppg1, 4, lower_band, upper_band)
        

        finalPPG = ppg3
        return finalPPG
    

    def HRV_measure(self, window_size=10, band=0.2, step=2.82):
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
        
        self.RR = np.array(ibi)        
        HRV_SDNN = np.std(self.RR)

        
        result_dictionary = {}
        result_dictionary['HRV_SDNN'] = np.round(HRV_SDNN,1)
        result_dictionary['HR'] = np.round(1000/np.mean(self.RR)*60,1)
        result_dictionary['rPPG'] = final_signal
        
        return result_dictionary
        


       

