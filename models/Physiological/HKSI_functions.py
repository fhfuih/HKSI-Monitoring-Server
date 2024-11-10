"""
@author: I.O.

This file contain functions for HR and HRV. It is intended to be used 
in HKSI project, HKUST, 2023-
"""

import numpy as np

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

