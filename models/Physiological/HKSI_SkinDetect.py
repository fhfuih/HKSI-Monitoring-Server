import numpy as np
from scipy import signal
import cv2
#from HDi import *

### FROM HDi.py file ####

import scipy.stats.kde as kde

def calc_min_interval(x, alpha):
    """Internal method to determine the minimum interval of a given width
    Assumes that x is sorted numpy array.
    """

    n = len(x)
    cred_mass = 1.0-alpha

    interval_idx_inc = int(np.floor(cred_mass*n))
    n_intervals = n - interval_idx_inc
    interval_width = x[interval_idx_inc:] - x[:n_intervals]

    if len(interval_width) == 0:
        raise ValueError('Too few elements for interval calculation')

    min_idx = np.argmin(interval_width)
    hdi_min = x[min_idx]
    hdi_max = x[min_idx+interval_idx_inc]
    return hdi_min, hdi_max

def hdi(x, alpha=0.05):
    """Calculate highest posterior density (HPD) of array for given alpha. 
    The HPD is the minimum width Bayesian credible interval (BCI).
    :Arguments:
        x : Numpy array
        An array containing MCMC samples
        alpha : float
        Desired probability of type I error (defaults to 0.05)
    """

    # Make a copy of trace
    x = x.copy()
    # For multivariate node
    if x.ndim > 1:
        # Transpose first, then sort
        tx = np.transpose(x, list(range(x.ndim))[1:]+[0])
        dims = np.shape(tx)
        # Container list for intervals
        intervals = np.resize(0.0, dims[:-1]+(2,))

        for index in make_indices(dims[:-1]):
            try:
                index = tuple(index)
            except TypeError:
                pass

            # Sort trace
            sx = np.sort(tx[index])
            # Append to list
            intervals[index] = calc_min_interval(sx, alpha)
        # Transpose back before returning
        return np.array(intervals)
    else:
        # Sort univariate node
        sx = np.sort(x)
        return np.array(calc_min_interval(sx, alpha))


def hdi2(sample, alpha=0.05, roundto=2):
    """Calculate highest posterior density (HPD) of array for given alpha. 
    The HPD is the minimum width Bayesian credible interval (BCI). 
    The function works for multimodal distributions, returning more than one mode
    Parameters
    ----------
    
    sample : Numpy array or python list
        An array containing MCMC samples
    alpha : float
        Desired probability of type I error (defaults to 0.05)
    roundto: integer
        Number of digits after the decimal point for the results
    Returns
    ----------
    hpd: array with the lower 
          
    """
    sample = np.asarray(sample)
    sample = sample[~np.isnan(sample)]
    # get upper and lower bounds
    l = np.min(sample)
    u = np.max(sample)
    density = kde.gaussian_kde(sample)
    x = np.linspace(l, u, 2000)
    y = density.evaluate(x)
    #y = density.evaluate(x, l, u) waitting for PR to be accepted
    xy_zipped = zip(x, y/np.sum(y))
    xy = sorted(xy_zipped, key=lambda x: x[1], reverse=True)
    xy_cum_sum = 0
    hdv = []

    for val in xy:
        xy_cum_sum += val[1]
        hdv.append(val[0])
        if xy_cum_sum >= (1-alpha):
            break
    hdv.sort()
    diff = (u-l)/20  # differences of 5%
    hpd = []
    hpd.append(round(min(hdv), roundto))

    for i in range(1, len(hdv)):
        if hdv[i]-hdv[i-1] >= diff:
            hpd.append(round(hdv[i-1], roundto))
            hpd.append(round(hdv[i], roundto))
    hpd.append(round(max(hdv), roundto))
    ite = iter(hpd)
    hpd = list(zip(ite, ite))
    modes = []
    for value in hpd:
         x_hpd = x[(x > value[0]) & (x < value[1])]
         y_hpd = y[(x > value[0]) & (x < value[1])]

         modes.append(round(x_hpd[np.argmax(y_hpd)], roundto))
    return hpd, x, y, modes
### FROM HDi.py file ####

class SkinDetect():

    def __init__(self, strength=0.2):
        self.description = 'Skin Detection Module'
        self.strength = strength
        self.stats_computed = False

    def compute_stats(self, face):

        assert (self.strength > 0 and self.strength < 1), "'strength' parameter must have values in [0,1]"

        faceColor = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        h = faceColor[:,:,0].reshape(-1,1)
        s = faceColor[:,:,1].reshape(-1,1)
        v = faceColor[:,:,2].reshape(-1,1)
       
        alpha = self.strength    #the highest, the stronger the masking  

        hpd_h, x_h, y_h, modes_h = hdi2(np.squeeze(h), alpha=alpha)
        min_s, max_s = hdi(np.squeeze(s), alpha=alpha)
        min_v, max_v = hdi(np.squeeze(v), alpha=alpha)

        if len(hpd_h) > 1:

            self.multiple_modes = True

            if len(hpd_h) > 2:
                print('WARNING!! Found more than 2 HDIs in Hue Channel empirical Distribution... Considering only 2')
                from scipy.spatial.distance import pdist, squareform
                m = np.array(modes_h).reshape(-1,1)
                d = squareform(pdist(m))
                maxij = np.where(d==d.max())[0]
                i = maxij[0]
                j = maxij[1]
            else:
                i = 0
                j = 1

            min_h1 = hpd_h[i][0]
            max_h1 = hpd_h[i][1]
            min_h2 = hpd_h[j][0]
            max_h2 = hpd_h[j][1]
            
            self.lower1 = np.array([min_h1, min_s, min_v], dtype = "uint8")
            self.upper1 = np.array([max_h1, max_s, max_v], dtype = "uint8")
            self.lower2 = np.array([min_h2, min_s, min_v], dtype = "uint8")
            self.upper2 = np.array([max_h2, max_s, max_v], dtype = "uint8")
            

        elif len(hpd_h) == 1:

            self.multiple_modes = False
            
            min_h = hpd_h[0][0]
            max_h = hpd_h[0][1]
            
            self.lower = np.array([min_h, min_s, min_v], dtype = "uint8")
            self.upper = np.array([max_h, max_s, max_v], dtype = "uint8")
            


        self.stats_computed = True


    def get_skin(self, face, filt_kern_size=7, verbose=False, plot=False):

        if not self.stats_computed:
            raise ValueError("ERROR! You must compute stats at least one time")

        faceColor = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)

        if self.multiple_modes:
            if verbose:
                print('\nLower1: ' + str(self.lower1))
                print('Upper1: ' + str(self.upper1))
                print('\nLower2: ' + str(self.lower2))
                print('Upper2: ' + str(self.upper2) + '\n')

            skinMask1 = cv2.inRange(faceColor, self.lower1, self.upper1)
            skinMask2 = cv2.inRange(faceColor, self.lower2, self.upper2)
            skinMask = np.logical_or(skinMask1, skinMask2).astype(np.uint8)*255
            THreshold = np.zeros((4,3))
            THreshold[0,] = self.lower1
            THreshold[1,] = self.upper1
            THreshold[2,] = self.lower2
            THreshold[3,] = self.upper2
        else:

            if verbose:
                print('\nLower: ' + str(self.lower))
                print('Upper: ' + str(self.upper) + '\n')
            
            THreshold = np.zeros((2,3))
            THreshold[0,] = self.lower
            THreshold[1,] = self.upper
            skinMask = cv2.inRange(faceColor, self.lower, self.upper)

        if filt_kern_size > 0:
            skinMask = signal.medfilt2d(skinMask, kernel_size=filt_kern_size)
        skinFace = cv2.bitwise_and(face, face, mask=skinMask)

        if plot:
            
            h = faceColor[:,:,0].reshape(-1,1)
            s = faceColor[:,:,1].reshape(-1,1)
            v = faceColor[:,:,2].reshape(-1,1)

            import matplotlib.pyplot as plt
            plt.figure()              
            plt.subplot(2,2,1)               
            plt.hist(h, 20)
            plt.title('Hue')
            plt.subplot(2,2,2)
            plt.hist(s, 20)
            plt.title('Saturation')
            plt.subplot(2,2,3)
            plt.hist(v, 20)
            plt.title('Value')
            plt.subplot(2,2,4)
            plt.imshow(skinFace)
            plt.title('Masked Face')
            plt.show()

        return (skinFace, THreshold)
