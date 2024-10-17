"""
@author: I.O.
"""
from HKSI_functions import DetectFace
from HKSI_SkinDetect import SkinDetect
import numpy as np



class model_FaceAnalysis:   
    def __init__(self, strength = 0.2):
        self.sd = SkinDetect(strength)
        self.count = 0
        self.meanRGB = []
    
    def DetectSkin(self, frame, fs):
        if self.count >= fs*2: #throw away first 2 seconds
            face = DetectFace(frame, ROI_list=['face'])[0]
            sd = self.sd
            
            if self.count == fs*2:
                sd.compute_stats(face)
    
            skinFace, Threshold = sd.get_skin(
                face, filt_kern_size=0, verbose=False, plot=False)
            
            b, g, r = self.rgb_mean(skinFace)
            self.meanRGB.append([r,g,b])
            
            
            
        
        self.count += 1
        
        return self.meanRGB
        
    def rgb_mean(self, skinFace):
        ''' This function calculates mean of the each channel for the 
        given image
        '''
        # Count how many colored pixel
        non_black_pixels_mask = np.any(skinFace != [0, 0, 0], axis=-1)
        num_pixel = np.count_nonzero(non_black_pixels_mask)

        # Count the sum for all pixels
        RGB_sum = np.sum(skinFace, axis=(0, 1))

        return RGB_sum/num_pixel
        
            
        