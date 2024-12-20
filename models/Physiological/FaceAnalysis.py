"""
@author: I.O.
"""

# from HKSI_functions import DetectFace
import cv2
import dlib
import numpy as np
import numpy.typing as npt

from .HKSI_SkinDetect import SkinDetect


class model_FaceAnalysis:
    def __init__(self, strength=0.2):
        self.sd = SkinDetect(strength)
        self.detector = dlib.get_frontal_face_detector()
        self.count = 0
        self.meanRGB = []

    def DetectSkin(self, frame, fs) -> npt.NDArray:
        if self.count >= fs * 2:  # throw away first 2 seconds
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces_detector = self.detector(frame_gray)
            if len(faces_detector) == 0:
                self.count += 1
                return np.array(self.meanRGB).reshape(-1, 3), self.count-1, False
            face_detector = faces_detector[0]
            x = face_detector.left()
            y = face_detector.top()
            w = face_detector.width()
            h = face_detector.height()
            face = frame[y : y + h, x : x + w, :]

            if self.count == fs * 2:
                self.sd.compute_stats(face)

            try:
                skinFace, Threshold = self.sd.get_skin(
                    face, filt_kern_size=0, verbose=False, plot=False
                )

                b, g, r = self.rgb_mean(skinFace)
                self.meanRGB.append([r, g, b])
                self.count += 1
                return np.array(self.meanRGB).reshape(-1, 3), self.count-1, True
            except ValueError:
                pass
        self.count += 1
        return np.array(self.meanRGB).reshape(-1, 3), self.count-1, False



    def rgb_mean(self, skinFace):
        """This function calculates mean of the each channel for the
        given image
        """
        # Count how many colored pixel
        non_black_pixels_mask = np.any(skinFace != [0, 0, 0], axis=-1)
        num_pixel = np.count_nonzero(non_black_pixels_mask)

        # Count the sum for all pixels
        RGB_sum = np.sum(skinFace, axis=(0, 1))

        return RGB_sum / num_pixel
