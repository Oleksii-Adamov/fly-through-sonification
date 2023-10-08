import cv2
import numpy as np
from astropy.stats import sigma_clipped_stats


class Nebula:
    def __init__(self, contour, is_tracked, color = None):
        self.contour = contour
        self.is_tracked = is_tracked
        self.color = color


def find_nebulae(gray_image):
    # 0.05*255 20
    _, thresholded_frame = cv2.threshold(gray_image, 20, 255, cv2.THRESH_BINARY)
    # dilate ?
    # cv2.CHAIN_APPROX_SIMPLE CHAIN_APPROX_TC89_KCOS RETR_TREE RETR_LIST
    contours, _ = cv2.findContours(thresholded_frame, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    nebulae = []
    for i, contour in enumerate(contours):
        # epsilon=15
        polinomial = cv2.approxPolyDP(contour, epsilon=18, closed=True)
        if polinomial.shape[0] > 3:
            #polinomial = polinomial.astype('float32')
            polinomial = polinomial.astype('int32')
            polinomial = polinomial.reshape((polinomial.shape[0], 2))
            nebulae.append(Nebula(polinomial, np.full((len(contours)), True)))
    return nebulae