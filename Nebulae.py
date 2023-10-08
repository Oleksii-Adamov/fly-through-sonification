import cv2
import numpy as np
from astropy.stats import sigma_clipped_stats

from ColorfulPoint import ColorfulPoint
from RGBColor import RGBColor


class Nebulae:
    def __init__(self, contours=None, colorful_points=None):
        if colorful_points is None:
            self.colorful_points = []
        else:
            self.colorful_points = colorful_points

        if contours is None:
            self.contours = []
        else:
            self.contours = contours


def find_nebulae(image, gray_image):
    # 0.05*255 20
    _, thresholded_frame = cv2.threshold(gray_image, 20, 255, cv2.THRESH_BINARY)
    # cv2.CHAIN_APPROX_SIMPLE CHAIN_APPROX_TC89_KCOS RETR_TREE RETR_LIST
    contours, _ = cv2.findContours(thresholded_frame, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    nebulae = Nebulae()
    for i, contour in enumerate(contours):
        # epsilon=15
        polinomial = cv2.approxPolyDP(contour, epsilon=18, closed=True)
        if polinomial.shape[0] > 3:
            polinomial = polinomial.astype('int32')
            polinomial = polinomial.reshape((polinomial.shape[0], 2))
            nebulae.contours.append(polinomial)
            for pt in polinomial:
                nebulae.colorful_points.append(ColorfulPoint(pt[0], pt[1], RGBColor(image[pt[1], pt[0], 2],
                                                                            image[pt[1], pt[0], 1],
                                                                            image[pt[1], pt[0], 0])))
    return nebulae