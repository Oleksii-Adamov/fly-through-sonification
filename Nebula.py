import cv2
import numpy as np


class Nebula:
    def __init__(self, contour, is_tracked, color = None):
        self.contour = contour
        self.is_tracked = is_tracked
        self.color = color


def find_nebulae(gray_image):
    _, thresholded_frame = cv2.threshold(gray_image, 0.05*255, 255, cv2.THRESH_BINARY)
    # dilate ?
    # cv2.CHAIN_APPROX_SIMPLE
    contours, _ = cv2.findContours(thresholded_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    nebulae = []
    for i, contour in enumerate(contours):
        polinomial = cv2.approxPolyDP(contour, epsilon=15, closed=True)
        if polinomial.shape[0] > 10:
            polinomial = polinomial.astype('float32')
            nebulae.append(Nebula(polinomial, np.full((len(contours)), True)))
    return nebulae