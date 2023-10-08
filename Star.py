import cv2
import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder

from RGBColor import RGBColor


class Star:
    def __init__(self, x, y, flux, color, diameter = None):
        self.x = x
        self.y = y
        self.flux = flux
        self.color = color
        self.diameter = diameter

    def __repr__(self):
        return "x = " + str(self.x) + ", y = " + str(self.y) + ", flux = " + str(
            self.flux) + ", color: " + str(self.color) + ", diameter = " + str(self.diameter)

    def __str__(self):
        return self.__repr__()


def find_small_stars(image, gray_image, mask = None):
    mean, median, std = sigma_clipped_stats(gray_image, sigma=3.0)

    # threshold = 5 or 0.65 or 3 or 1.5 or 7, fwhm = 3 or 4
    daofind = DAOStarFinder(fwhm=3, threshold=7 * std)
    sources = daofind(gray_image - median, mask)
    small_stars = []
    if not sources is None:
        positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
        for i, position in enumerate(positions):
            x, y = int(round(position[0])), int(round(position[1]))
            small_stars.append(Star(x, y, gray_image[y, x], RGBColor(image[y, x, 2], image[y, x, 1], image[y, x, 0])))
    return small_stars


def find_big_stars(image, gray_image):

    _, thresholded_frame = cv2.threshold(gray_image, 0.98 * 255, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    big_stars = [None] * len(contours)
    for i, contour in enumerate(contours):
         x,y,w,h = cv2.boundingRect(cv2.approxPolyDP(contour, epsilon=4, closed=True))
         center_x = min(int(round(x + w / 2)), gray_image.shape[1] - 1)
         center_y = min(int(round(y + h / 2)), gray_image.shape[0] - 1)
         d = max(w, h)
         big_stars[i] = Star(center_x, center_y, gray_image[center_y, center_x],
                            RGBColor(image[center_y, center_x, 2], image[center_y, center_x, 1], image[center_y, center_x, 0]), d)

    return big_stars