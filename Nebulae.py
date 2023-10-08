import cv2
import numpy as np
from astropy.stats import sigma_clipped_stats

from ColorfulPoint import ColorfulPoint
from Planet import Planet
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


def find_nebulae_and_planets(image, gray_image):
    _, thresholded_frame = cv2.threshold(gray_image, 15, 255, cv2.THRESH_BINARY)
    # cv2.CHAIN_APPROX_SIMPLE CHAIN_APPROX_TC89_KCOS RETR_TREE RETR_LIST cv2.RETR_CCOMP cv2.RETR_EXTERNAL
    contours, hierarchy = cv2.findContours(thresholded_frame, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    nebulae = Nebulae()
    planets = []
    for i, contour in enumerate(contours):
        is_planet = False
        # check for planet
        if hierarchy[0][i][3] == -1:
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            if perimeter > 0 and area > 100 and area < 4000:
                roundness = (4 * np.pi * area) / perimeter ** 2
                if roundness > 0.55:
                    is_planet = True
                    x, y, w, h = cv2.boundingRect(contour)
                    acc_color = np.array([0, 0, 0])
                    npix = 0.0
                    for row in range(y, y+h):
                        for col in range(x, x+w):
                            acc_color += image[row][col]
                            npix += 1
                    avg_color = acc_color / npix
                    planet = Planet(round(x+w/2), round(y+h/2), max(w, h), RGBColor(avg_color[2], avg_color[1], avg_color[0]))
                    planets.append(planet)

        if not is_planet:
            polinomial = cv2.approxPolyDP(contour, epsilon=18, closed=True)
            if polinomial.shape[0] > 2:
                polinomial = polinomial.astype('int32')
                polinomial = polinomial.reshape((polinomial.shape[0], 2))
                nebulae.contours.append(polinomial)
                for pt in polinomial:
                    nebulae.colorful_points.append(ColorfulPoint(pt[0], pt[1], RGBColor(image[pt[1], pt[0], 2],
                                                                                image[pt[1], pt[0], 1],
                                                                                image[pt[1], pt[0], 0])))

    return nebulae, planets
