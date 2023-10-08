import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder

from RGBColor import RGBColor


class SmallStar:
    def __init__(self, x, y, flux, color = None):
        self.x = x
        self.y = y
        self.flux = flux
        self.color = color

    def __repr__(self):
        return "x = " + str(self.x) + ", y = " + str(self.y) + ", flux = " + str(
            self.flux) + ", color: " + str(self.color)

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
            small_stars.append(SmallStar(x, y, gray_image[y, x], RGBColor(image[y, x, 2], image[y, x, 1], image[y, x, 0])))
    return small_stars