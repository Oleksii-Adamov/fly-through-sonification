import cv2

from RGBColor import RGBColor


class BigStar:
    def __init__(self, x, y, diameter, flux, color = None):
        self.x = x
        self.y = y
        self.diameter = diameter
        self.flux = flux
        self.color = color

    def __repr__(self):
        return "x = " + str(self.x) + ", y = " + str(self.y) + ", diameter = " + str(self.diameter) +  ", flux" +\
               ", color = " + str(self.color)

    def __str__(self):
        return self.__repr__()


def find_big_stars(image, gray_image):

    _, thresholded_frame = cv2.threshold(gray_image, 0.98 * 255, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    big_stars = [None] * len(contours)
    for i, contour in enumerate(contours):
         x,y,w,h = cv2.boundingRect(cv2.approxPolyDP(contour, epsilon=4, closed=True))
         center_x = int(round(x + w / 2))
         center_y = int(round(y + h / 2))
         d = max(w, h)
         big_stars[i] = BigStar(center_x, center_y, d, gray_image[center_y, center_x],
                            RGBColor(image[center_y, center_x, 2], image[center_y, center_x, 1], image[center_y, center_x, 0]))

    return big_stars