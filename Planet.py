import cv2
import numpy as np


class Planet:
    def __init__(self, x, y, diameter, color = None):
        self.x = x
        self.y = y
        self.color = color
        self.diameter = diameter

    def __repr__(self):
        return "x = " + str(self.x) + ", y = " + str(self.y) + ", diameter = " + str(self.diameter) + ", color: " + str(self.color)

    def __str__(self):
        return self.__repr__()


def find_planets(image, gray_image, thresholded_frame = None):
    planets = []
    # if thresholded_frame is None:
    #     _, thresholded_frame = cv2.threshold(gray_image, 20, 255, cv2.THRESH_BINARY)
    # circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, 1, 30, param1=100, param2=30, minRadius=10, maxRadius=100)
    # if circles is not None:
    #     circles = np.round(circles[0, :]).astype("int")
    #     for (x, y, r) in circles:
    #         planets.append(Planet(x, y, r * 2))
    # blobDetectorParameters = cv2.SimpleBlobDetector_Params()
    # blobDetectorParameters.filterByArea = True
    # blobDetectorParameters.minArea = 10
    # blobDetectorParameters.minDistBetweenBlobs = 5
    # blobDetectorParameters.filterByCircularity = True
    # blobDetectorParameters.minCircularity = 0.95
    # blobDetectorParameters.filterByColor = False
    # blobDetectorParameters.filterByConvexity = False
    # blobDetectorParameters.filterByInertia = False
    # blobDetectorParameters.minThreshold = 1
    # blobDetectorParameters.minThreshold = 3
    # blobDetectorParameters.thresholdStep = 1
    #
    # detector = cv2.SimpleBlobDetector_create(blobDetectorParameters)
    #
    # # Detect blobs.
    # keypoints = detector.detect(gray_image)
    #
    # planets = []
    # for keypoint in keypoints:
    #     # maybe compute flux more comprehensive
    #     flux = gray_image[round(keypoint.pt[1]), round(keypoint.pt[0])]
    #     print(keypoint.pt[0], keypoint.pt[1], keypoint.size)
    #     planets.append(Planet(keypoint.pt[0], keypoint.pt[1], keypoint.size))

    return planets