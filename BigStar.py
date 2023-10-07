import cv2


class BigStar:
    def __init__(self, x, y, diameter, flux, color = None):
        self.x = x
        self.y = y
        self.diameter = diameter
        self.flux = flux
        self.color = color

    def __repr__(self):
        return "x = " + str(self.x) + ", y = " + str(self.y) + ", diameter = " + str(self.diameter) +  ", flux" +\
               ", color = Not implemented"

    def __str__(self):
        return self.__repr__()


def find_big_stars(gray_image):
    blobDetectorParameters = cv2.SimpleBlobDetector_Params()
    blobDetectorParameters.filterByArea = False
    blobDetectorParameters.minDistBetweenBlobs = 5
    blobDetectorParameters.filterByCircularity = False
    blobDetectorParameters.filterByColor = False
    blobDetectorParameters.filterByConvexity = False
    blobDetectorParameters.filterByInertia = False
    blobDetectorParameters.minThreshold = 0.97 * 255
    blobDetectorParameters.maxThreshold = 255
    blobDetectorParameters.thresholdStep = 0.01 * 255 # 5  # 1

    detector = cv2.SimpleBlobDetector_create(blobDetectorParameters)

    # Detect blobs.
    keypoints = detector.detect(gray_image)

    big_stars = []
    for keypoint in keypoints:
        # maybe compute flux more comprehensive
        flux = gray_image[round(keypoint.pt[1]), round(keypoint.pt[0])]
        # print(flux)
        big_stars.append(BigStar(keypoint.pt[0], keypoint.pt[1], keypoint.size, flux))

    return big_stars