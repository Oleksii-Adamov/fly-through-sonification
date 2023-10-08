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
    # blobDetectorParameters = cv2.SimpleBlobDetector_Params()
    # blobDetectorParameters.filterByArea = False
    # blobDetectorParameters.minDistBetweenBlobs = 5
    # blobDetectorParameters.filterByCircularity = False
    # blobDetectorParameters.filterByColor = False
    # blobDetectorParameters.filterByConvexity = False
    # blobDetectorParameters.filterByInertia = False
    # blobDetectorParameters.minThreshold = 0.97 * 255
    # blobDetectorParameters.maxThreshold = 255
    # blobDetectorParameters.thresholdStep = 0.01 * 255 # 5  # 1
    #
    # detector = cv2.SimpleBlobDetector_create(blobDetectorParameters)
    #
    # # Detect blobs.
    # keypoints = detector.detect(gray_image)
    #
    # big_stars = []
    # for keypoint in keypoints:
    #     # maybe compute flux more comprehensive
    #     flux = gray_image[round(keypoint.pt[1]), round(keypoint.pt[0])]
    #     # print(flux)
    #     big_stars.append(BigStar(keypoint.pt[0], keypoint.pt[1], keypoint.size, flux))

    _, thresholded_frame = cv2.threshold(gray_image, 0.98 * 255, 255, cv2.THRESH_BINARY)
    # thresholded_frame = cv2.dilate(thresholded_frame, cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4)))
    # consider not using RETR_TREE
    contours, _ = cv2.findContours(thresholded_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    big_stars = [None] * len(contours)
    for i, contour in enumerate(contours):
         x,y,w,h = cv2.boundingRect(cv2.approxPolyDP(contour, epsilon=4, closed=True))
         # print(x,y,w,h)
         #center_x = bbox[0] + (bbox[2] - bbox[0]) / 2
         #center_y = bbox[1] + (bbox[3] - bbox[1]) / 2
         center_x = x + w / 2
         center_y = y + h / 2
         d = max(w, h)
         big_stars[i] = BigStar(center_x, center_y, d, gray_image[int(center_y), int(center_x)])
         # big_stars[i] = BigStar(center_x, center_y, max(bbox[2] - bbox[0], bbox[3] - bbox[1]),
         #                        gray_image[int(center_x), int(center_y)])

    return big_stars