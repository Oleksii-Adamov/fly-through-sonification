import cv2
import numpy as np


def get_part_that_went_offscreen(frame, prev_frame, video_w, video_h):
    # Initialize the SIFT feature detector and extractor
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors for both images
    prev_keypoints, prev_descriptors = sift.detectAndCompute(prev_frame, None)
    cur_keypoints, cur_descriptors = sift.detectAndCompute(frame, None)

    # Initialize the feature matcher using FLANN matching
    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Match the descriptors using FLANN matching
    matches_flann = flann.match(cur_descriptors, prev_descriptors)

    src_points = np.float32([cur_keypoints[m.queryIdx].pt for m in matches_flann]).reshape(-1, 1, 2)
    dst_points = np.float32([prev_keypoints[m.trainIdx].pt for m in matches_flann]).reshape(-1, 1, 2)

    # Estimate the homography matrix using RANSAC
    homography, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

    # Print the estimated homography matrix
    pts = np.array([[0, 0], [0, video_h], [video_w, video_h], [video_w, 0]], np.float32)
    pts = pts.reshape(-1, 1, 2).astype(np.float32)
    dst = cv2.perspectiveTransform(pts, homography)
    quadrangle = np.empty((4, 2))
    for i in range(0, dst.shape[0]):
        quadrangle[i] = dst[i][0]
        quadrangle[i][quadrangle[i] < 0] = 0
    cv2.polylines(prev_frame, [quadrangle.astype(int)], True, (0, 0, 255), 1)
    cv2.imshow("prev_frame", prev_frame)