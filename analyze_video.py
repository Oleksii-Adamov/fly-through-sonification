import math

import cv2
import numpy as np

from BigStar import BigStar, find_big_stars
from Nebula import Nebula, find_nebulae
from SmallStar import SmallStar, find_small_stars
from sort_tracking import SortTracker
from utils import assign_subsquare


def track_nebulae_by_contours(frame, gray_frame, prev_nebulae, prev_gray_frame):
    if len(prev_nebulae) == 0 or prev_gray_frame is None:
        return find_nebulae(gray_frame)
    else:
        lk_params = dict(winSize=(5, 5),
                         maxLevel=3,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        nebulae = []
        for nebula in prev_nebulae:
            new_points, status, err = cv2.calcOpticalFlowPyrLK(prev_gray_frame, gray_frame, nebula.contour, None, **lk_params)
            nebulae.append(Nebula(new_points, status))
        return nebulae


def get_objects_from_frame(frame, gray_frame, prev_gray_frame, prev_objects, mask = None):
    obj_dict = {}
    obj_dict['small_stars'] = []
    obj_dict['big_stars'] = []
    obj_dict['big_stars'] = find_big_stars(gray_frame)

    if mask is None:
        mask = np.full((gray_frame.shape[0], gray_frame.shape[1]), False)

    for big_star in obj_dict['big_stars']:
        for row in range(max(int(big_star.y - big_star.diameter / 2), 0),
                         min(math.ceil(big_star.y + big_star.diameter / 2), gray_frame.shape[0])):
            for column in range(max(int(big_star.x - big_star.diameter / 2), 0),
                                min(math.ceil(big_star.x + big_star.diameter / 2), gray_frame.shape[1])):
                mask[row][column] = True
        # assign_subsquare(mask, big_star.x, big_star.y, big_star.diameter / 2, True)

    obj_dict['small_stars'] = find_small_stars(gray_frame, mask)

    for big_star in obj_dict['big_stars']:
        assign_subsquare(gray_frame, big_star.x, big_star.y, big_star.diameter / 2, 0)

    for small_star in obj_dict['small_stars']:
        assign_subsquare(gray_frame, small_star.x, small_star.y, 10, 0)

    obj_dict['nebulae'] = track_nebulae_by_contours(frame, gray_frame, prev_objects['nebulae'], prev_gray_frame)

    return obj_dict


def track_objects_dynamic(video_cap, video_w, video_h, visualize = False, number_of_frames = None):
    objects_by_frames = []
    small_star_size = 4
    prev_gray_frame = None
    prev_objects = {'nebulae': []}
    tracker = SortTracker()
    frame_idx = -1
    while True:
        frame_idx = frame_idx + 1
        if number_of_frames is not None and frame_idx > number_of_frames:
            break
        ret, frame = video_cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (video_w, video_h))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        original_gray_frame = np.copy(gray_frame)

        mask = np.full((video_h, video_w), True)
        for i in range(0, video_h):
            for j in range(0, int(video_w * 0.05)):
                mask[i][j] = False
            for j in range(int(video_w * 0.95), video_w):
                mask[i][j] = False
        for j in range(0, video_w):
            for i in range(0, int(video_h * 0.05)):
                mask[i][j] = False
            for i in range(int(video_h * 0.95), video_h):
                mask[i][j] = False

        objects = get_objects_from_frame(frame, gray_frame, prev_gray_frame, prev_objects, mask)

        detections = []
        for small_star in objects['small_stars']:
            x1, y1, x2, y2 = int(small_star.x - small_star_size), int(small_star.y - small_star_size),\
                             math.ceil(small_star.x + small_star_size), math.ceil(small_star.y + small_star_size)
            detections.append([x1, y1, x2, y2, 1.0])

        trackers, unmatched_trackers = tracker.update(frame, detections)

        objects['small_stars_went_offscreen'] = []
        for track in unmatched_trackers:
            x1, y1, x2, y2, tracker_id = track
            if x1 < 0 or y1 < 0 or x2 > video_w or y2 > video_h:
                center_x = x1 + (x2 - x1) / 2
                center_y = y1 + (y2 - y1) / 2
                center_x = min(max(0, center_x), video_w)
                center_y = min(max(0, center_y), video_h)
                objects['small_stars_went_offscreen'].append(SmallStar(center_x, center_y, np.random.randint(50, 255)))

        if visualize:
            for track in trackers:
                x1, y1, x2, y2, tracker_id = track
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            for small_star in objects['small_stars_went_offscreen']:
                cv2.circle(frame, (int(small_star.x), int(small_star.y)), small_star_size, (255, 0, 0), 1)

        prev_gray_frame = original_gray_frame
        prev_objects = objects
        objects_by_frames.append(objects)
        if visualize:
            cv2.imshow("Video", frame)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
    return objects_by_frames




