import math

import cv2
import numpy as np

from BigStar import BigStar, find_big_stars
from Nebulae import Nebulae, find_nebulae
from SmallStar import SmallStar, find_small_stars
from utils import assign_subsquare


def get_objects_from_frame(frame, gray_frame, small_star_box_size, mask = None):
    obj_dict = {}
    obj_dict['big_stars'] = find_big_stars(frame, gray_frame)

    if mask is None:
        mask = np.full((gray_frame.shape[0], gray_frame.shape[1]), False)

    for big_star in obj_dict['big_stars']:
        for row in range(max(int(big_star.y - big_star.diameter / 2), 0),
                         min(math.ceil(big_star.y + big_star.diameter / 2), gray_frame.shape[0])):
            for column in range(max(int(big_star.x - big_star.diameter / 2), 0),
                                min(math.ceil(big_star.x + big_star.diameter / 2), gray_frame.shape[1])):
                mask[row][column] = True

    obj_dict['small_stars'] = find_small_stars(frame, gray_frame, mask)

    for big_star in obj_dict['big_stars']:
        assign_subsquare(gray_frame, big_star.x, big_star.y, math.ceil(big_star.diameter / 2), 0)

    for small_star in obj_dict['small_stars']:
        assign_subsquare(gray_frame, small_star.x, small_star.y, small_star_box_size, 0)

    obj_dict['nebulae'] = find_nebulae(frame, gray_frame)

    return obj_dict


def track_objects(frame, video_w, video_h, tracked_objects, tracker, is_dynamic = True, visualize = False):
    small_star_box_size = 4
    frame = cv2.resize(frame, (video_w, video_h))
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = None
    if is_dynamic:
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

    objects = get_objects_from_frame(frame, gray_frame, small_star_box_size, mask)

    detections = []
    for small_star in objects['small_stars']:
        x1, y1, x2, y2 = int(small_star.x - small_star_box_size), int(small_star.y - small_star_box_size),\
                         math.ceil(small_star.x + small_star_box_size), math.ceil(small_star.y + small_star_box_size)
        detections.append([x1, y1, x2, y2, 1.0])
    for big_star in objects['big_stars']:
        r = max(big_star.diameter / 2, small_star_box_size)
        x1, y1, x2, y2 = int(big_star.x - r), int(big_star.y - r),\
                         math.ceil(big_star.x + r), math.ceil(big_star.y + r)
        detections.append([x1, y1, x2, y2, 1.0])

    trackers, unmatched_trackers = tracker.update(detections, objects['small_stars'] + objects['big_stars'])
    for track in trackers:
        tracked_objects[int(track.id)] = track.data

    objects['stars_went_offscreen'] = []
    for track in unmatched_trackers:
        x1, y1, x2, y2 = track.get_state()[0]
        if x1 < 0 or y1 < 0 or x2 > video_w or y2 > video_h:
            objects['stars_went_offscreen'].append(tracked_objects[track.id])
    if visualize:
        for track in trackers:
            x1, y1, x2, y2 = track.get_state()[0]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        for star in objects['stars_went_offscreen']:
            cv2.circle(frame, (int(star.x), int(star.y)), small_star_box_size, (255, 0, 0), 1)
        for polygon in objects['nebulae'].contours:
            cv2.polylines(frame, [polygon], True, (0, 255, 0), 2)

    objects['nebulae'] = objects['nebulae'].colorful_points
    return objects, frame





