import math

import cv2
import numpy as np

from Nebulae import find_nebulae_and_planets
from Planet import find_planets
from RGBColor import RGBColor
from Star import find_small_stars, find_big_stars
from utils import assign_subsquare


def get_objects_from_frame(frame, gray_frame, small_star_box_size, mask = None):
    obj_dict = {}
    obj_dict['stars'] = find_big_stars(frame, gray_frame)

    if mask is None:
        mask = np.full((gray_frame.shape[0], gray_frame.shape[1]), False)

    for big_star in obj_dict['stars']:
        for row in range(max(int(big_star.y - big_star.diameter / 2), 0),
                         min(math.ceil(big_star.y + big_star.diameter / 2), gray_frame.shape[0])):
            for column in range(max(int(big_star.x - big_star.diameter / 2), 0),
                                min(math.ceil(big_star.x + big_star.diameter / 2), gray_frame.shape[1])):
                mask[row][column] = True

    obj_dict['stars'] = obj_dict['stars'] + find_small_stars(frame, gray_frame, mask)

    for star in obj_dict['stars']:
        if star.diameter is None:
            assign_subsquare(gray_frame, star.x, star.y, small_star_box_size, 0)
        else:
            assign_subsquare(gray_frame, star.x, star.y, math.ceil(star.diameter / 2), 0)

    obj_dict['nebulae'], obj_dict['planets'] = find_nebulae_and_planets(frame, gray_frame)

    return obj_dict


def track_objects(frame, video_w, video_h, tracked_objects, tracker, visualize = False):
    small_star_box_size = 4
    frame = cv2.resize(frame, (video_w, video_h))
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = None

    objects = get_objects_from_frame(frame, gray_frame, small_star_box_size, mask)

    detections = []
    for star in objects['stars']:
        if star.diameter is None:
            x1, y1, x2, y2 = int(star.x - small_star_box_size), int(star.y - small_star_box_size), \
                             math.ceil(star.x + small_star_box_size), math.ceil(
                star.y + small_star_box_size)
            detections.append([x1, y1, x2, y2, 1.0])
        else:
            r = max(star.diameter / 2, small_star_box_size)
            x1, y1, x2, y2 = int(star.x - r), int(star.y - r), \
                             math.ceil(star.x + r), math.ceil(star.y + r)
            detections.append([x1, y1, x2, y2, 1.0])

    trackers, unmatched_trackers = tracker.update(detections, objects['stars'])
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
            cv2.circle(frame, (int(star.x), int(star.y)), small_star_box_size, (255, 0, 0), 2)
        for polygon in objects['nebulae'].contours:
            cv2.polylines(frame, [polygon], True, (0, 255, 0), 2)
        for planet in objects['planets']:
            cv2.circle(frame, (int(planet.x), int(planet.y)), math.ceil(planet.diameter / 2), (0, 0, 255), 2)

    objects['nebulae'] = objects['nebulae'].colorful_points
    return objects, frame





