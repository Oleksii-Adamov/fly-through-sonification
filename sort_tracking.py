import numpy as np
from sort import Sort


class SortTracker:
    tracker = None

    def __init__(self):
        self.tracker = Sort()

    def update(self, frame, detections, data = None, return_data=False):

        if len(detections) == 0:
            self.tracker.update([])
            return
        detections = np.array(detections)
        track_bbs_ids, unmatched_trackers, data = self.tracker.update(detections, data, return_data)
        return track_bbs_ids, unmatched_trackers, data