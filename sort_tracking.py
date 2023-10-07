import numpy as np
from sort import Sort


class SortTracker:
    tracker = None

    def __init__(self):
        self.tracker = Sort()

    def update(self, frame, detections):

        if len(detections) == 0:
            self.tracker.update([])
            return
        detections = np.array(detections)
        track_bbs_ids, unmatched_trackers = self.tracker.update(detections)
        return track_bbs_ids, unmatched_trackers