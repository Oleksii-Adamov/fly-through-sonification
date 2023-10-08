import numpy as np
from sort import Sort


class SortTracker:
    tracker = None

    def __init__(self):
        self.tracker = Sort(max_age=0, min_hits=0)

    def update(self, detections, data = None):

        if len(detections) == 0:
            self.tracker.update([])
            return
        detections = np.array(detections)
        track_bbs_ids, unmatched_trackers = self.tracker.update(detections, data)
        return track_bbs_ids, unmatched_trackers