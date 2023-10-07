# from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
# from deep_sort.tools import generate_detections as gdet
# from deep_sort.deep_sort import nn_matching
# from deep_sort.deep_sort.detection import Detection
from deep_sort_realtime.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.tools import generate_detections as gdet
from deep_sort_realtime.deep_sort import nn_matching
from deep_sort_realtime.deep_sort.detection import Detection
import numpy as np
# from deep_sort_realtime.deepsort_tracker import DeepSort
from sort.sort import Sort


class Tracker:
    tracker = None
    encoder = None
    tracks = []
    lost_tracks = []

    def __init__(self):
        # max_cosine_distance = 0.4
        # nn_budget = None
        #
        # encoder_model_filename = 'model_data/mars-small128.pb'
        #
        # metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        # self.tracker = DeepSortTracker(metric, max_age=0)
        # self.encoder = gdet.create_box_encoder(encoder_model_filename, batch_size=1)

        # self.tracker = DeepSort(max_age=0)

        self.tracker = Sort()

    def update(self, frame, detections):

        if len(detections) == 0:
            # self.tracker.predict()
            self.tracker.update([])
            # self.update_tracks()
            return

        bboxes = np.asarray([d[:-1] for d in detections])
        bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, 0:2]
        scores = [d[-1] for d in detections]

        # features = self.encoder(frame, bboxes)

        # dets = []
        # for bbox_id, bbox in enumerate(bboxes):
        #     dets.append(Detection(bbox, scores[bbox_id], features[bbox_id]))

        # self.tracker.predict()
        # self.tracker.update(dets)
        # # self.tracker.update_tracks(bboxes, frame)
        # self.update_tracks()
        detections = np.array(detections)
        # print(detections)
        track_bbs_ids = self.tracker.update(detections)

    def update_tracks(self):
        tracks = []
        lost_tracks = []
        for track in self.tracker.tracks:
            bbox = track.to_tlbr()
            if not track.is_confirmed() or track.time_since_update > 1:
                lost_tracks.append(Track(track, bbox))
            elif track.is_deleted():
                pass
            else:
                tracks.append(Track(track, bbox))

        self.tracks = tracks
        self.lost_tracks = lost_tracks


class Track:
    internal_track = None
    bbox = None

    def __init__(self, internal_track, bbox):
        self.internal_track = internal_track
        self.bbox = bbox