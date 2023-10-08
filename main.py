import cv2
from analyze_video import track_objects
from sort_tracking import SortTracker
from sonification.sonification import sonificate_video

if __name__ == '__main__':
    visualize = True
    video_w = 960
    video_h = 540
    video_cap = cv2.VideoCapture('data/cosmic_reef_1920.mp4')
    # Check if camera opened successfully
    if video_cap.isOpened() == False:
        print("Error opening video stream or file")

    tracked_objects = {}
    tracker = SortTracker()
    frames_to_go_static = 150
    is_dynamic = True
    objects_from_video = []
    frame_number = 0
    while True:
        ret, frame = video_cap.read()
        if not ret:
            break

        objects, visualized_frame = track_objects(frame, video_w, video_h, tracked_objects, tracker, True)
        objects_from_video.append([objects, frame_number])
        frame_number += 1
        if visualize:
            cv2.imshow("Visualization", visualized_frame)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    video_cap.release()
    cv2.destroyAllWindows()

    sonificate_video(objects_from_video, frames_to_go_static, video_w, video_h, 60)


