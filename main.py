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
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    original_video_w = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_h = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    visualization_video = cv2.VideoWriter('results/cosmic_reef_1920_visualization.mp4',
                                          cv2.VideoWriter_fourcc(*'mp4v'),
                                          fps, (original_video_w, original_video_h))

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
        visualization_video.write(cv2.resize(visualized_frame, (original_video_w, original_video_h)))
    video_cap.release()
    visualization_video.release()
    cv2.destroyAllWindows()

    sonificate_video(objects_from_video, frames_to_go_static, video_w, video_h, 60)


