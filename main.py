import cv2
from analyze_video import track_objects
from sort_tracking import SortTracker
from sonification.sonification import SonificationTools

def sonificate_video(objects_from_video, frames_to_go_static, vid_w, vid_h, length):
    static_small_stars = []
    static_big_stars = []
    for pair in objects_from_video:
        frame_objects = pair[0]
        frame_n = pair[1]
        if frame_n % frames_to_go_static == 0:
            for star in frame_objects['stars']:
                if star.diameter is None:
                    static_small_stars.append([star, frame_n])
                else:
                    static_big_stars.append([star, frame_n])
    sss = SonificationTools(vid_w, vid_h, length)
    sss.sonificate_small_stars(sss.get_data_from_small_stars_in_list(static_small_stars, frames_to_go_static, 100))

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
    frames_to_go_static = 10
    is_dynamic = True
    objects_from_video = []
    frame_number = 0
    while True:
        ret, frame = video_cap.read()
        if not ret:
            break

        objects, visualized_frame = track_objects(frame, video_w, video_h, tracked_objects, tracker, is_dynamic, visualize=True)
        objects_from_video.append([objects, frame_number])
        frame_number += 1
        if visualize:
            cv2.imshow("Visualization", visualized_frame)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    video_cap.release()
    cv2.destroyAllWindows()
    #print(objects_from_video)
    sonificate_video(objects_from_video, frames_to_go_static, video_w, video_h, 10)
