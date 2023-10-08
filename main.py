import cv2
from analyze_video import track_objects

if __name__ == '__main__':
    video_w = 960
    video_h = 540
    video_cap = cv2.VideoCapture('data/cosmic_reef_1920.mp4')
    # Check if camera opened successfully
    if video_cap.isOpened() == False:
        print("Error opening video stream or file")

    frames_to_go_static = 150
    is_dynamic = True
    while True:
        ret, objects = track_objects(video_cap, video_w, video_h, is_dynamic, visualize=True)

    video_cap.release()
    cv2.destroyAllWindows()