import cv2

from analyze_video import track_objects_dynamic

if __name__ == '__main__':
    visualize = True
    min_visualization_size = 2
    video_w = 960
    video_h = 540
    video_cap = cv2.VideoCapture('data/cosmic_reef_1920.mp4')
    # Check if camera opened successfully
    if video_cap.isOpened() == False:
        print("Error opening video stream or file")

    objects_by_frames = track_objects_dynamic(video_cap, video_w, video_h, visualize = True, number_of_frames = 10)
    print(objects_by_frames)
    # sonify objects['small_stars_went_offscreen']
    video_cap.release()
    cv2.destroyAllWindows()