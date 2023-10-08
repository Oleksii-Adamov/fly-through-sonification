import cv2
from analyze_video import track_objects
from sort_tracking import SortTracker
from sonification.sonification import SonificationTools


def sonificate_video(objects_from_video, frames_to_go_static, vid_w, vid_h, length):
    #prepare data

    static_small_stars = []
    static_big_stars = []
    dynamic_stars = []
    nebulae = []
    planets = []
    is_any_small = False
    is_any_big = False
    is_any_dyn = False
    is_any_neb = False
    is_any_plan = False

    for pair in objects_from_video:
        frame_objects = pair[0]
        frame_n = pair[1]

        #prepare static stars and planets
        if frame_n % frames_to_go_static == 0:
            for star in frame_objects['stars']:
                if star.diameter is None:
                    is_any_small = True
                    static_small_stars.append([star, frame_n])
                else:
                    is_any_big = True
                    static_big_stars.append([star, frame_n])
            for planet in frame_objects['planets']:
                is_any_plan = True
                planets.append([planet, frame_n])

        #prepare dynamic stars
        for star in frame_objects['stars_went_offscreen']:
            is_any_dyn = True
            dynamic_stars.append([star, frame_n])

        #prepare nebulae
        for point in frame_objects['nebulae']:
            is_any_neb = True
            nebulae.append([point, frame_n])



    sss = SonificationTools(vid_w, vid_h, length)

    chords_small_static = [["C4", "C5", "C6"]]
    chords_big_static = [["A3", "A4", "C5", "A6"]]
    chords_dynamic = [["A3", "A4", "E4", "C5"]]
    chords_nebulae = [["A4", "A5", "C5", "E5", "C6"]]
    chords_planets = [["C3", "E3", "C4", "E4"]]

    #sonification
    if is_any_small:
        sss.sonificate_small_stars(sss.get_data_from_small_stars_in_list(static_small_stars, frames_to_go_static, 1), chords_small_static)
    if is_any_big:
        sss.sonificate_big_stars(sss.get_data_from_big_stars_in_list(static_big_stars, frames_to_go_static), chords_big_static)
    if is_any_dyn:
        sss.sonificate_stars_went_offscreen(sss.get_data_from_dynamyc_stars_in_list(dynamic_stars, 1), chords_dynamic, 0.2)
    if is_any_neb:
        sss.sonificate_nebulae_point_list(sss.get_data_from_nebulas_in_list(nebulae), chords_nebulae, 0.2)
    if is_any_plan:
        sss.sonificate_planets(sss.get_data_from_planets_in_list(planets, frames_to_go_static), chords_planets, 0.1)

    #mix audiostreams
    sss.mix_all()


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
    frames_to_go_static = 50
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


