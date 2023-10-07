import cv2
from analyze_video import track_objects_dynamic
from sonification.sonification import SonificationTools

if __name__ == '__main__':
    visualize = True
    min_visualization_size = 2
    video_w = 960
    video_h = 540
    video_cap = cv2.VideoCapture('data/cosmic_reef_1920.mp4')
    # Check if camera opened successfully
    if video_cap.isOpened() == False:
        print("Error opening video stream or file")
    number_of_frames = 300
    objects_by_frames = track_objects_dynamic(video_cap, video_w, video_h, visualize = True, number_of_frames=number_of_frames)

    sonif = SonificationTools(video_w, video_h, number_of_frames / 24);
    small_stars_went_offscreen = []
    nebulae = []
    for t, objects in enumerate(objects_by_frames):
        for star in objects['small_stars_went_offscreen']:
            small_stars_went_offscreen.append([star , t])
        for neb in objects['nebulae']:
            nebulae.append([neb, t])


    sonif.sonificate_small_stars_went_offscreen(sonif.get_data_from_small_stars_in_list(small_stars_went_offscreen))
    sonif.sonificate_nebulae_point_list(sonif.get_data_from_nebulas_in_list(nebulae))
    sonif.mix_2_wavs("out/small_stars_went_offscreen.wav", "out/nebula.wav", "out/result.wav")
    sonif.sonificate_windy()
    sonif.mix_2_wavs("out/windy.wav", "out/result.wav", "out/result2.wav")

    video_cap.release()
    cv2.destroyAllWindows()