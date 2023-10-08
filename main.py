import cv2
import numpy as np
from astropy.convolution import convolve
from astropy.visualization import ImageNormalize, SqrtStretch
from photutils.segmentation import make_2dgaussian_kernel, detect_sources, SourceCatalog

from analyze_video import track_objects_dynamic
from photutils.background import Background2D, MedianBackground
import matplotlib.pyplot as plt

if __name__ == '__main__':
    visualize = True
    min_visualization_size = 2
    video_w = 960
    video_h = 540
    video_cap = cv2.VideoCapture('data/cosmic_reef_1920.mp4')
    # Check if camera opened successfully
    if video_cap.isOpened() == False:
        print("Error opening video stream or file")
    # , number_of_frames = 60
    objects_by_frames = track_objects_dynamic(video_cap, video_w, video_h, visualize = True)
    # print(objects_by_frames)
    # print(objects_by_frames)
    # sonify objects['small_stars_went_offscreen']
    # while True:
    #     ret, frame = video_cap.read()
    #     if not ret:
    #         break
    #     frame = cv2.resize(frame, (video_w, video_h))
    #     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     # bkg_estimator = MedianBackground()
    #     # bkg = Background2D(gray_frame, (50, 50), filter_size=(3, 3),
    #     #
    #     #                    bkg_estimator=bkg_estimator)
    #     # gray_frame -= bkg.background.astype('uint8')  # subtract the background
    #     #threshold = 1.5 * bkg.background_rms
    #     threshold = 0.7 * 255
    #     kernel = make_2dgaussian_kernel(3.0, size=5)  # FWHM = 3.0
    #
    #     convolved_data = convolve(gray_frame, kernel)
    #     # convolved_data = gray_frame
    #     segment_map = detect_sources(convolved_data, threshold, npixels=10)
    #     cat = SourceCatalog(gray_frame, segment_map, convolved_data=convolved_data)
    #     sources = cat.to_table()
    #     positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
    #     for i, position in enumerate(positions):
    #         x, y = int(round(position[0])), int(round(position[1]))
    #         cv2.rectangle(frame,
    #                       (int(x - 4), int(y - 4)),
    #                       (int(x + 4), int(y + 4)),
    #                       (0, 0, 255), 2)
    #     cv2.imshow("Video", frame)
    #     # Press Q on keyboard to  exit
    #     if cv2.waitKey(25) & 0xFF == ord('q'):
    #         break
        # norm = ImageNormalize(stretch=SqrtStretch())
        #
        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12.5))
        #
        # ax1.imshow(gray_frame, origin='lower', cmap='Greys_r', norm=norm)
        #
        # ax1.set_title('Background-subtracted Data')
        #
        # plt.imshow(segment_map, origin='lower', cmap=segment_map.cmap,
        #
        #            interpolation='nearest')
        #
        # ax2.set_title('Segmentation Image')
        #
        # plt.show()

    video_cap.release()
    cv2.destroyAllWindows()