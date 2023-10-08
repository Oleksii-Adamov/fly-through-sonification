import os
import random
import colorsys

from .strauss.score import Score
from .strauss.sources import Events, Objects
from .strauss.sonification import Sonification

import shutil
import numpy as np
import os
from pydub import AudioSegment
from .instruments import *

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



    sss = SonificationTools(vid_w, vid_h, length, length*30)

    chords_small_static = [["C4", "C5", "C6"]]
    chords_big_static = [["A3", "A4", "C5"]]
    chords_dynamic = [["A3", "A4", "E4", "C5"]]
    chords_nebulae = [["A4", "A5", "C5", "E5", "C6"]]
    chords_planets = [["C3", "E3", "C4", "E4"]]

    #sonification
    if is_any_small:
        sss.sonificate_small_stars(sss.get_data_from_small_stars_in_list(static_small_stars, frames_to_go_static,100), chords_small_static, 0.4)
    if is_any_big:
        sss.sonificate_big_stars(sss.get_data_from_big_stars_in_list(static_big_stars, frames_to_go_static), chords_big_static, 0.3)
    if is_any_dyn:
        sss.sonificate_stars_went_offscreen(sss.get_data_from_dynamyc_stars_in_list(dynamic_stars, 1), chords_dynamic, 0.2)
    if is_any_neb:
        sss.sonificate_nebulae_point_list(sss.get_data_from_nebulas_in_list(nebulae), chords_nebulae, 0.1)
    if is_any_plan:
        sss.sonificate_planets(sss.get_data_from_planets_in_list(planets, frames_to_go_static), chords_planets, 0.01)

    #mix audiostreams
    sss.mix_all()

class SonificationTools:
    def __init__(self, vid_w, vid_h, length, scale):
        self.scale = scale
        self.vid_w = vid_w
        self.vid_h = vid_h
        self.length = length
        self.audio_system = "stereo"
        self.mapvals = {'phi': lambda x: [self.convert_x_to_phi(x_coord) for x_coord in x],
                        'theta': lambda x: [self.convert_y_to_theta(y_coord) for y_coord in x],
                        'time': lambda x: x,
                        'pitch': lambda x: x,
                        'volume': lambda x: x,
                        'time_evo': lambda x: x
                        }
        self.maplims = {'phi': (0, 360),
                        'theta': (0, 180),
                        'time': ('0', '110'),
                        'pitch': ('0', '100'),
                        'volume': ('0', '100'),
                        'time_evo':('0', '100')
                        }

    def get_h_fromrgb(self, rgb_point):
        (h, s, v) = colorsys.rgb_to_hsv(rgb_point.r / 255, rgb_point.g / 255, rgb_point.b / 255)
        return h

    def scale_between(self, x, a, b, x_max, x_min):
        return (b - a) * (x - x_min) / (x_max-x_min) + a

    def convert_x_to_phi(self, x_list):
        phi = []
        x_list_type = type(x_list)
        if x_list_type not in [float, np.int32, np.int64, np.float32, np.float64]:
            for x in x_list:
                x_with_coef = x * float(180) / float(self.vid_w)
                if x < self.vid_w / 2:
                    phi.append(90 - x_with_coef)
                else:
                    phi.append(360 - (x_with_coef - 90))

            return np.array(phi)
        else:
            x_with_coef = x_list * float(180) / float(self.vid_w)
            if x_list < self.vid_w / 2:
                phi = (90 - x_with_coef)
            else:
                phi = (360 - (x_with_coef - 90))

            return phi

    def convert_y_to_theta(self, y_list):
        y_list_type = type(y_list)
        if y_list_type not in [float, np.int32, np.int64, np.float32, np.float64]:
            theta = [y * float(180) / float(self.vid_h) for y in y_list]
            return np.array(theta)
        else:
            theta = y_list * float(180) / float(self.vid_h)
            return theta

    def get_data_from_small_stars_in_list(self, objects, static_time, flux_filter = 230) -> dict:
        data = dict()

        smallStars = []
        t = []
        for pair in objects:
            if pair[0].flux >= flux_filter:
                smallStars.append(pair[0])
                t.append(pair[1] + (pair[0].x * self.scale / self.vid_w) % static_time)

        data["time"] = np.array(t)
        data["phi"] = np.array([star.x for star in smallStars])
        data["theta"] = np.array([star.y for star in smallStars])
        data["pitch"] = np.array([self.get_h_fromrgb(star.color) for star in smallStars])
        data["volume"] = np.array([star.flux for star in smallStars])

        return data

    def get_data_from_big_stars_in_list(self, objects, static_time) -> dict:
        data = dict()

        data["phi"] = np.array([star[0].x for star in objects])
        data["theta"] = np.array([star[0].y for star in objects])
        data["time"] = np.array([star[1] + (star[0].x * self.scale / self.vid_w) % static_time for star in objects])
        data["pitch"] = np.array([self.get_h_fromrgb(star[0].color) for star in objects])
        data["volume"] = np.array([star[0].flux + star[0].diameter*10 for star in objects])

        return data

    def get_data_from_dynamyc_stars_in_list(self, objects, flux_filter = 230) -> dict:
        data = {
            "pitch": np.array([]),
            "phi": np.array([]),
            "theta": np.array([]),
            "time": np.array([]),
            "volume": np.array([])
        }

        for star, t in objects:
            if star.flux >= flux_filter:
                data["phi"] = np.append(data["phi"], star.x)
                data["theta"] = np.append(data["theta"], star.y)
                data["time"] = np.append(data["time"], t)
                data["pitch"] = np.append(data["pitch"], self.get_h_fromrgb(star.color))
                if star.diameter is None:
                    data["volume"] = np.append(data["volume"], star.flux/2)
                else:
                    data["volume"] = np.append(data["volume"], star.flux*0.5+star.diameter*0.5)

        return data

    def get_data_from_nebulas_in_list(self, list_of_nebulaes_data, step=30):
        data = {
                "pitch": np.array([]),
                "phi": np.array([]),
                "theta": np.array([]),
                "time": np.array([]),
                "volume": np.array([])
        }

        for idx in range(0, len(list_of_nebulaes_data), step):
            idx = random.randint(idx, idx + step - 1) % len(list_of_nebulaes_data)
            point = list_of_nebulaes_data[idx][0]
            t = list_of_nebulaes_data[idx][1]
            data["pitch"] = np.append(data["pitch"], self.get_h_fromrgb(point.color) + point.y)
            data["phi"] = np.append(data["phi"], point.x)
            data["theta"] = np.append(data["theta"], point.y)
            data["time"] = np.append(data["time"], t)
            data["volume"] = np.append(data["volume"], abs(1 - (point.x/self.vid_w)**2))


        return data

    def get_data_from_planets_in_list(self, objects, static_time) -> dict:
        data = {
            "pitch": np.array([]),
            "phi": np.array([]),
            "theta": np.array([]),
            "time": np.array([]),
            "volume": np.array([])
        }

        for planet, t in objects:
            data["phi"] = np.append(data["phi"], planet.x)
            data["theta"] = np.append(data["theta"], planet.y)
            data["time"] = np.append(data["time"], t + (planet.x * self.scale / self.vid_w) % static_time)
            data["pitch"] = np.append(data["pitch"], self.get_h_fromrgb(planet.color))
            data["volume"] = np.append(data["volume"], planet.diameter)

        return data

    def sonificate_stars_went_offscreen(self, data, chords, volume=0.3):
        # ---------Sources----------
        mapvals = self.mapvals.copy()
        maplims = self.maplims.copy()

        source = Events(data.keys())
        source.fromdict(data)
        source.apply_mapping_functions(mapvals, maplims)

        # -------Score------------
        score = Score(chords, self.length)

        # --------Generator--------
        generator = Hang()

        # ------Sonification--------
        sonification = Sonification(score, source, generator, self.audio_system)
        sonification.render()
        sonification.save("out/stars_went_offscreen.wav", volume)

    def sonificate_small_stars(self, data, chords, volume=0.5):
        # ---------Sources----------
        mapvals = self.mapvals.copy()
        maplims = self.maplims.copy()

        source = Events(data.keys())
        source.fromdict(data)
        source.apply_mapping_functions(mapvals, maplims)

        # -------Score------------
        score = Score(chords, self.length)

        # --------Generator--------
        generator = Xylophon()

        # ------Sonification--------
        sonification = Sonification(score, source, generator, self.audio_system)
        sonification.render()
        sonification.save("out/static_small_stars.wav", volume)

    def sonificate_big_stars(self, data, chords, volume=0.4):
        # ---------Sources----------
        mapvals = self.mapvals.copy()
        maplims = self.maplims.copy()

        source = Events(data.keys())
        source.fromdict(data)
        source.apply_mapping_functions(mapvals, maplims)

        # -------Score------------
        score = Score(chords, self.length)

        # --------Generator--------
        generator = Piano()

        # ------Sonification--------
        sonification = Sonification(score, source, generator, self.audio_system)
        sonification.render()
        sonification.save("out/static_big_stars.wav", volume)

    def sonificate_nebulae_point_list(self, data, chords, volume=0.25):
        # ---------Sources---------
        mapvals = self.mapvals.copy()
        maplims = self.maplims.copy()

        source = Events(data.keys())
        source.fromdict(data)
        source.apply_mapping_functions(mapvals, maplims)

        # -------Score------------
        score = Score(chords, self.length)

        # --------Generator--------
        generator = Violin()

        # ------Sonification--------
        sonification = Sonification(score, source, generator, self.audio_system)
        sonification.render()
        sonification.save("out/nebulae.wav", volume)

    def sonificate_windy(self, data = None):
        data = {
            "phi": np.array([self.vid_w / 2 for i in range(0, int(self.length))]),
            "theta": np.array([self.vid_h / 2 for i in range(0, int(self.length))]),
            "time_evo": np.array([i for i in range(0, int(self.length))]),
            "pitch": 1,
            #"volume": np.array([1]),
            #"time": np.array([3])
        }

        # ---------Sources---------
        mapvals = self.mapvals.copy()
        maplims = self.maplims.copy()


        source = Objects(data.keys())
        source.fromdict(data)
        source.apply_mapping_functions(mapvals, maplims)

        # -------Score------------
        chords = [["A3"]]
        score = Score(chords, self.length)

        # --------Generator--------
        generator = Wind()

        # ------Sonification--------
        sonification = Sonification(score, source, generator, self.audio_system)
        sonification.render()
        sonification.save("out/windy.wav", 0.005)

    def sonificate_planets(self, data, chords, volume=0.5):
        # ---------Sources----------
        mapvals = self.mapvals.copy()
        maplims = self.maplims.copy()

        source = Events(data.keys())
        source.fromdict(data)
        source.apply_mapping_functions(mapvals, maplims)

        # -------Score------------
        score = Score(chords, self.length)

        # --------Generator--------
        generator = Synth()

        # ------Sonification--------
        sonification = Sonification(score, source, generator, self.audio_system)
        sonification.render()
        sonification.save("out/planets.wav", volume)


    def mix_2_wavs(self, file_path1, file_path2, file_path_export):
        file1 = AudioSegment.from_wav(file_path1)
        file2 = AudioSegment.from_wav(file_path2)
        mixed = file1.overlay(file2)
        mixed.export(out_f=file_path_export, format="wav")

    def mix_all(self):
        for fname in os.listdir("out/result/"):
            os.remove("out/result/" + fname)

        filenames = os.listdir("out/")
        filenames.remove("result")
        if len(filenames) <= 1:
            return
        shutil.copy("out/"+filenames[0], "out/result/")
        os.rename("out/result/"+filenames[0], "out/result/"+"result.wav")
        for file_n in range(1, len(filenames)):
            self.mix_2_wavs("out/result/result.wav", "out/"+filenames[file_n], "out/result/result.wav")
