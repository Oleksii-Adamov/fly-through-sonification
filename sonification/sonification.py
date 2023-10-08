import os
import random

from .strauss.score import Score
from .strauss.sources import Events, Objects
from .strauss.sonification import Sonification

import shutil
import numpy as np
import os
from pydub import AudioSegment
from .instruments import *



class SonificationTools:
    def __init__(self, vid_w, vid_h, length, chords=None):
        if chords is None:
            chords = [["A5", "C5", "C6"]]

        self.vid_w = vid_w
        self.vid_h = vid_h
        self.length = length
        self.chords = chords
        self.audio_system = "stereo"
        self.mapvals = {'phi': lambda x: [self.convert_x_to_phi(x_coord) for x_coord in x],
                        'theta': lambda x: [self.convert_y_to_theta(y_coord) for y_coord in x],
                        'time': lambda x: (x-0)/(length-0),
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
                t.append((pair[0].x - pair[1] + static_time) / static_time)

        data["time"] = np.array(t)
        data["phi"] = np.array([star.x for star in smallStars])
        data["theta"] = np.array([star.y for star in smallStars])
        data["pitch"] = np.array([star.color.r for star in smallStars])
        data["volume"] = np.array([star.flux for star in smallStars])

        return data

    def get_data_from_big_stars_in_list(self, objects, static_time) -> dict:
        data = dict()

        data["phi"] = np.array([star[0].x for star in objects])
        data["theta"] = np.array([star[0].y for star in objects])
        data["time"] = np.array([((star[0].x - star[1] + static_time) / static_time) for star in objects])
        data["pitch"] = np.array([star[0].color.r for star in objects])
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
                data["pitch"] = np.append(data["pitch"], star.color.b)
                data["volume"] = np.append(data["volume"], star.flux)

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
            data["pitch"] = np.append(data["pitch"], point.color.b*0.5 + point.y*0.5)
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
            data["phi"] = np.append(data["phi"], (planet.x - t + static_time) / static_time)
            data["theta"] = np.append(data["theta"], planet.y)
            data["time"] = np.append(data["time"], t)
            data["pitch"] = np.append(data["pitch"], planet.color.b)
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
        sonification.notebook_display()

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
        sonification.notebook_display()

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
        sonification.notebook_display()

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
        sonification.notebook_display()

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
        sonification.notebook_display()
        sonification.save_combined()

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
        sonification.notebook_display()

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
