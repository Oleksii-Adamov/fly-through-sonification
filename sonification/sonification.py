import os
import random

from .strauss.score import Score
from .strauss.sources import Events, Objects
from .strauss.sonification import Sonification

import numpy as np
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
                        'time': lambda x: [float(i * self.length) / float(vid_w) for i in x],
                        'pitch': lambda x: (x-np.min(x))/(np.max(x)-np.min(x)),
                        'volume': lambda x: (x-np.min(x))/(np.max(x)-np.min(x)),
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
        if x_list_type not in [np.int32, np.int64, np.float32, np.float64]:
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
        if y_list_type not in [np.int32, np.int64, np.float32, np.float64]:
            theta = [y * float(180) / float(self.vid_h) for y in y_list]
            return np.array(theta)
        else:
            theta = y_list * float(180) / float(self.vid_h)
            return theta

    def get_data_from_small_stars_in_list(self, objects) -> dict:
        data = dict()

        if type(objects) == dict:
            smallStars = objects["small_stars"]
            data["time"] = np.array([star.x for star in smallStars])
        else:
            smallStars = []
            t = []
            for pair in objects:
                smallStars.append(pair[0])
                t.append(pair[1])
            data["time"] = np.array(t)

        data["phi"] = np.array([star.x for star in smallStars])
        data["theta"] = np.array([star.y for star in smallStars])
        data["pitch"] = np.array([star.y for star in smallStars])
        data["volume"] = np.array([star.flux for star in smallStars])

        return data

    def get_data_from_big_stars_in_list(self, objects) -> dict:
        data = dict()

        bigStars = objects["big_stars"]
        bigStars.append(bigStars[0])

        data["phi"] = np.array([star.x for star in bigStars])
        data["theta"] = np.array([star.y for star in bigStars])
        data["time"] = np.array([star.x for star in bigStars])
        data["pitch"] = np.array([star.y for star in bigStars])
        data["volume"] = np.array([star.flux for star in bigStars])

        return data

    def get_data_from_nebulas_in_list(self, list_of_nebulaes_data) -> list:
        data = list(
            {
                "pitch": 1,
                "phi": np.array([]),
                "theta": np.array([]),
                "time_evo": np.array([]),
                "volume": np.array([])
                #"pitch_shift": np.array([])
            }
            for i in range(len(list_of_nebulaes_data[0].contour)))

        for time_idx, nebulae in enumerate(list_of_nebulaes_data):
            for p_idx, point in enumerate(nebulae.contour):
                x = point[0][0]
                y = point[0][1]
                #data[p_idx]["pitch"] = np.append(data[p_idx]["pitch"], random.randint(1, 4))
                data[p_idx]["phi"] = np.append(data[p_idx]["phi"], x)
                data[p_idx]["theta"] = np.append(data[p_idx]["theta"], y)
                data[p_idx]["time_evo"] = np.append(data[p_idx]["time_evo"], time_idx/len(list_of_nebulaes_data))
                if nebulae.is_tracked[p_idx]:
                    vol = 1
                else:
                    vol = 0
                vol = 1
                data[p_idx]["volume"] = np.append(data[p_idx]["volume"], vol)

        return data

    def sonificate_small_stars_went_offscreen(self, data):
        # ---------Sources----------
        mapvals = self.mapvals.copy()
        maplims = self.maplims.copy()

        source = Events(data.keys())
        source.fromdict(data)
        source.apply_mapping_functions(mapvals, maplims)

        # -------Score------------
        score = Score([["C3", "C4", "C5", "C6", "C7"]], self.length)

        # --------Generator--------
        generator = Xylophon()

        # ------Sonification--------
        sonification = Sonification(score, source, generator, self.audio_system)
        sonification.render()
        sonification.save("out/small_stars_went_offscreen.wav", 1/200)
        sonification.notebook_display()

    def sonificate_small_stars(self, data):
        # ---------Sources----------
        mapvals = self.mapvals.copy()
        maplims = self.maplims.copy()

        source = Events(data.keys())
        source.fromdict(data)
        source.apply_mapping_functions(mapvals, maplims)

        # -------Score------------
        score = Score(self.chords, self.length)

        # --------Generator--------
        generator = Piano()

        # ------Sonification--------
        sonification = Sonification(score, source, generator, self.audio_system)
        sonification.render()
        sonification.save("out/small_stars.wav", 1)
        sonification.notebook_display()

    def sonificate_big_stars(self, data):
        # ---------Sources----------
        mapvals = self.mapvals.copy()
        maplims = self.maplims.copy()

        source = Events(data.keys())
        source.fromdict(data)
        source.apply_mapping_functions(mapvals, maplims)

        # -------Score------------
        score = Score([["A4"], ["A5"], ["C5"], ["E5"]], self.length)

        # --------Generator--------
        generator = Violin()

        # ------Sonification--------
        sonification = Sonification(score, source, generator, self.audio_system)
        sonification.render()
        sonification.save("out/big_stars.wav", 1)
        sonification.notebook_display()

    def sonificate_nebulae_point(self, point_data, filename="out/nebula_point.wav", chords = [["A2"]]):
        # ---------Sources---------
        mapvals = self.mapvals.copy()
        maplims = self.maplims.copy()

        source = Objects(point_data.keys())
        source.fromdict(point_data)
        source.apply_mapping_functions(map_funcs=mapvals, map_lims=maplims)

        # -------Score------------
        score = Score(chords, self.length)

        # --------Generator--------
        generator = Pad()

        # ------Sonification--------
        sonification = Sonification(score, source, generator, self.audio_system)
        sonification.render()
        sonification.save(filename, 1.0/200)

    def sonificate_nebulae_point_list(self, point_list):
        chords = [["C3"], ["C4"], ["C5"]]
        for idx, point in enumerate(point_list):
            self.sonificate_nebulae_point(point, f"out/nebula_point{idx}.wav", [chords[random.randint(0, len(chords)-1)]])
        for i in range(1, len(point_list)):
            file1_path = "out/nebula_point0.wav"
            file2_path = f"out/nebula_point{i}.wav"
            self.mix_2_wavs(file1_path, file2_path, file1_path)
            os.remove(file2_path)

    def mix_2_wavs(self, file_path1, file_path2, file_path_export):
        file1 = AudioSegment.from_wav(file_path1)
        file2 = AudioSegment.from_wav(file_path2)
        mixed = file1.overlay(file2)
        mixed.export(out_f=file_path_export, format="wav")
