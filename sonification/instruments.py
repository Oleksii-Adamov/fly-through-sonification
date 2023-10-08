from .strauss.generator import Sampler
from .strauss.generator import Synthesizer


class Mallets(Sampler):
    def __init__(self):
        super().__init__("strauss/data/samples/mallets")


class Piano(Sampler):
    def __init__(self):
        super().__init__("sonification/piano")
        self.load_preset("piano")


class Violin(Sampler):
    def __init__(self):
        super().__init__("sonification/violin")
        self.load_preset("violin")


class Xylophon(Sampler):
    def __init__(self):
        super().__init__("sonification/xylophon")


class Hang(Sampler):
    def __init__(self):
        super().__init__("sonification/hang")


class Synth(Sampler):
    def __init__(self):
        super().__init__("sonification/synth")


class Pad(Synthesizer):
    def __init__(self):
        super().__init__()
        self.load_preset("nebula")


class Wind(Synthesizer):
    def __init__(self):
        super().__init__()
        self.load_preset("windy")
