from strauss.generator import Sampler
from strauss.generator import Synthesizer


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

class Pad(Synthesizer):
    def __init__(self):
        super().__init__()
        self.load_preset("nebula")
