class RGBColor:
    def __init__(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b

    def __repr__(self):
        return "r = " + str(self.r) + ", g = " + str(self.g) + ", b = " + str(self.b)

    def __str__(self):
        return self.__repr__()