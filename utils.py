import math


def assign_subsquare(matrix, center_x, center_y, radius, value):
    for row in range(max(int(center_y - radius), 0),
                     min(math.ceil(center_y + radius), matrix.shape[0])):
        for column in range(max(int(center_x - radius), 0),
                            min(math.ceil(center_x + radius), matrix.shape[1])):
            matrix[row][column] = value