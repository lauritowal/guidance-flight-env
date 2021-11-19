import math
import numpy as np


class Object3D():
    def __init__(self, x, y, z=0, heading=0):
        self.x = x
        self.y = y
        self.z = z
        self.heading = heading

    def distance_to_target(self, target: 'Object3D'):
        return np.sqrt(np.square(target.x - self.x) + np.square(target.y - self.y) + np.square(target.z - self.z))

    def direction_2d_deg(self):
        ''' 2D direction of self.x, self.y from origin'''
        direction_rad = math.atan2(self.x, self.y)
        direction_deg_normalised = (math.degrees(direction_rad) + 360) % 360
        return direction_deg_normalised

    def __sub__(self, other) -> 'Object3D':
        return Object3D(self.x - other.x,
                        self.y - other.y,
                        self.z - other.z)

    def __str__(self):
        return f'x: {self.x}, y: {self.y}, z: {self.z}'