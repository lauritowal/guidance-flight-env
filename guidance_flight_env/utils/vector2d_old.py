import math
from guidance_flight_env.properties import v_east_fps, v_north_fps


class Vector2D(object):
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def heading_deg(self):
        """ Calculate heading in degrees of vector from origin """
        heading_rad = math.atan2(self.x, self.y)
        heading_deg_normalised = (math.degrees(heading_rad) + 360) % 360
        return heading_deg_normalised

    @staticmethod
    def from_sim(sim: 'simulation.Simulation') -> 'Vector2D':
        return Vector2D(sim[v_east_fps], sim[v_north_fps])