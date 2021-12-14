import math

import numpy as np
from guidance_flight_env.properties import lat_geod_deg, lng_geoc_deg
from guidance_flight_env.utils.vector2d_old import Vector2D


class GeoPosition(object):
    def __init__(self, latitude_deg: float, longitude_deg: float):
        self.lat_deg = latitude_deg
        self.long_deg = longitude_deg

    def to_array(self):
        return [self.lat_deg, self.long_deg]

    def true_bearing_deg_to(self, destination: 'GeoPosition') -> float:
        """ Determines heading in degrees of course between self and destination """
        difference_vector = destination - self
        return difference_vector.heading()

    def distance_haversine_km(self, destination: 'GeoPosition') -> float:
        """ Determines distance from current point to a destination point for small distances """
        φ1_rad = np.deg2rad(destination.lat_deg)
        φ2_rad = np.deg2rad(self.lat_deg)
        λ1_rad = np.deg2rad(self.long_deg)
        λ2_rad = np.deg2rad(destination.long_deg)
        R_m = 6371e3

        x = (λ2_rad - λ1_rad) * math.cos((φ1_rad + φ2_rad) / 2)
        y = (φ2_rad - φ1_rad)
        distance_m = math.sqrt(x * x + y * y) * R_m

        distance_km = distance_m / 1000

        return distance_km

    @staticmethod
    def from_sim(sim: 'simulation.Simulation') -> 'GeoPosition':
        """ Return a GeodeticPosition object with lat and lon from simulation """
        lat_deg = sim[lat_geod_deg]
        lon_deg = sim[lng_geoc_deg]
        return GeoPosition(lat_deg, lon_deg)

    def __sub__(self, other) -> Vector2D:
        """ Returns difference between two coords as (delta_lat, delta_long) """
        return Vector2D(self.long_deg - other.long_deg, self.lat_deg - other.lat_deg)