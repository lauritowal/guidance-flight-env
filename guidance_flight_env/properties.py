# Taken and adapted from https://github.com/Gor-Ren/gym-jsbsim

import math
import collections

class BoundedProperty(collections.namedtuple('BoundedProperty', ['name', 'description', 'min', 'max'])):
    def get_legal_name(self):
        return self.name


class Property(collections.namedtuple('Property', ['name', 'description'])):
    def get_legal_name(self):
        return self.name


# aero
alpha_rad = Property("aero/alpha-rad", "rad")

# atmosphere
## wind
wind_north_fps = BoundedProperty('atmosphere/wind-north-fps', 'Wind north [fps]', -100, 100)
wind_east_fps = BoundedProperty('atmosphere/wind-east-fps', 'Wind north [fps]', -100, 100)
wind_down_fps = BoundedProperty('atmosphere/wind-down-fps', 'Wind north [fps]', -100, 100)
### read only
total_wind_north_fps = Property('atmosphere/total-wind-north-fps', 'Total Wind north [fps]')
total_wind_east_fps = Property('atmosphere/total-wind-east-fps', 'Total Wind east [fps]')
total_wind_down_fps = Property('atmosphere/total-wind-down-fps', 'Total Wind down [fps]')

# attitude
altitude_sl_ft = BoundedProperty('position/h-sl-ft', 'altitude above mean sea level [ft]', -1400, 85000)
altitude_sl_m = BoundedProperty('position/h-sl-meters', 'altitude above mean sea level [meters]', -1400, 85000)
pitch_rad = BoundedProperty('attitude/pitch-rad', 'pitch [rad]', -0.5 * math.pi, 0.5 * math.pi)
roll_rad = BoundedProperty('attitude/roll-rad', 'roll [rad]', -math.pi, math.pi)
heading_deg = BoundedProperty('attitude/psi-deg', 'heading [deg]', 0, 360)
heading_true_rad = BoundedProperty('attitude/heading-true-rad', 'heading true [rad]', -2 * math.pi, 2 * math.pi)
sideslip_deg = BoundedProperty('aero/beta-deg', 'sideslip [deg]', -180, +180)

# position
lng_geoc_deg = BoundedProperty('position/long-gc-deg', 'geocentric longitude [deg]', -180, 180)
lat_geod_deg = BoundedProperty("position/lat-geod-deg", "geodetic latitude [deg]", -90, 90)
lat_geod_rad = Property("position/lat-geod-rad", "rad")

dist_travel_m = Property('position/distance-from-start-mag-mt', 'distance travelled from starting position [m]')
dist_from_start_lon_m = Property('position/distance-from-start-lon-mt', 'lon distance travelled from starting position [m]')
dist_from_start_lat_m = Property('position/distance-from-start-lat-mt', 'lat distance travelled from starting position [m]')

# Check http://ftp.igh.cnrs.fr/pub/flightgear/www/Docs/Scenery/CoordinateSystem/CoordinateSystem.html
# for the difference between geocentric and geodetic lat / lng
lat_gc_deg = BoundedProperty("position/lat-gc-deg", "deg", -180, 180)
lat_gc_rad = Property("position/lat-gc-rad", "rad")
long_gc_deg = BoundedProperty("position/long-gc-deg", "geocentric longitude [deg]", -180, 180)
long_gc_rad = Property("position/long-gc-rad", "rad")

# velocities
u_fps = BoundedProperty('velocities/u-fps', 'body frame x-axis velocity [ft/s]', -2200, 2200)
v_fps = BoundedProperty('velocities/v-fps', 'body frame y-axis velocity [ft/s]', -2200, 2200)
w_fps = BoundedProperty('velocities/w-fps', 'body frame z-axis velocity [ft/s]', -2200, 2200)
v_north_fps = BoundedProperty('velocities/v-north-fps', 'velocity true north [ft/s]', float('-inf'), float('+inf'))
v_east_fps = BoundedProperty('velocities/v-east-fps', 'velocity east [ft/s]', float('-inf'), float('+inf'))
v_down_fps = BoundedProperty('velocities/v-down-fps', 'velocity downwards [ft/s]', float('-inf'), float('+inf'))
p_radps = BoundedProperty('velocities/p-rad_sec', 'roll rate [rad/s]', -2 * math.pi, 2 * math.pi)
q_radps = BoundedProperty('velocities/q-rad_sec', 'pitch rate [rad/s]', -2 * math.pi, 2 * math.pi)
r_radps = BoundedProperty('velocities/r-rad_sec', 'yaw rate [rad/s]', -2 * math.pi, 2 * math.pi)
altitude_rate_fps = Property('velocities/h-dot-fps', 'Rate of altitude change [ft/s]')

# flightpath
# flight-path/psi-gt-rad

# controls state
aileron_left = BoundedProperty('fcs/left-aileron-pos-norm', 'left aileron position, normalised', -1, 1)
aileron_right = BoundedProperty('fcs/right-aileron-pos-norm', 'right aileron position, normalised', -1, 1)
elevator = BoundedProperty('fcs/elevator-pos-norm', 'elevator position, normalised', -1, 1)
rudder = BoundedProperty('fcs/rudder-pos-norm', 'rudder position, normalised', -1, 1)
throttle = BoundedProperty('fcs/throttle-pos-norm', 'throttle position, normalised', 0, 1)
gear = BoundedProperty('gear/gear-pos-norm', 'landing gear position, normalised', 0, 1)

# engines
engine_running = Property('propulsion/engine/set-running', 'engine running (0/1 bool)')
all_engine_running = Property('propulsion/set-running', 'set engine running (-1 for all engines)')
engine_thrust_lbs = Property('propulsion/engine/thrust-lbs', 'engine thrust [lb]')

# controls command
aileron_cmd = BoundedProperty('fcs/aileron-cmd-norm', 'aileron commanded position, normalised', -1., 1.)
elevator_cmd = BoundedProperty('fcs/elevator-cmd-norm', 'elevator commanded position, normalised', -1., 1.)
rudder_cmd = BoundedProperty('fcs/rudder-cmd-norm', 'rudder commanded position, normalised', -1., 1.)
throttle_cmd = BoundedProperty('fcs/throttle-cmd-norm', 'throttle commanded position, normalised', 0., 1.)
mixture_cmd = BoundedProperty('fcs/mixture-cmd-norm', 'engine mixture setting, normalised', 0., 1.)
throttle_1_cmd = BoundedProperty('fcs/throttle-cmd-norm[1]', 'throttle 1 commanded position, normalised', 0., 1.)
mixture_1_cmd = BoundedProperty('fcs/mixture-cmd-norm[1]', 'engine mixture 1 setting, normalised', 0., 1.)
gear_all_cmd = BoundedProperty('gear/gear-cmd-norm', 'all landing gear commanded position, normalised', 0, 1)

# simulation
sim_dt = Property('simulation/dt', 'JSBSim simulation timestep [s]')
sim_time_s = Property('simulation/sim-time-sec', 'Simulation time [s]')

# initial conditions
initial_altitude_ft = Property('ic/h-sl-ft', 'initial altitude MSL [ft]')
initial_terrain_altitude_ft = Property('ic/terrain-elevation-ft', 'initial terrain alt [ft]')
initial_longitude_geoc_deg = Property('ic/long-gc-deg', 'initial geocentric longitude [deg]')
initial_latitude_geod_deg = Property('ic/lat-geod-deg', 'initial geodesic latitude [deg]')
initial_u_fps = Property('ic/u-fps', 'body frame x-axis velocity; positive forward [ft/s]')
initial_v_fps = Property('ic/v-fps', 'body frame y-axis velocity; positive right [ft/s]')
initial_w_fps = Property('ic/w-fps', 'body frame z-axis velocity; positive down [ft/s]')
initial_p_radps = Property('ic/p-rad_sec', 'roll rate [rad/s]')
initial_q_radps = Property('ic/q-rad_sec', 'pitch rate [rad/s]')
initial_r_radps = Property('ic/r-rad_sec', 'yaw rate [rad/s]')
initial_roc_fpm = Property('ic/roc-fpm', 'initial rate of climb [ft/min]')
initial_heading_deg = Property('ic/psi-true-deg', 'initial (true) heading [deg]')
