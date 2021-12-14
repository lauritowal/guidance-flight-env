import math
import gym
import numpy as np
from PIL import Image
from gym.utils import seeding
from guidance_flight_env.pid.pid_controller import PidController
from guidance_flight_env.services.map_plotter_old import MapPlotter
from guidance_flight_env.simulation_old import Simulation
from guidance_flight_env.utils import utils_old as utils
from typing import Tuple
import guidance_flight_env.properties as prp


class CartesianPosition():
    def __init__(self, x, y, z=0, heading=0, offset=0):
        self.x = x
        self.y = y
        self.z = z

        self.heading = heading

        self.offset = offset

    def distance_to_target(self, target: 'CartesianPosition'):
        return np.sqrt(np.square(target.x - self.x) + np.square(target.y - self.y) + np.square(target.z - self.z))

    def vector_direction_deg(self):
        """ Calculate heading in degrees of vector from origin """
        heading_rad = math.atan2(self.x, self.y)
        heading_deg_normalised = (math.degrees(heading_rad) - self.offset + 360) % 360
        return heading_deg_normalised

    def direction_to_target_deg(self, target: 'CartesianPosition'):
        difference_vector = target - self
        return difference_vector.vector_direction_deg()

    def __sub__(self, other) -> 'CartesianPosition':
        """ Returns difference between two coords as (delta_lat, delta_long) """
        return CartesianPosition(self.x - other.x, self.y - other.y, self.z - other.z, heading=other.vector_direction_deg(), offset=self.offset)

    def __str__(self):
        return f'x: {self.x}, y: {self.y}, z: {self.z}'



class GuidanceEnv(gym.Env):
    MIN_DISTANCE_TO_TARGET_KM = 100 / 1000
    MIN_HEIGHT_FOR_FLARE_FT = 20
    MIN_HEIGHT_FOR_FLARE_M = MIN_HEIGHT_FOR_FLARE_FT / 3.281
    MAX_TARGET_DISTANCE_KM = 1.5  # ca. 30sec of flight for Cessna
    MAX_HEIGHT_FT = 3500
    HEIGHT_THRESHOLD_FT = 30
    HEIGHT_THRESHOLD_M = HEIGHT_THRESHOLD_FT / 3.281
    CRASH_HEIGHT_FT = 6

    JSBSIM_DT_HZ: int = 60  # JSBSim integration frequency

    pid_controller: PidController = PidController()

    metadata = {
        "render.modes": ["rgb-array", "flightgear", "jsbsim"]
    }

    episode_counter = 0

    continuous = False
    observation_space: gym.Space = gym.spaces.Box(-np.inf, np.inf, shape=(15,), dtype=np.float32)
    action_space: gym.Space = gym.spaces.Discrete(360)

    def __init__(self, config):
        self.min_runway_angle_threshold_deg = 5
        self.runway_angle_threshold_deg = 10
        self.render_progress_image = config["render_progress_image"]
        self.render_progress_image_path = config["render_progress_image_path"]
        self.target_radius_km = config["target_radius"]
        self.infos = []
        self.rewards = []
        self.done = False
        self.map_plotter = None
        self.last_track_error = 0
        self.last_track_error_perpendicular = 0

        if config["agent_interaction_freq"] > self.JSBSIM_DT_HZ:
            raise ValueError('agent interaction frequency must be less than '
                             'or equal to JSBSim integration frequency of '
                             f'{self.JSBSIM_DT_HZ} Hz.')

        self.sim: Simulation = None
        self.sim_steps_per_agent_step: int = self.JSBSIM_DT_HZ // config["agent_interaction_freq"]
        self.sim_steps = self.sim_steps_per_agent_step

        self.max_episode_time_s = config["max_episode_time_s"]

        # math.ceil: round up to next largest integer
        self.episode_steps = math.ceil(self.max_episode_time_s * config["agent_interaction_freq"])
        self.steps_left = self.episode_steps

        self.target_position = None

        self.localizer_position: CartesianPosition = None
        self.heading_at_localizer_deg = 0

        # set visualisation objects

        self.step_delay = None
        self.jsbsim_path = config["jsbsim_path"]

        self.aircraft = config["aircraft"]
        self.max_distance_km = config["max_distance_km"]

        self.max_target_distance_km = config["max_target_distance_km"]
        self.spawn_target_distance_km = 0.5

        self.last_state = None

        # Set the seed.
        self.np_random = None

        self.to_low_height = None
        if "evaluation" in config and config["evaluation"]:
            seed = config["seed"]
        else:
            seed = config["seed"] + config.worker_index + config.num_workers + config.vector_index
        self.seed(seed)

        self.runway_angle_deg = None

        self.glide_angle_deg = 4

        self.phase = config["phase"]

        self.last_distance_km = []
        self.last_runway_heading_error_deg = []

        self.last_distance_to_perpendicular_localizer_km = 0

        self.offset = config["offset"]

    def step(self, action: np.ndarray):
        if not (action.shape == self.action_space.shape):
            raise ValueError('mismatch between action and action space size')

        heading_deg = 0
        _delta_ft = 0
        if self.continuous:
            # for continuous action space: invert normalizaation and unpack action
            # action = utils.invert_normalization(x_normalized=action[0], min_x=0.0, max_x=360.0, a=-1, b=1)
            x = action[0]
            y = action[1]
            heading_deg = math.degrees(math.atan2(y, x))
            # altitude_delta_ft = np.interp(abs(action[2]), [-1, 1], [-100, 100])

        # print("altitude_delta_ft:", altitude_delta_ft)

        action_target_heading_deg = heading_deg % 360
        self.sim[prp.elevator_cmd] = self.pid_controller.elevator_hold(pitch_angle_reference=math.radians(0),
                                                                       pitch_angle_current=self.sim[prp.pitch_rad],
                                                                       pitch_angle_rate_current=self.sim[prp.q_radps])


        ground_speed = np.sqrt(np.square(self.sim[prp.v_north_fps]) + np.square(self.sim[prp.v_east_fps]))

        # replace with flight_path_angle_hold
        # self.sim[prp.elevator_cmd] = self.pid_controller.altitude_hold(altitude_reference_ft=self.sim[prp.altitude_sl_ft] + altitude_delta_ft,
        #                                                              altitude_ft=self.sim[prp.altitude_sl_ft],
        #                                                              ground_speed=ground_speed,
        #                                                              pitch_rad=self.sim[prp.pitch_rad],
        #                                                              alpha_rad=self.sim[prp.alpha_rad],
        #                                                              roll_rad=self.sim[prp.roll_rad],
        #                                                              q_radps=self.sim[prp.q_radps],
        #                                                              r_radps=self.sim[prp.r_radps])


        # self.sim[prp.elevator_cmd] = self.pid_controller.flight_path_angle_hold(gamma_reference_rad=math.radians(0),
        #                                                                       pitch_rad=self.sim[prp.pitch_rad],
        #                                                                       alpha_rad=self.sim[prp.alpha_rad],
        #                                                                       q_radps=self.sim[prp.q_radps],
        #                                                                       roll_rad=self.sim[prp.roll_rad],
        #                                                                       r_radps=self.sim[prp.r_radps])

        self.sim[prp.aileron_cmd] = self.pid_controller.heading_hold(
            heading_reference_deg=action_target_heading_deg,
            heading_current_deg=self.sim.get_heading_true_deg(),
            roll_angle_current_rad=self.sim[prp.roll_rad],
            roll_angle_rate=self.sim[prp.p_radps],
            true_air_speed=self.sim.get_true_air_speed()
        )

        for step in range(self.sim_steps):
            self.sim.run()

        reward = self._reward()
        state = self._get_observation()

        self.rewards.append(reward)
        self.last_state = state
        self.steps_left -= 1

        self.done = self._is_done()

        info = self.get_info(reward=reward)
        self.infos.append(info)

        if self.render_progress_image and self._is_done():
            rgb_array = self.render(mode="rgb_array")
            image: Image = Image.fromarray(rgb_array)
            image.save(f'{self.render_progress_image_path}/episode_{self.episode_counter}_{info["terminal_state"]}.png')
            print("done with episode: ", self.episode_counter)

        self.episode_counter += 1
        return state, reward, self.done, info

    def reset(self):
        initial_conditions = self._get_initial_conditions()

        if self.sim:
            self.sim.reinitialise(init_conditions=initial_conditions)
        else:
            self.sim = self._init_new_sim(self.JSBSIM_DT_HZ, self.aircraft, initial_conditions)

        self.steps_left = self.episode_steps

        # TODO: find more elegant solution...
        if self.phase == 0:
            self.spawn_target_distance_km = 0.5
            self.runway_angle_deg = self.np_random.uniform(-20, 20) % 360
        elif self.phase == 1:
            self.spawn_target_distance_km = 1
            self.runway_angle_deg = self.np_random.uniform(-45, 45) % 360
        elif self.phase == 2:
            self.spawn_target_distance_km = 1.5
            self.runway_angle_deg = self.np_random.uniform(-90, 90) % 360
        elif self.phase == 3:
            self.spawn_target_distance_km = 2
            self.runway_angle_deg = self.np_random.uniform(-120, 120) % 360
        elif self.phase == 4:
            self.spawn_target_distance_km = self.max_target_distance_km
            self.runway_angle_deg = self.np_random.uniform(0, 360)

        if self.max_distance_km is None:
            self.max_distance_km = self.sim.calculate_max_distance_km(self.max_episode_time_s)

        # self.sim.start_engines()  # start engines for testing the algorithm in simplest form
        # Important for heading hold and altitude control when motor on:
        # Mixture control - Sets the amount of fuel added to the intake airflow. At higher altitudes, the air pressure (and therefore the oxygen level) declines so the fuel volume must also be reduced to give the correct air–fuel mixture. This process is known as "leaning".
        # self.sim.set_throttle_mixture_controls(0.8, 0.7)
        self.sim.raise_landing_gear()
        self.sim.stop_engines()

        # if self.episode_counter > 100:
        #     self.runway_angle_deg = self.np_random.uniform(-90, 90) % 360
        self.runway_angle_deg = 0 # keep on 0! move offset after training...

        print(f"episode: {self.episode_counter}, runway anlge:", self.runway_angle_deg)

        self.target_position = self._generate_random_target_position()

        self.localizer_position = self._create_localizer()
        self.localizer_glide_position = self._create_localizer_glide()

        self.localizer_perpendicular_position = self._create_perpendicular_localizer()
        self.last_distance_to_perpendicular_localizer_km = self.max_distance_km

        relative_bearing_deg = utils.reduce_reflex_angle_deg(self.target_position.direction_to_target_deg(self.localizer_position) - self.runway_angle_deg)
        relative_bearing_to_perpendicular_deg = utils.reduce_reflex_angle_deg(self.target_position.direction_to_target_deg(self.localizer_perpendicular_position) - self.runway_angle_deg)

        self.example_point_position = self._create_example_point()

        cross_track_error = self._calc_cross_track_error(self.example_point_position, self.target_position)
        vertical_track_error = self._calc_vertical_track_error(self.example_point_position, self.target_position)

        cross_track_error_perpendicular = self._calc_cross_track_error(self.example_point_position,
                                                         self.localizer_perpendicular_position)

        print("cross_track_error example", f"{cross_track_error:.20f}")
        print("vertical_track_error example", f"{vertical_track_error:.20f}")
        print("cross_track_error_perpendicular example", f"{cross_track_error_perpendicular:.20f}")

        if abs(cross_track_error) < 0.1:
            print("smaller 0","-111")

        self.heading_at_localizer_deg = (self.runway_angle_deg - 180) % 360

        self.map_plotter = MapPlotter(target=self.target_position,
                                      glide_angle_deg=self.glide_angle_deg,
                                      aircraft_initial_position=CartesianPosition(0,0,0, offset=self.offset),
                                      target_radius_km=self.target_radius_km,
                                      localizer_position=self.localizer_position,
                                      localizer_glide_position=self.localizer_glide_position,
                                      localizer_perpendicular_position=self.localizer_perpendicular_position,
                                      target_spawn_area_radius_km=self.spawn_target_distance_km,
                                      example_position=self.example_point_position,
                                      bounds_radius_km=self.max_distance_km,
                                      runway_angle=self.runway_angle_deg,
                                      offset=self.offset)

        aircraft_position = self.aircraft_cartesian_position()
        self.last_distance_km.append(aircraft_position.distance_to_target(self.target_position))



        self.done = False

        runway_heading_error_deg = utils.reduce_reflex_angle_deg(self.sim.get_heading_true_deg() - self.runway_angle_deg)
        self.last_runway_heading_error_deg.append(runway_heading_error_deg)

        self.to_low_height = (self.target_position.z - 10 / 1000) * 3281

        self.infos = []
        info = self.get_info(0)
        self.infos.append(info)

        return self._get_observation()

    def render_html(self, path):
        self.map_plotter.plot_html(infos=self.infos, path=path)

    def _get_initial_conditions(self):
        return {
            prp.initial_altitude_ft: self.np_random.uniform(GuidanceEnv.MAX_HEIGHT_FT - GuidanceEnv.MAX_HEIGHT_FT / 3, GuidanceEnv.MAX_HEIGHT_FT),
            prp.initial_terrain_altitude_ft: 0.00000001,
            prp.initial_longitude_geoc_deg: self.np_random.uniform(-160, 160), # decreased range to avoid problems close to eqautor at -180 / 180
            prp.initial_latitude_geod_deg: self.np_random.uniform(-70, 70), # decreased range to avoid problems close to poles at -90 / 90
            prp.initial_u_fps: self.aircraft.get_cruise_speed_fps(),
            prp.initial_v_fps: 0,
            prp.initial_w_fps: 0,
            prp.initial_p_radps: 0,
            prp.initial_q_radps: 0,
            prp.initial_r_radps: 0,
            prp.initial_roc_fpm: 0,
            prp.initial_heading_deg: self.np_random.uniform(0, 360),
        }

    def _create_localizer(self):
        # 500 meters before runway...
        distance_km = -1
        heading = self.runway_angle_deg + self.offset
        # rotate from  N(90°);E(0°) to N(0°);E(90°)
        x = self.target_position.x + distance_km * math.cos(math.radians((heading - 90) % 360))
        y = self.target_position.y + distance_km * math.sin(math.radians((heading + 90) % 360))
        z = self.target_position.z

        localizer = CartesianPosition(x, y, z, heading=self.runway_angle_deg, offset=self.offset)
        return localizer

    def _create_localizer_glide(self):
        # 500 meters before runway...
        distance_km = -1
        heading = self.runway_angle_deg + self.offset
        # rotate from  N(90°);E(0°) to N(0°);E(90°)
        x = self.target_position.x + distance_km * math.cos(math.radians((heading - 90) % 360))
        y = self.target_position.y + distance_km * math.sin(math.radians((heading + 90) % 360))
        z = self.target_position.z + distance_km * math.sin(math.radians(-self.glide_angle_deg % 360))

        localizer = CartesianPosition(x, y, z, heading=self.runway_angle_deg, offset=self.offset)
        return localizer

    def _create_perpendicular_localizer(self):
        heading = self.runway_angle_deg + self.offset + 90

        # 500 meters before runway...
        distance_km = -1
        # rotate from  N(90°);E(0°) to N(0°);E(90°)
        x = self.localizer_position.x + distance_km * math.cos(math.radians((heading - 90) % 360))
        y = self.localizer_position.y + distance_km * math.sin(math.radians((heading + 90) % 360))

        localizer = CartesianPosition(x, y, heading=heading, offset=self.offset)
        return localizer


    def _create_example_point(self):
        distance = -0.5
        # 500 meters before runway...
        # rotate from  N(90°);E(0°) to N(0°);E(90°)
        x = self.target_position.x + distance * math.cos(math.radians((self.runway_angle_deg - 90 + self.offset) % 360))
        y = self.target_position.y + distance * math.sin(math.radians((self.runway_angle_deg + 90 - self.offset) % 360))
        z = self.target_position.z + distance * math.sin(math.radians(-self.glide_angle_deg % 360))

        return CartesianPosition(x, y, z, heading=self.runway_angle_deg, offset=self.offset)


    def _generate_random_target_position(self) -> (CartesianPosition, float):
        start_distance = 600 / 1000

        def random_sign():
            if self.np_random.random() < 0.5:
                return 1
            return -1

        x = self.np_random.uniform(0, self.spawn_target_distance_km) * random_sign()

        print("max_target_distance_km", self.spawn_target_distance_km)

        y = self.np_random.uniform(start_distance, self.spawn_target_distance_km) * random_sign()
        z = self.np_random.uniform(0.2, (self.sim[prp.initial_altitude_ft] / 3281) / 2) + GuidanceEnv.MIN_HEIGHT_FOR_FLARE_M / 1000

        return CartesianPosition(x, y, z, heading=self.runway_angle_deg, offset=self.offset)

    def _init_new_sim(self, dt, aircraft, initial_conditions):
        return Simulation(sim_frequency_hz=dt,
                          aircraft=aircraft,
                          init_conditions=initial_conditions,
                          jsbsim_path=self.jsbsim_path,
                          offset=self.offset)

    def render(self, mode='rgb_array') -> np.array:
        print_props: Tuple = (prp.u_fps, prp.altitude_sl_ft, prp.roll_rad, prp.sideslip_deg)

        if mode == 'html':
            self.map_plotter.plot_html(self.infos, path="./htmls/test.html")
        elif mode == 'rgb_array':
            '''
            rgb_array: Return an numpy.ndarray with shape (x, y, 3),
            representing RGB values for an x-by-y pixel image
            '''
            return self.map_plotter.render(infos=self.infos)

    def close(self):
        if self.sim:
            self.sim.close()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        return [seed]

    def set_phase(self, phase: int):
        self.phase = phase
        gym.logger.info(f"set phase to: {phase}")
        print(f"set phase to: {phase}")

    def get_info(self, reward):
        is_aircraft_at_target = self._is_aircraft_at_target(aircraft_position=self.aircraft_cartesian_position(),
                                                       target_position=self.target_position,
                                                       threshold=GuidanceEnv.MIN_DISTANCE_TO_TARGET_KM)

        is_aircraft_out_of_bounds = self.sim.is_aircraft_out_of_bounds(max_distance_km=self.max_distance_km)

        runway_angle_error_deg = utils.reduce_reflex_angle_deg(self.sim.get_heading_true_deg() - self.runway_angle_deg)
        is_heading_correct = abs(runway_angle_error_deg) < self.runway_angle_threshold_deg

        terminal_state = "other"
        if is_aircraft_at_target:
            terminal_state = "target"
            if is_heading_correct:
                terminal_state = "heading"
        if is_aircraft_out_of_bounds:
            terminal_state = "bounds"

        is_aircraft_altitude_to_low = self.sim.is_aircraft_altitude_to_low(GuidanceEnv.CRASH_HEIGHT_FT)
        aircraft_position = self.aircraft_cartesian_position()

        in_area = self._in_area()
        vertical_track_error = self._calc_vertical_track_error(aircraft_position, self.target_position)
        if in_area:
            cross_track_error = self._calc_cross_track_error(aircraft_position, self.target_position)
        else:
            cross_track_error = self._calc_cross_track_error(aircraft_position, self.localizer_perpendicular_position)

        track_error = abs(cross_track_error) + abs(vertical_track_error)

        diff = self.target_position - aircraft_position
        return {
            "altitude_error": diff.z,
            "aircraft_heading_true_deg": self.sim.get_heading_true_deg(),
            "aircraft_lat_deg": aircraft_position.y,
            "aircraft_long_deg": aircraft_position.x,
            "aircraft_track_angle_deg": self.sim.get_track_angle_deg(),
            "aircraft_v_down_fps": self.sim[prp.v_down_fps],
            "aircraft_v_north_fps": self.sim[prp.v_north_fps],
            "aircraft_v_east_fps": self.sim[prp.v_east_fps],
            "aircraft_z": aircraft_position.z,
            "altitude": self.sim[prp.altitude_sl_ft],
            "altitude_rate_fps": self.sim[prp.altitude_rate_fps],
            "target_lat_deg": self.target_position.y,
            "target_long_deg": self.target_position.x,
            "target_z": self.target_position.z,
            "total_wind_north_fps": self.sim[prp.total_wind_north_fps],
            "total_wind_east_fps": self.sim[prp.total_wind_east_fps],
            "total_wind_down_fps": self.sim[prp.total_wind_down_fps],
            "aircraft_p_radps": self.sim[prp.p_radps],
            "alpha_rad": self.sim[prp.alpha_rad],
            "aircraft_q_radps": self.sim[prp.q_radps],
            "simulation_time_step": self.sim.get_sim_time(),
            "reward": reward,
            "true_airspeed": self.sim.get_true_air_speed(),
            "is_heading_correct": is_heading_correct,
            "terminal_state": terminal_state,
            "is_aircraft_at_target": is_aircraft_at_target,
            "is_aircraft_out_of_bounds": is_aircraft_out_of_bounds,
            "distance_to_target": aircraft_position.distance_to_target(self.target_position),
            "runway_angle": self.runway_angle_deg,
            "runway_angle_error": runway_angle_error_deg,
            "runway_angle_threshold_deg": self.runway_angle_threshold_deg,
            "in_area": in_area,
            "is_on_track": self._is_on_track(),
            "pitch_rad": self.sim[prp.pitch_rad],
            "gamma_deg": math.degrees(self.sim[prp.pitch_rad] - self.sim[prp.alpha_rad]),
            "vertical_track_error": vertical_track_error,
            "cross_track_error": cross_track_error,
            "track_error": track_error,
            "is_aircraft_altitude_to_low": is_aircraft_altitude_to_low
        }

    def aircraft_cartesian_position(self):
        x = self.sim[prp.dist_from_start_lon_m]
        y = self.sim[prp.dist_from_start_lat_m]

        if self.sim[prp.lat_geod_deg] < self.sim[prp.initial_latitude_geod_deg]:
            y = -y
        if self.sim[prp.long_gc_deg] < self.sim[prp.initial_longitude_geoc_deg]:
            x = -x

        z = self.sim[prp.altitude_sl_ft] / 3281 # convert to km

        return CartesianPosition(x / 1000, y / 1000, z, self.sim.get_heading_true_deg(), offset=self.offset)


    # Check if vertical error is correct...
    def _calc_vertical_track_error(self, current_position, target_position):
        diff = current_position - target_position
        return - diff.y * math.sin(math.radians(-self.glide_angle_deg % 360)) + diff.z * math.cos(math.radians(-self.glide_angle_deg % 360))

    def _calc_cross_track_error(self, current_position, target_position):
        # TODO: Turn around as in other places or turn around others...?
        diff = current_position - target_position
        heading = target_position.heading
        return - diff.x * math.sin(math.radians(heading + 90 - self.offset)) + diff.y * math.cos(math.radians(heading - 90 + self.offset))

    def get_true_bearing_to_target_deg(self):
        aircraft_position = self.aircraft_cartesian_position()
        difference_vector = self.target_position - aircraft_position
        return  difference_vector.vector_direction_deg()

    def _get_observation(self) -> np.array:
        aircraft_position = self.aircraft_cartesian_position()

        diff = self.target_position - aircraft_position

        runway_heading_error_deg = utils.reduce_reflex_angle_deg(self.sim.get_heading_true_deg() - self.runway_angle_deg)
        true_airspeed = self.sim.get_true_air_speed()
        # # yaw_rate = self.sim[prp.r_radps]
        turn_rate = self.sim.get_turn_rate()

        altitude_ft = self.sim[prp.altitude_sl_ft]
        altitude_rate_fps = self.sim[prp.altitude_rate_fps]

        in_area = self._in_area()
        vertical_track_error = self._calc_vertical_track_error(aircraft_position, self.target_position)
        if in_area:
            cross_track_error = self._calc_cross_track_error(aircraft_position, self.target_position)
        else:
            cross_track_error = self._calc_cross_track_error(aircraft_position, self.localizer_perpendicular_position)

        # altitude_ft to km ?
        # altitude_rate_fps to km/s ?

        distance_to_target_km = aircraft_position.distance_to_target(self.target_position) / self.max_distance_km

        rest_height_ft = (diff.z * 3281) / GuidanceEnv.MAX_HEIGHT_FT # altitude_ft / GuidanceEnv.MAX_HEIGHT_FT

        return np.array([
            in_area,
            cross_track_error,
            vertical_track_error,
            abs(rest_height_ft),
            altitude_rate_fps / GuidanceEnv.MAX_HEIGHT_FT,
            distance_to_target_km,
            true_airspeed / 1000,
            turn_rate,
            diff.x,
            diff.y,
            diff.z,
            math.sin(math.radians(self.sim.get_heading_true_deg())),
            math.cos(math.radians(self.sim.get_heading_true_deg())),
            math.sin(math.radians(runway_heading_error_deg)),
            math.cos(math.radians(runway_heading_error_deg))
        ], dtype=np.float32)

    def _is_done(self) -> bool:
        is_terminal_step = self.steps_left < 1

        # is_aircraft_out_of_bounds = self.sim.is_aircraft_out_of_bounds(max_distance_km=self.max_distance_km)
        is_aircraft_altitude_to_low = self.sim.is_aircraft_altitude_to_low(self.to_low_height) # convert to ft

        runway_heading_error_deg = utils.reduce_reflex_angle_deg(self.sim.get_heading_true_deg() - self.runway_angle_deg)

        is_aircraft_at_target = self._is_aircraft_at_target(aircraft_position=self.aircraft_cartesian_position(),
                                                       target_position=self.target_position,
                                                       threshold=GuidanceEnv.MIN_DISTANCE_TO_TARGET_KM) # and is_heading_correct

        is_done = is_terminal_step or is_aircraft_altitude_to_low  or is_aircraft_at_target # or is_aircraft_out_of_bounds

        if(is_done):
            print("is_terminal_step", is_terminal_step)
            print("is_aircraft_altitude_to_low", is_aircraft_altitude_to_low)
            print("is_aircraft_at_target", is_aircraft_at_target)
            print("obs", self._get_observation())
            print("self.target_position.z", self.target_position.z)

        return is_done

    def _in_area(self):
        relative_bearing_to_aircraft_deg = utils.reduce_reflex_angle_deg(self.target_position.direction_to_target_deg(self.aircraft_cartesian_position()) - self.runway_angle_deg) % 360
        in_area = False
        if 90 <= relative_bearing_to_aircraft_deg <= 270:
            in_area = True
        return in_area

    def _is_on_track(self):
        aircraft_position = self.aircraft_cartesian_position()
        runway_heading_error_deg = utils.reduce_reflex_angle_deg(self.sim.get_heading_true_deg() - self.runway_angle_deg)
        current_distance_km = aircraft_position.distance_to_target(self.target_position)

        cross_track_error = self._calc_cross_track_error(aircraft_position, self.target_position)
        vertical_track_error = self._calc_vertical_track_error(aircraft_position, self.target_position)

        track_error = abs(cross_track_error) + abs(vertical_track_error)

        track_medium_error = (abs(self.last_track_error) + abs(track_error)) / 2

        if abs(track_medium_error) < 0.15 and current_distance_km < self.last_distance_km[-1] and abs(runway_heading_error_deg) < 90:
            return True
        return False

    # continous reward
    def _reward(self):
        in_area = self._in_area()
        aircraft_position = self.aircraft_cartesian_position()
        diff_position = self.target_position - aircraft_position

        if self.sim.is_aircraft_altitude_to_low(self.to_low_height):
            distance_error = aircraft_position.distance_to_target(self.target_position) * 6
            print("is_aircraft_altitude_to_low: distance_error", distance_error)
            return - np.clip(abs(distance_error), 0, 10)

        runway_heading_error_deg = utils.reduce_reflex_angle_deg(self.sim.get_heading_true_deg() - self.runway_angle_deg)
        is_heading_correct = abs(runway_heading_error_deg) < self.runway_angle_threshold_deg


        if self._is_aircraft_at_target(aircraft_position=self.aircraft_cartesian_position(),
                                                       target_position=self.target_position,
                                                       threshold=GuidanceEnv.MIN_DISTANCE_TO_TARGET_KM): # and is_heading_correct
            heading_bonus = 1 - np.interp(abs(runway_heading_error_deg), [0, self.runway_angle_threshold_deg], [0, 1])
            reward = 9 + heading_bonus

            print("at target, positive reward: ", reward)

            # reward for height
            return reward

        if self._is_aircraft_at_target(aircraft_position=self.aircraft_cartesian_position(),
                                       target_position=self.target_position,
                                       threshold=GuidanceEnv.MIN_DISTANCE_TO_TARGET_KM) and not in_area:
            return -10



        current_distance_km = aircraft_position.distance_to_target(self.target_position)
        reward_heading = 0
        reward_track = 0
        penalty_area_2 = 0

        if in_area:
            cross_track_error = self._calc_cross_track_error(aircraft_position, self.target_position)
            vertical_track_error = self._calc_vertical_track_error(aircraft_position, self.target_position)

            track_error = abs(cross_track_error) + abs(vertical_track_error)

            track_medium_error = (abs(self.last_track_error) + abs(track_error)) / 2
            diff_track = abs(self.last_track_error - track_error)

            diff_headings = abs(math.radians(utils.reduce_reflex_angle_deg(runway_heading_error_deg - self.last_runway_heading_error_deg[-1])) / math.pi)
            if abs(track_medium_error) < 0.15 and current_distance_km < self.last_distance_km[-1] and abs(runway_heading_error_deg) < 90:
                if abs(runway_heading_error_deg) < abs(self.last_runway_heading_error_deg[-1]):
                    reward_heading = diff_headings
                else:
                    reward_heading = -diff_headings
                reward_track = 1
            else:
                reward_track = -diff_track * 2

            self.last_distance_km.append(current_distance_km)
            self.last_runway_heading_error_deg.append(runway_heading_error_deg)
            self.last_track_error = track_error
        else:
            cross_track_error = self._calc_cross_track_error(aircraft_position, self.localizer_perpendicular_position)
            # cross_track_medium_error = (abs(self.last_cross_track_error_perpendicular) + abs(cross_track_error)) / 2
            self.last_track_error_perpendicular = cross_track_error
            track_error = cross_track_error
            penalty_area_2 = -2

        # print("in_area", in_area, "track error", track_error, "cross", cross_track_error, "vertical", vertical_track_error)
        clipped = np.clip(np.exp(abs(track_error)), math.exp(0), math.exp(self.max_distance_km))
        reward_track_shaped = - np.interp(clipped,
                                          [math.exp(0), math.exp(self.max_distance_km)],
                                          [0, 1])

        # print("reward_cross_shaped", reward_cross_shaped)

        reward_altitude_shaped = - abs(diff_position.z) / 10

        reward_shaped = reward_track_shaped + reward_altitude_shaped
        reward_sparse = reward_track + penalty_area_2 + reward_heading

        return reward_shaped + reward_sparse

    def calc_distance_from_point_to_line(self, line_point1: CartesianPosition, line_point2: CartesianPosition, point: CartesianPosition):
        x1 = line_point1.x
        y1 = line_point1.y

        x2 = line_point2.x
        y2 = line_point2.y

        x0 = point.x
        y0 = point.y

        distance = abs((x2 - x1)*(y1-y0) - (x1-x0)*(y2-y1)) / math.sqrt(math.pow((x2-x1), 2) + math.pow((y2-y1), 2))
        return distance

    def _is_aircraft_at_target(self, aircraft_position: CartesianPosition,
                               target_position: CartesianPosition,
                               threshold: float):
        diff_position = self.target_position - aircraft_position
        return aircraft_position.distance_to_target(target_position) <= threshold and abs(diff_position.z) <= 30 / 1000


class GuidanceEnvContinuos(GuidanceEnv):
    action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
    continuous = True