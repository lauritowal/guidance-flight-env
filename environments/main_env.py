import math

import gym
import numpy as np
from gym.utils import seeding
from pid.pid_controller import PidController
from services.plotter import Plotter
from simulation import Simulation
from utils import utils
from aircraft import cessna172P
import properties as prp

from utils.element_3d import Element3d
from abc import ABC, abstractmethod


class MainEnv(gym.Env, ABC):
    episode_counter = 0
    pid_controller = PidController()
    metadata = {
        "render.modes": ["rgb-array", "jsbsim"]
    }

    def __init__(self,
                 jsbsim_path,
                 aircraft=cessna172P,
                 agent_interaction_freq=5,
                 target_radius=100 / 1000,
                 max_distance_km=4,
                 max_target_distance_km=2,
                 max_episode_time_s=60 * 5,
                 phase=0,
                 glide_angle_deg=4,
                 seed=None,
                 min_distance_to_target_km=100 / 1000,
                 min_height_for_flare_ft: float = 20.0,
                 max_height_ft=3500,
                 height_threshold_ft=30,
                 crash_height_ft=6,
                 html_dict="./htmls/test.html",
                 evaluation=False,
                 jsbsim_dt_hz: int = 60):

        self.min_runway_angle_threshold_deg = 5
        self.runway_angle_threshold_deg = 10
        self.min_height_for_flare_ft = min_height_for_flare_ft
        self.max_target_distance_km = max_target_distance_km
        self.max_height_ft = max_height_ft
        self.height_threshold_ft = height_threshold_ft
        self.height_threshold_m = height_threshold_ft / 3.281
        self.crash_height_ft = crash_height_ft
        self.jsbsim_dt_hz = jsbsim_dt_hz
        self.min_height_for_flare_m = self.min_height_for_flare_ft / 3.281
        self.min_distance_to_target_km = min_distance_to_target_km
        self.target_radius_km = target_radius
        self.jsbsim_path = jsbsim_path
        self.aircraft = aircraft
        self.html_dict = html_dict
        self.max_distance_km = max_distance_km
        self.max_target_distance_km = max_target_distance_km
        self.max_episode_time_s = max_episode_time_s
        self.spawn_target_distance_km = 0.5
        self.glide_angle_deg = glide_angle_deg

        if agent_interaction_freq > self.jsbsim_dt_hz:
            raise ValueError('agent interaction frequency must be less than '
                             'or equal to JSBSim integration frequency of '
                             f'{self.jsbsim_dt_hz} Hz.')
        self.sim_steps_per_agent_step: int = self.jsbsim_dt_hz // agent_interaction_freq
        self.sim_steps = self.sim_steps_per_agent_step
        self.sim: Simulation = None

        self.episode_steps = math.ceil(self.max_episode_time_s * agent_interaction_freq)
        self.steps_left = self.episode_steps

        self.infos = []
        self.rewards = []
        self.done = False
        self.plotter = None

        self.target_position = None
        self.perpendicular_point: Element3d = None

        self.to_low_height = None
        self.last_state = None

        self.np_random = None

        if seed is not None:
            self.seed(seed)

        self.runway_angle_deg = None

        self.phase = phase

        self.last_track_error = 0
        self.last_track_error_perpendicular = 0

        self.last_distance_km = []
        self.last_runway_heading_error_deg = []

    @abstractmethod
    def setup_episode(self):
        ...

    @abstractmethod
    def get_info(self, reward):
        ...

    @abstractmethod
    def _reward(self):
        ...

    @abstractmethod
    def _is_done(self):
        ...

    @abstractmethod
    def _get_observation(self):
        ...



    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        return [seed]

    def reset(self):
        initial_conditions = self._get_sim_initial_conditions()

        if self.sim:
            self.sim.reinitialise(init_conditions=initial_conditions)
        else:
            self.sim = self._init_new_sim(self.jsbsim_dt_hz, self.aircraft, initial_conditions)

        self.steps_left = self.episode_steps

        self.runway_angle_deg = 0

        if self.max_distance_km is None:
            self.max_distance_km = self.sim.calculate_max_distance_km(self.max_episode_time_s)

        self.sim.start_engines()  # start engines for testing the algorithm in simplest form
        self.sim.set_throttle_mixture_controls(0.8, 0.7)

        self.sim.start_engines()  # start engines for testing the algorithm in simplest form
        self.sim.set_throttle_mixture_controls(0.8, 0.7)
        self.sim.raise_landing_gear()

        self.target_position = self._generate_random_target_position()
        aircraft_position = self._aircraft_position()
        self.last_distance_km.append(aircraft_position.distance_to_target(self.target_position))
        self.to_low_height = (self.target_position.z - 10 / 1000) * 3281
        self.perpendicular_point = self._create_point_perpendicular_to_runway()
        runway_heading_error_deg = utils.normalize_angle_deg(self.sim.get_heading_true_deg() - self.runway_angle_deg)
        self.last_runway_heading_error_deg.append(runway_heading_error_deg)

        self.setup_episode()

        self.plotter = Plotter(target=self.target_position,
                               glide_angle_deg=self.glide_angle_deg,
                               aircraft_initial_position=Element3d(0, 0, 0),
                               localizer_perpendicular_position=self.perpendicular_point,
                               target_radius_km=self.target_radius_km,
                               target_spawn_area_radius_km=self.spawn_target_distance_km,
                               bounds_radius_km=self.max_distance_km,
                               runway_angle=self.runway_angle_deg)

        self.infos = []
        info = self.get_info(0)
        self.infos.append(info)

        self.done = False

        return self._get_observation()

    def step(self, action: np.ndarray):
        if not (action.shape == self.action_space.shape):
            raise ValueError('mismatch between action and action space size')

        x = action[0]
        y = action[1]
        heading_deg = math.degrees(math.atan2(y, x))

        action_target_heading_deg = heading_deg % 360
        self.sim[prp.elevator_cmd] = self.pid_controller.elevator_hold(pitch_angle_reference=math.radians(0),
                                                                       pitch_angle_current=self.sim[prp.pitch_rad],
                                                                       pitch_angle_rate_current=self.sim[prp.q_radps])

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

        self.episode_counter += 1
        return state, reward, self.done, info

    def set_phase(self, phase: int):
        self.phase = phase

    def render(self, mode='rgb_array') -> np.array:
        if mode == 'rgb_array':
            return self.plotter.render_rgb_array(infos=self.infos)

    def _init_new_sim(self, dt, aircraft, initial_conditions):
        return Simulation(sim_frequency_hz=dt,
                          aircraft=aircraft,
                          init_conditions=initial_conditions,
                          jsbsim_path=self.jsbsim_path)

    def _aircraft_position(self):
        x = self.sim[prp.dist_from_start_lon_m]
        y = self.sim[prp.dist_from_start_lat_m]

        if self.sim[prp.lat_geod_deg] < self.sim[prp.initial_latitude_geod_deg]:
            y = -y
        if self.sim[prp.long_gc_deg] < self.sim[prp.initial_longitude_geoc_deg]:
            x = -x

        z = self.sim[prp.altitude_sl_ft] / 3281  # convert to km

        return Element3d(x / 1000, y / 1000, z, self.sim.get_heading_true_deg())

    def _generate_random_target_position(self) -> (Element3d, float):
        start_distance = 600 / 1000

        def random_sign():
            if self.np_random.random() < 0.5:
                return 1
            return -1

        x = self.np_random.uniform(0, self.spawn_target_distance_km) * random_sign()
        y = self.np_random.uniform(start_distance, self.spawn_target_distance_km) * random_sign()
        z = self.np_random.uniform(0.2,
                                   (self.sim[prp.initial_altitude_ft] / 3281) / 2) + self.min_height_for_flare_m / 1000

        return Element3d(x, y, z, heading=self.runway_angle_deg)

    def _get_sim_initial_conditions(self):
        return {
            prp.initial_altitude_ft: self.np_random.uniform(self.max_height_ft - self.max_height_ft / 3,
                                                            self.max_height_ft),
            prp.initial_terrain_altitude_ft: 0.00000001,
            prp.initial_longitude_geoc_deg: self.np_random.uniform(-160, 160),
            # decreased range to avoid problems close to eqautor at -180 / 180
            prp.initial_latitude_geod_deg: self.np_random.uniform(-70, 70),
            # decreased range to avoid problems close to poles at -90 / 90
            prp.initial_u_fps: self.aircraft.get_cruise_speed_fps(),
            prp.initial_v_fps: 0,
            prp.initial_w_fps: 0,
            prp.initial_p_radps: 0,
            prp.initial_q_radps: 0,
            prp.initial_r_radps: 0,
            prp.initial_roc_fpm: 0,
            prp.initial_heading_deg: self.np_random.uniform(0, 360),
        }