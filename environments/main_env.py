import math
from array import array
from typing import List

import gym
import numpy as np
from gym.utils import seeding
from pid.pid_controller import PidController
from services.plotter import Plotter
from simulation import Simulation
from utils import utils
from aircraft import cessna172P, Aircraft
import properties as prp
from dataclasses import dataclass, field

from utils.object_3d import Object3D
from abc import ABC, abstractmethod


@dataclass
class MainEnv(gym.Env, ABC):
    infos = []
    last_distance_km = []
    pid_controller = PidController()
    metadata = {"render.modes": ["rgb-array"]}
    jsbsim_path: str
    episode_counter: int = 0
    max_distance_km: float = math.inf
    aircraft: Aircraft = cessna172P
    agent_interaction_freq: int = 5
    target_radius_km: float = 100 / 1000
    max_target_distance_km: int = 2
    max_episode_time_s: int = 60 * 5
    phase: int = 0
    glide_angle_deg: float = 4
    seed_value: int = None
    min_distance_to_target_km: float = 100 / 1000
    min_height_for_flare_ft: float = 20.0
    min_height_for_flare_m: float = min_height_for_flare_ft / 3.281
    max_starting_height_ft: float = 3500
    max_starting_height_ft: float = max_starting_height_ft
    jsbsim_dt_hz: int = 60
    spawn_target_distance_km: float = 0.5
    sim: Simulation = None
    done: bool = False
    plotter: Plotter = None
    target_position: Object3D = None
    crash_height_ft: float = None
    last_state: np.array = None
    np_random: float = None
    steps_left: int = 0
    evaluation: bool = False

    def __post_init__(self):
        self.infos = []
        self.pid_controller = PidController()
        self.last_distance_km = []
        self.last_runway_heading_error_deg = []

        if self.seed_value is not None:
            self.seed(self.seed_value)

        if self.agent_interaction_freq > self.jsbsim_dt_hz:
            raise ValueError('agent interaction frequency must be less than '
                             'or equal to JSBSim integration frequency of '
                             f'{self.jsbsim_dt_hz} Hz.')

        self.sim_steps_per_agent_step: int = self.jsbsim_dt_hz // self.agent_interaction_freq
        self.sim_steps = self.sim_steps_per_agent_step
        self.episode_steps = math.ceil(self.max_episode_time_s * self.agent_interaction_freq)
        self.steps_left = self.episode_steps


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
        self.max_distance_km = self.sim.calculate_max_distance_km(self.max_episode_time_s)
        self.sim.start_engines()
        self.sim.set_throttle_mixture_controls(0.8, 0.7)
        self.sim.raise_landing_gear()

        self.target_position = self._generate_random_target_position()
        aircraft_position = self._aircraft_position()
        self.last_distance_km.append(aircraft_position.distance_to_target(self.target_position))
        self.crash_height_ft = (self.target_position.z - 10 / 1000) * 3281
        runway_heading_error_deg = utils.normalize_angle_deg(self.sim.get_heading_true_deg() - self.runway_angle_deg)
        self.last_runway_heading_error_deg.append(runway_heading_error_deg)

        self.setup_episode()

        self.plotter = Plotter(target=self.target_position,
                               glide_angle_deg=self.glide_angle_deg,
                               aircraft_initial_position=Object3D(0, 0, 0),
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

        z = self.sim[prp.altitude_sl_ft] / 3281

        return Object3D(x / 1000, y / 1000, z, self.sim.get_heading_true_deg())

    def _generate_random_target_position(self) -> (Object3D, float):
        start_distance = 600 / 1000

        def random_sign():
            if self.np_random.random() < 0.5:
                return 1
            return -1

        x = self.np_random.uniform(0, self.spawn_target_distance_km) * random_sign()
        y = self.np_random.uniform(start_distance, self.spawn_target_distance_km) * random_sign()
        z = self.np_random.uniform(0.2,
                                   (self.sim[prp.initial_altitude_ft] / 3281) / 2) + self.min_height_for_flare_m / 1000

        return Object3D(x, y, z, heading=self.runway_angle_deg)

    def _get_sim_initial_conditions(self):
        return {
            prp.initial_altitude_ft: self.np_random.uniform(self.max_starting_height_ft - self.max_starting_height_ft / 3,
                                                            self.max_starting_height_ft),
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