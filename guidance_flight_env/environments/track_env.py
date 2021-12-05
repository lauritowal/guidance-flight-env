import math
import gym
from gym.utils import seeding
import numpy as np
from dataclasses import dataclass
from guidance_flight_env import properties as prp
from guidance_flight_env.pid.pid_controller import PidController
from guidance_flight_env.services.plotter import Plotter
from guidance_flight_env.simulation import Simulation
from guidance_flight_env.utils import utils
from guidance_flight_env.aircraft import cessna172P, Aircraft
from guidance_flight_env import properties as prp
from guidance_flight_env.utils.object_3d import Object3D


@dataclass
class TrackEnvWind(gym.Env):
    action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(19,), dtype=np.float32)
    runway_angle_threshold_deg = 10
    last_track_error = 0
    last_track_error_perpendicular = 0
    infos = []
    last_distance_km = []
    pid_controller = PidController()
    metadata = {"render.modes": ["rgb-array"]}
    jsbsim_path: str
    episode_counter: int = 0
    aircraft: Aircraft = cessna172P
    agent_interaction_freq: int = 5
    target_radius_km: float = 100 / 1000
    max_distance_km: int = 2
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

        if self.max_distance_km == None:
            self.max_distance_km = self.sim.calculate_max_distance_km(self.max_episode_time_s)


    def setup_episode(self):
        # Increasing difficulty for incrementing phase
        if self.phase == 0:
            self.spawn_target_distance_km = 0.5
            self.sim[prp.wind_east_fps] = 0
        elif self.phase == 1:
            self.spawn_target_distance_km = 1
            self.sim[prp.wind_east_fps] = self.np_random.uniform(-10, 10)
        elif self.phase == 2:
            self.spawn_target_distance_km = 1.5
            self.sim[prp.wind_east_fps] = self.np_random.uniform(-20, 20)
        elif self.phase == 3:
            self.spawn_target_distance_km = 2
            self.sim[prp.wind_east_fps] = self.np_random.uniform(-40, 40)
        elif self.phase == 4:
            self.spawn_target_distance_km = self.max_distance_km
            self.sim[prp.wind_east_fps] = self.np_random.uniform(-55, 55)

        self.last_track_error = 0
        self.last_track_error_perpendicular = 0
        self.localizer_position = self._create_localizer()
        self.localizer_perpendicular_position = self._create_perpendicular_localizer()
        self.sim.raise_landing_gear()
        self.sim.stop_engines()

    def get_info(self, reward):
        aircraft_position = self._aircraft_position()

        runway_angle_error_deg = utils.normalize_angle_deg(self.sim.get_heading_true_deg() - self.runway_angle_deg)

        is_aircraft_out_of_bounds = self.sim.is_aircraft_out_of_bounds(max_distance_km=self.max_distance_km)
        is_heading_correct = abs(runway_angle_error_deg) < self.runway_angle_threshold_deg
        is_aircraft_altitude_to_low = self.sim.is_aircraft_altitude_to_low(self.crash_height_ft)
        is_in_area = self._is_in_area()
        is_aircraft_at_target = self.sim.is_aircraft_at_target(self.target_position,
                                                               aircraft_position=aircraft_position,
                                                               target_position=self.target_position,
                                                               threshold=self.min_distance_to_target_km)

        if is_in_area:
            cross_track_error = self._calc_cross_track_error(aircraft_position, self.target_position)
        else:
            cross_track_error = self._calc_cross_track_error(aircraft_position, self.localizer_perpendicular_position)
        vertical_track_error = self._calc_vertical_track_error(aircraft_position, self.target_position)
        track_error = abs(cross_track_error) + abs(vertical_track_error)

        diff = self.target_position - aircraft_position
        return {
            "altitude_error": diff.z,
            "aircraft_heading_true_deg": self.sim.get_heading_true_deg(),
            "aircraft_x": aircraft_position.y,
            "aircraft_y": aircraft_position.x,
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
            "drift_deg": self._get_drift_deg(),
            "simulation_time_step": self.sim.get_sim_time(),
            "reward": reward,
            "is_heading_correct": is_heading_correct,
            "is_aircraft_at_target": is_aircraft_at_target,
            "is_aircraft_out_of_bounds": is_aircraft_out_of_bounds,
            "distance_to_target": aircraft_position.distance_to_target(self.target_position),
            "runway_angle": self.runway_angle_deg,
            "runway_angle_error": runway_angle_error_deg,
            "runway_angle_threshold_deg": self.runway_angle_threshold_deg,
            "in_area": is_in_area,
            "is_on_track": self._is_on_track(),
            "pitch_rad": self.sim[prp.pitch_rad],
            "gamma_deg": math.degrees(self.sim[prp.pitch_rad] - self.sim[prp.alpha_rad]),
            "vertical_track_error": vertical_track_error,
            "cross_track_error": cross_track_error,
            "track_error": track_error,
            "is_aircraft_altitude_to_low": is_aircraft_altitude_to_low
        }

    def _get_reference_heading_deg(self, action: np.ndarray):
        x = action[0]
        y = action[1]
        heading_deg = math.degrees(math.atan2(y, x))
        return heading_deg % 360

    def _create_localizer(self):
        distance_km = -1
        heading = self.runway_angle_deg
        # rotate from  N(90°);E(0°) to N(0°);E(90°)
        x = self.target_position.x + distance_km * math.cos(math.radians((heading - 90) % 360))
        y = self.target_position.y + distance_km * math.sin(math.radians((heading + 90) % 360))
        z = self.target_position.z

        localizer = Object3D(x, y, z, heading=self.runway_angle_deg)
        return localizer

    def _create_perpendicular_localizer(self):
        heading = self.runway_angle_deg + 90

        distance_km = -1
        # rotate from  N(90°);E(0°) to N(0°);E(90°)
        x = self.localizer_position.x + distance_km * math.cos(math.radians((heading - 90) % 360))
        y = self.localizer_position.y + distance_km * math.sin(math.radians((heading + 90) % 360))

        localizer = Object3D(x, y, heading=heading)
        return localizer

    def _calc_vertical_track_error(self, current_position, target_position):
        diff = current_position - target_position
        return - diff.y * math.sin(math.radians(-self.glide_angle_deg % 360)) + diff.z * math.cos(math.radians(-self.glide_angle_deg % 360))

    def _calc_cross_track_error(self, current_position, target_position):
        diff = current_position - target_position
        heading = target_position.heading
        return - diff.x * math.sin(math.radians(heading + 90)) + diff.y * math.cos(math.radians(heading - 90))

    def _get_drift_deg(self):
        return utils.normalize_angle_deg(self.sim.get_heading_true_deg() - self.sim.get_track_angle_deg()) % 360

    def _get_observation(self) -> np.array:
        aircraft_position = self._aircraft_position()

        diff = self.target_position - aircraft_position

        runway_heading_error_deg = utils.normalize_angle_deg(self.sim.get_heading_true_deg() - self.runway_angle_deg)
        true_airspeed = self.sim.get_true_air_speed()
        turn_rate = self.sim.get_turn_rate()
        altitude_rate_fps = self.sim[prp.altitude_rate_fps]

        in_area = self._is_in_area()
        if in_area:
            cross_track_error = self._calc_cross_track_error(aircraft_position, self.target_position)
        else:
            cross_track_error = self._calc_cross_track_error(aircraft_position, self.localizer_perpendicular_position)
        vertical_track_error = self._calc_vertical_track_error(aircraft_position, self.target_position)

        distance_to_target_km = aircraft_position.distance_to_target(self.target_position) / self.max_distance_km
        rest_height_ft = (diff.z * 3281) / self.max_starting_height_ft

        drift_deg = self._get_drift_deg()

        return np.array([
            in_area,
            cross_track_error,
            vertical_track_error,
            self.sim[prp.total_wind_north_fps],
            self.sim[prp.total_wind_east_fps],
            math.sin(math.radians(drift_deg)),
            math.cos(math.radians(drift_deg)),
            abs(rest_height_ft),
            altitude_rate_fps / self.max_starting_height_ft,
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

    def _reward(self):
        in_area = self._is_in_area()
        aircraft_position = self._aircraft_position()

        if self.sim.is_aircraft_altitude_to_low(self.crash_height_ft):
            distance_error = aircraft_position.distance_to_target(self.target_position) * 6
            print("is_aircraft_altitude_to_low: distance_error", distance_error)
            return - np.clip(abs(distance_error), 0, 10)

        runway_heading_error_deg = utils.normalize_angle_deg(self.sim.get_heading_true_deg() - self.runway_angle_deg)

        if self.sim.is_aircraft_at_target(self.target_position, aircraft_position=self._aircraft_position(),
                                  target_position=self.target_position,
                                  threshold=self.min_distance_to_target_km): # and is_heading_correct
            heading_bonus = 1 - np.interp(abs(runway_heading_error_deg), [0, self.runway_angle_threshold_deg], [0, 1])
            reward = 9 + heading_bonus

            print("at target, positive reward: ", reward)
            return reward

        if self.sim.is_aircraft_at_target(self.target_position, aircraft_position=self._aircraft_position(),
                                  target_position=self.target_position,
                                  threshold=self.min_distance_to_target_km) and not in_area:
            return -10

        current_distance_km = aircraft_position.distance_to_target(self.target_position)
        reward_heading = 0
        reward_track = 0
        penalty_area_2 = 0

        if in_area:
            cross_track_error = self._calc_cross_track_error(aircraft_position, self.target_position)
            vertical_track_error = self._calc_vertical_track_error(aircraft_position, self.target_position)

            track_error = abs(cross_track_error) + abs(vertical_track_error)
            diff_track = abs(self.last_track_error - track_error)

            diff_headings = abs(math.radians(
                utils.normalize_angle_deg(runway_heading_error_deg - self.last_runway_heading_error_deg[-1])) / math.pi)
            if self._is_on_track():
                if abs(runway_heading_error_deg) < abs(self.last_runway_heading_error_deg[-1]):
                    reward_heading = diff_headings
                else:
                    reward_heading = -diff_headings
                reward_track = 1 # Maybe diff_track * 2 instead of 1
            else:
                reward_track = -diff_track * 2

            self.last_distance_km.append(current_distance_km)
            self.last_runway_heading_error_deg.append(runway_heading_error_deg)
            self.last_track_error = track_error
        else:
            cross_track_error = self._calc_cross_track_error(aircraft_position, self.localizer_perpendicular_position)
            self.last_track_error_perpendicular = cross_track_error
            track_error = cross_track_error
            penalty_area_2 = -2

        # Maybe replace exp again with linear
        clipped = np.clip(np.exp(abs(track_error)), math.exp(0), math.exp(self.max_distance_km))
        reward_track_shaped = - np.interp(clipped,
                                          [math.exp(0), math.exp(self.max_distance_km)],
                                          [0, 1])

        reward_shaped = reward_track_shaped
        reward_sparse = reward_track + penalty_area_2 + reward_heading

        return reward_shaped + reward_sparse

    def _is_done(self) -> bool:
        is_terminal_step = self.steps_left < 1
        is_aircraft_altitude_to_low = self.sim.is_aircraft_altitude_to_low(self.crash_height_ft)
        is_aircraft_at_target = self.sim.is_aircraft_at_target(self.target_position, aircraft_position=self._aircraft_position(),
                                                            target_position=self.target_position,
                                                            threshold=self.min_distance_to_target_km)

        is_done = is_terminal_step or is_aircraft_altitude_to_low  or is_aircraft_at_target

        if(is_done):
            print("is_terminal_step", is_terminal_step)
            print("is_aircraft_altitude_to_low", is_aircraft_altitude_to_low)
            print("is_aircraft_at_target", is_aircraft_at_target)
            print("obs", self._get_observation())
            print("self.target_position.z", self.target_position.z)

        return is_done

    def _is_in_area(self):
        difference_vector = self.target_position - self._aircraft_position()
        relative_bearing_to_aircraft_deg = utils.normalize_angle_deg(difference_vector.direction_2d_deg() - self.runway_angle_deg) % 360

        in_area = False
        if 90 <= relative_bearing_to_aircraft_deg <= 270:
            in_area = True
        return in_area

    def _is_on_track(self):
        aircraft_position = self._aircraft_position()
        runway_heading_error_deg = utils.normalize_angle_deg(self.sim.get_heading_true_deg() - self.runway_angle_deg)
        current_distance_km = aircraft_position.distance_to_target(self.target_position)

        cross_track_error = self._calc_cross_track_error(aircraft_position, self.target_position)
        vertical_track_error = self._calc_vertical_track_error(aircraft_position, self.target_position)

        track_error = abs(cross_track_error) + abs(vertical_track_error)

        track_medium_error = (abs(self.last_track_error) + abs(track_error)) / 2

        if abs(track_medium_error) < 0.15 and current_distance_km < self.last_distance_km[-1] and abs(runway_heading_error_deg) < 90:
            return True
        return False



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

        reference_heading_deg = self._get_reference_heading_deg(action)

        self.sim[prp.elevator_cmd] = self.pid_controller.elevator_hold(pitch_angle_reference=math.radians(0),
                                                                       pitch_angle_current=self.sim[prp.pitch_rad],
                                                                       pitch_angle_rate_current=self.sim[prp.q_radps])

        self.sim[prp.aileron_cmd] = self.pid_controller.heading_hold(
            heading_reference_deg=reference_heading_deg,
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
            return self.plotter.render_rgb_array_simple(infos=self.infos)

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


class TrackEnvNoWind(TrackEnvWind):
    action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
    observation_space: gym.Space = gym.spaces.Box(-np.inf, np.inf, shape=(15,), dtype=np.float32)

    def setup_episode(self):
        # no wind in this environment
        self.sim[prp.wind_east_fps] = 0

        # Increasing difficulty for incrementing phase
        if self.phase == 0:
            self.spawn_target_distance_km = 0.5
        elif self.phase == 1:
            self.spawn_target_distance_km = 1
        elif self.phase == 2:
            self.spawn_target_distance_km = 1.5
        elif self.phase == 3:
            self.spawn_target_distance_km = 2
        elif self.phase == 4:
            self.spawn_target_distance_km = self.max_distance_km

        self.sim.raise_landing_gear()
        self.sim.stop_engines()

    def _get_observation(self) -> np.array:
        aircraft_position = self._aircraft_position()

        diff = self.target_position - aircraft_position

        runway_heading_error_deg = utils.normalize_angle_deg(self.sim.get_heading_true_deg() - self.runway_angle_deg)
        true_airspeed = self.sim.get_true_air_speed()
        turn_rate = self.sim.get_turn_rate()
        altitude_rate_fps = self.sim[prp.altitude_rate_fps]

        in_area = self._is_in_area()
        if in_area:
            cross_track_error = self._calc_cross_track_error(aircraft_position, self.target_position)
        else:
            cross_track_error = self._calc_cross_track_error(aircraft_position, self.localizer_perpendicular_position)
        vertical_track_error = self._calc_vertical_track_error(aircraft_position, self.target_position)

        distance_to_target_km = aircraft_position.distance_to_target(self.target_position) / self.max_distance_km
        rest_height_ft = (diff.z * 3281) / self.max_starting_height_ft

        return np.array([
            in_area,
            cross_track_error,
            vertical_track_error,
            abs(rest_height_ft),
            altitude_rate_fps / self.max_starting_height_ft,
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