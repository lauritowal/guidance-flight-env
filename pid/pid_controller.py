import time
import properties as prp
from gym_jsbsim import aircraft
from simulation import Simulation
import numpy as np

import os
dirname = os.path.dirname(__file__)

# TODO: Add altitude hold controller?
class PidController(object):
    MAX_TURN_RATE_DEG = 3.0
    error_gammas_rad = 0

    def elevator_hold(self, pitch_angle_reference, pitch_angle_current, pitch_angle_rate_current):
        error_pitch_angle = pitch_angle_reference - pitch_angle_current
        derivative = pitch_angle_rate_current
        p_e = -8  # K_u=-10 T_u = 24
        d_e = -0.5
        elevatorCommand = np.clip(error_pitch_angle * p_e - derivative * d_e, -1, 1)
        return elevatorCommand

    def bank_angle_hold(self, roll_angle_reference, roll_angle_current, roll_angle_rate) -> float:
        roll_angle_reference = np.clip(roll_angle_reference, np.deg2rad(-20), np.deg2rad(20))
        diff_rollAngle = roll_angle_reference - roll_angle_current

        p = 5
        d = 0.1

        return np.clip(diff_rollAngle * p - roll_angle_rate * d, -1.0, 1.0) * 1.0

    def heading_hold(self, heading_reference_deg: float, heading_current_deg: float, roll_angle_current_rad: float, roll_angle_rate: float,
                     true_air_speed: float) -> float:

        difference_heading: float = heading_reference_deg - heading_current_deg
        difference_heading = difference_heading % 360

        if difference_heading >= 180:
            difference_heading = difference_heading - 360
        '''
        keep turn_rate at full max / min value 
        If p_h is too big --> higher turn_rate but can overshoot target
        if p_h is too low --> lower  turn_rate but can take too long to turn
        '''
        p_h = 0.09
        turn_rate: float = difference_heading * p_h
        turn_rate = np.clip(turn_rate, -PidController.MAX_TURN_RATE_DEG, PidController.MAX_TURN_RATE_DEG) * 1.0 # Standard turn rate needed for emergency ?

        roll_angle_command: float = turn_rate * true_air_speed / 9.81


        return self.bank_angle_hold(np.deg2rad(roll_angle_command),
                               roll_angle_current_rad,
                               roll_angle_rate)

    def flight_path_angle_hold(self, gamma_reference_rad, pitch_rad, alpha_rad, q_radps, roll_rad, r_radps):
        p = -0.2
        i = -0.3

        gamma_deg = np.degrees(pitch_rad - alpha_rad)
        error_gamma_deg = (np.degrees(gamma_reference_rad) - gamma_deg) * 0.4

        error_gamma_rad = np.radians(error_gamma_deg)
        # error_gamma_rad = np.clip(np.radians(error_gamma_deg), -1, 1)
        error = error_gamma_rad - (q_radps * np.cos(roll_rad) - r_radps * np.sin(roll_rad))
        self.error_gammas_rad += error

        return np.clip(p * error + i * self.error_gammas_rad, -1, 1)

    def vertical_speed_hold(self, speed_reference, ground_speed, pitch_rad, alpha_rad, q_radps, roll_rad, r_radps):
        vs = speed_reference / ground_speed
        return self.flight_path_angle_hold(gamma_reference_rad=vs,
                                           pitch_rad=pitch_rad,
                                           alpha_rad=alpha_rad,
                                           q_radps=q_radps,
                                           roll_rad=roll_rad,
                                           r_radps=r_radps)

        # return self.elevator_hold(pitch_angle_reference=vs, pitch_angle_current=pitch_rad, pitch_angle_rate_current=q_radps)

    def altitude_hold(self, altitude_reference_ft, altitude_ft, ground_speed, pitch_rad, alpha_rad, q_radps, roll_rad, r_radps):
        vs = (altitude_reference_ft - altitude_ft) * 0.0833333
        vs = np.clip(vs, -5.08, 5.08)
        return self.vertical_speed_hold(speed_reference=vs,
                                        ground_speed=ground_speed,
                                        pitch_rad=pitch_rad,
                                        alpha_rad=alpha_rad,
                                        q_radps=q_radps,
                                        roll_rad=roll_rad,
                                        r_radps=r_radps)