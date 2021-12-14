import math

import numpy as np
from guidance_flight_env.utils.geoposition_old import GeoPosition
from PIL import Image

EARTH_RADIUS_METERS = 6371e3
EARTH_RADIUS_KM = EARTH_RADIUS_METERS / 1000


# Unit test
def normalize(x, min_x, max_x, a, b) -> float:
    """
    Normalize value x between [a,b] e.g. a=-1, b=1
    # https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
    :param x:
    :param min_x: Describes value
    :param max_x:
    :param a:
    :param b:
    :return:
    """
    return a + (b - a) * ((x - min_x) / (max_x - min_x))


def invert_normalization(x_normalized, min_x, max_x, a, b) -> float:
    return min_x + (((x_normalized - a) * (max_x - min_x)) / (b - a))


# TODO: Write tests and check also manually if the results correspond to http://www.movable-type.co.uk/scripts/latlong.html
# Move to geo object
def get_destination_position_deg(initial_lat_rad, initial_long_rad, heading_rad, distance_km):
    """
    R the earth’s radius
    :param initial_lat_rad: φ
    :param initial_long_rad: λ
    :param heading_rad: θ (clockwise from north)
    :param distance_km: d
    :return:
    """
    φ1 = initial_lat_rad
    λ1 = initial_long_rad
    bearing = heading_rad
    d = distance_km
    R = EARTH_RADIUS_METERS / 1000

    φ2 = math.asin(math.sin(φ1) * math.cos(d / R) +
                   math.cos(φ1) * math.sin(d / R) * math.cos(bearing))

    λ2 = λ1 + math.atan2(math.sin(bearing) * math.sin(d / R) * math.cos(φ1),
                         math.cos(d / R) - math.sin(φ1) * math.sin(φ2))

    λ2_normalised = (np.rad2deg(λ2) + 540) % 360 - 180

    return GeoPosition(np.rad2deg(φ2), λ2_normalised)


def get_circle_around_point(latitude_deg, longitude_deg, radius_km):
    points_on_circle = []
    for heading in range(360):
        point_on_circle = get_destination_position_deg(initial_lat_rad=np.deg2rad(latitude_deg),
                                                       initial_long_rad=np.deg2rad(longitude_deg),
                                                       heading_rad=np.deg2rad(heading),
                                                       distance_km=radius_km)
        points_on_circle.append((point_on_circle.long_deg, point_on_circle.lat_deg))

    return points_on_circle


def get_angle_difference_deg(target_deg, current_deg) -> float:
    a = target_deg - current_deg
    result = (a + 180) % 360 - 180
    return result

def reduce_reflex_angle_deg(angle: float) -> float:
    """ Given an angle in degrees, normalises in [-179, 180] """
    # ATTRIBUTION: solution from James Polk on SO,
    # https://stackoverflow.com/questions/2320986/easy-way-to-keeping-angles-between-179-and-180-degrees#
    new_angle = angle % 360
    if new_angle > 180:
        new_angle -= 360
    return new_angle


def in_seconds(minutes: int) -> int:
    return minutes * 60


# TODO: Put into agent class itself instead
def evaluate(env, agent, num_episodes=1000, reward_threshold= None) -> (float, float, float):
    rewards_history = []
    episode_length_history = []
    simulation_time_step_length_history = []
    images = []
    episode_counter = 0
    infos = []

    for _ in range(num_episodes):
        state = env.reset()
        episode_rewards = []
        t = 0
        while True:
            action = agent.act(state)
            state, reward, done, info = env.step(action)
            infos.append(info)

            episode_rewards.append(reward)
            t += 1

            # print(f"###########################")
            # print(f"Step {t}, done:")
            # print("heading_to_target_deg", math.degrees(heading_to_target_rad))
            # print("runway_angle_rad", math.degrees(runway_angle_rad))
            # print("action", action)
            # print("reward", reward)
            # print(f"###########################")

            if done:
                print("###########################")
                print(f"done episode: {episode_counter}")
                print(f"episode time steps {t}")
                print(f"episode sum reward {np.sum(episode_rewards)}")
                print(f"episode mean reward {np.mean(episode_rewards)}")
                print(f"episode min reward {np.min(episode_rewards)}")
                print(f"episode max reward {np.max(episode_rewards)}")
                print(f"simulation time step {info['simulation_time_step']}")
                # print("heading_to_target_deg", math.degrees(heading_to_target_rad))
                # print("runway_angle_deg", math.degrees(runway_angle_rad))
                print("###########################")

                rewards_history.append(np.sum(episode_rewards))
                episode_length_history.append(t)
                simulation_time_step_length_history.append(info["simulation_time_step"])

                image = env.render("rgb_array")
                images.append(image)

                break
        episode_counter += 1

    std_reward = np.std(rewards_history)
    mean_reward = np.mean(rewards_history)
    print(f"std cumulative reward: {std_reward}")
    print(f"mean cumulative reward: {mean_reward}")
    print(f"min cumulative reward: {np.min(rewards_history)}")
    print(f"max cumulative reward: {np.max(rewards_history)}")

    print(f"mean episode length: {np.mean(episode_length_history)}")
    print(f"min episode length: {np.min(episode_length_history)}")
    print(f"max episode length: {np.max(episode_length_history)}")

    print(f"mean simulation time length: {np.mean(simulation_time_step_length_history)} s - min {np.mean(simulation_time_step_length_history) / 60}")

    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"

    return rewards_history, episode_length_history, simulation_time_step_length_history, images, infos