from guidance_flight_env.agents import  RandomAgent, ConstantAgent
from guidance_flight_env.services.plotter import Plotter
from guidance_flight_env.utils.utils import evaluate
from guidance_flight_env.aircraft import cessna172P

from guidance_flight_env.environments.track_env import TrackEnvNoWind

env = TrackEnvNoWind(
        jsbsim_path="/Users/walter/thesis_project/jsbsim",
        aircraft=cessna172P,
        agent_interaction_freq=5,
        target_radius_km=100 / 1000,
        max_distance_km=4,
        max_target_distance_km=2,
        max_episode_time_s=60 * 5,
        phase=4,
        seed_value=12,
        glide_angle_deg=4)


def test_random_agent(num_episodes = 1):
    agent = RandomAgent(action_space=env.action_space)
    return evaluate(env=env, agent=agent, num_episodes=num_episodes)


def test_constant_agent(num_episodes = 1):
    agent = ConstantAgent(0)
    return evaluate(env=env, agent=agent, num_episodes=num_episodes)


rewards_history, episode_length_history, simulation_time_step_length_history, images, infos = test_constant_agent()
Plotter.save_images(images=images, infos=infos)
