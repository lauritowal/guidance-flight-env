import ray
from guidance_flight_env.examples.rllib.custom_callbacks import CustomCallbacks
from guidance_flight_env.examples.rllib.rllib_wrapper_env import RllibWrapperEnv
from ray.rllib.agents.ddpg import TD3Trainer, td3
from ray.tune import register_env
import numpy as np
from guidance_flight_env.services.plotter import Plotter
from guidance_flight_env.aircraft import cessna172P
import os
dirname = os.path.dirname(__file__)
IMAGE_PATH = os.path.join(dirname, 'images')

dir_path = os.path.dirname(os.path.realpath(__file__))

def in_seconds(minutes: int) -> int:
    return minutes * 60

SEED = 1
def env_creator(config=None):
    return RllibWrapperEnv(config=config)


if __name__ == "__main__":
    ray.init()

    env_config = {
            "jsbsim_path": "/Users/walter/thesis_project/jsbsim",
            "aircraft": cessna172P,
            "agent_interaction_freq": 5,
            "target_radius_km": 100 / 1000,
            "max_distance_km": 2,
            "max_episode_time_s": 60 * 5,
            "phase": 4,
            "seed": SEED,
            "evaluation": True
    }

    default_config = td3.TD3_DEFAULT_CONFIG.copy()
    custom_config = {
        "lr": 0.0001,
        "framework": "torch",
        "callbacks": CustomCallbacks,
        "log_level": "WARN",
        "evaluation_interval": 20,
        "evaluation_num_episodes": 10,
        "num_gpus": 0,
        "explore": False,
        "num_workers": 0,
        "num_envs_per_worker": 1,
        "env_config": env_config
    }

    env = env_creator(env_config)

    config = {**default_config, **custom_config}
    register_env("trackEnvWind", lambda config: env_creator(config))
    resources = TD3Trainer.default_resource_request(config).to_json()

    ### Evaluate
    rewards_history = []
    episode_length_history = []
    simulation_time_step_length_history = []
    images = []
    reward_threshold = None
    episode_counter = 0

    runway_angle_errors = []
    at_targets = []
    on_tracks = []
    headings = []
    bounds = []
    others = []
    distances = []
    successes = 0

    agent = TD3Trainer(config=config, env="guidance_flight_env-continuous-v0")
    agent.restore(f'{dir_path}/data/checkpoints/checkpoint_1/checkpoint-1')

    infos = []
    num_episodes = 100
    for _ in range(num_episodes):
        observation = env.reset()
        episode_rewards = []
        t = 0
        while True:
            action = agent.compute_action(observation)
            observation, reward, done, info = env.step(action)

            episode_rewards.append(reward)

            if done:
                print("###########################")
                print(f"done episode: {episode_counter}")
                print(f"episode time steps {t}")
                print(f"episode sum reward {np.sum(episode_rewards)}")
                print(f"episode mean reward {np.mean(episode_rewards)}")
                print("aircraft_heading_true_deg", info["aircraft_heading_true_deg"])
                print("aircraft_track_angle_deg", info["aircraft_track_angle_deg"])
                print("aircraft_heading_true_deg - aircraft_track_angle_deg", info["aircraft_heading_true_deg"] - info["aircraft_track_angle_deg"])
                print("###########################")

                rewards_history.append(np.sum(episode_rewards))
                episode_length_history.append(t)
                simulation_time_step_length_history.append(info["simulation_time_step"])

                at_targets.append(info["is_aircraft_at_target"])
                on_tracks.append(info["is_on_track"])
                headings.append(info["is_heading_correct"])
                others.append(1 if info["terminal_state"] == "other" else 0)
                bounds.append(1 if info["terminal_state"] == "bounds" else 0)
                distances.append(info["distance_to_target"])
                runway_angle_errors.append(info["runway_angle_error"])

                if info["is_aircraft_at_target"] or info["is_on_track"]:
                    successes += 1

                image = env.render("rgb_array")
                images.append(image)

                path = ""
                if info["is_aircraft_out_of_bounds"]:
                    path = f'{IMAGE_PATH}/episode_{episode_counter}_bounds'
                elif info["is_aircraft_at_target"]:
                    if info["is_heading_correct"]:
                        path = f'{IMAGE_PATH}/episode_{episode_counter}_heading'
                    else:
                        path = f'{IMAGE_PATH}/episode_{episode_counter}_target'
                elif info["is_on_track"]:
                    path = f'{IMAGE_PATH}/episode_{episode_counter}_track'
                else:
                    path = f'{IMAGE_PATH}/episode_{episode_counter}_other'

                image.save(f'{path}.png')
                env.render_html(f'{path}.html')
                infos.append(info)

                episode_counter += 1
                break

    std_reward = np.std(rewards_history)
    mean_reward = np.mean(rewards_history)
    runway_angle_errors_mean = np.mean(np.abs(runway_angle_errors))
    at_targets_sum = np.sum(at_targets)
    on_tracks_sum = np.sum(on_tracks)
    headings_sum = np.sum(headings)
    bounds_sum = np.sum(bounds)
    others_sum = np.sum(others)
    distances_mean = np.mean(distances)

    print("#################################")
    print("######      Results      ########")
    print("#################################")
    print("std_reward", std_reward)
    print("mean_reward", mean_reward)
    print("at target", at_targets_sum)
    print("on tracks", on_tracks_sum)
    print("headings_sum", headings_sum)
    print("others_sum", others_sum)
    print("bounds_sum", bounds_sum)
    print("num total episodes", num_episodes)
    print("distances", distances_mean)
    print("runway_angle_errors (all)", runway_angle_errors_mean)

    indices = [i for i, x in enumerate(at_targets) if x]
    print("success total", successes)
    print("success", successes / num_episodes)

    Plotter.save_images(images=images, infos=infos)
