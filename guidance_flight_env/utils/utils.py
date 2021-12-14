import numpy as np

EARTH_RADIUS_METERS = 6371e3
EARTH_RADIUS_KM = EARTH_RADIUS_METERS / 1000


def normalize_angle_deg(angle: float) -> float:
    """ Given an angle in degrees, normalises in [-179, 180] """
    # ATTRIBUTION: https://github.com/Gor-Ren/gym-jsbsim
    new_angle = angle % 360
    if new_angle > 180:
        new_angle -= 360
    return new_angle

def in_seconds(minutes: int) -> int:
    return minutes * 60


# own file
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

            if done:
                print("###########################")
                print(f"done episode: {episode_counter}")
                print(f"episode time steps {t}")
                print(f"episode sum reward {np.sum(episode_rewards)}")
                print(f"episode mean reward {np.mean(episode_rewards)}")
                print(f"episode min reward {np.min(episode_rewards)}")
                print(f"episode max reward {np.max(episode_rewards)}")
                print(f"simulation time step {info['simulation_time_step']}")
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

