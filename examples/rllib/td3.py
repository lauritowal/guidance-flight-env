import ray
from ray.rllib.agents.ddpg import TD3Trainer, td3
from ray.tune import register_env
import datetime
from ray import tune
import os
import sys

sys.path.append("../..")
checkpoint_dir = os.path.dirname(os.path.realpath(__file__))

from aircraft import cessna172P
from examples.rllib.rllib_wrapper_env import RllibWrapperEnv
from examples.rllib.custom_callbacks import CustomCallbacks

SEED = 1


def env_creator(config=None):
    print("config", config)
    return RllibWrapperEnv(config)


def train(config, reporter):
    agent = TD3Trainer(config=config, env="track-env-no-wind")

    # Uncomment the following and select checkpoint for restoring agent
    # checkpoint_path = f'{checkpoint_dir}/checkpoints/checkpoint_6001/checkpoint-6001'
    # agent.restore(checkpoint_path)

    for i in range(5000):
        agent.train()
        if i % 100 == 0:
            checkpoint = agent.save(checkpoint_dir=f"{checkpoint_dir}/checkpoints")
            print("checkpoint saved at", checkpoint)
    agent.stop()


if __name__ == "__main__":
    ray.init()

    default_config = td3.TD3_DEFAULT_CONFIG.copy()
    custom_config = {
        "lr": 0.0001,
        "framework": "torch",
        "callbacks": CustomCallbacks,
        "log_level": "WARN",
        "evaluation_interval": 20,
        "evaluation_num_episodes": 10,
        "num_gpus": 0,
        "num_workers": 1,
        "num_envs_per_worker": 1,
        "seed": SEED,
        "env_config": {
            "jsbsim_path": "/home/walt/jsbsim",
            "aircraft": cessna172P,
            "agent_interaction_freq": 5,
            "target_radius": 100 / 1000,
            "max_target_distance_km": 2,
            "max_episode_time_s": 60 * 5,
            "phase": 0,
            "seed_value": SEED,
            "evaluation": False
        },
        "evaluation_config": {
            "explore": False
        },
        "evaluation_num_workers": 1,
    }

    config = {**default_config, **custom_config}
    register_env("track-env-no-wind", lambda config: env_creator(config))
    resources = TD3Trainer.default_resource_request(config).to_json()

    # start training
    now = datetime.datetime.now().strftime("date_%d-%m-%Y_time_%H-%M-%S")
    tune.run(train,
             checkpoint_freq=100,
             checkpoint_at_end=True,
             reuse_actors=True,
             keep_checkpoints_num=5,
             name=f"experiment_{now}_seed_{SEED}",
             resources_per_trial=resources,
             config=config)