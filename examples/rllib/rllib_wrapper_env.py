from environments.track_env import TrackEnvWind


class RllibWrapperEnv(TrackEnvWind):
    def __init__(self, config):

        if config["evaluation"]:
            seed = config["seed"]
        else:
            seed = config["seed"] + config.worker_index + config.num_workers + config.vector_index

        config["seed"] = seed
        super().__init__(**config)

        self.seed(seed)

