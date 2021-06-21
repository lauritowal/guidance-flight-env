import gym.envs.registration
from aircraft import Aircraft, cessna172P
from gym_jsbsim import utils
from environments.environment import GuidanceEnv
from utils.utils import in_seconds

gym.envs.register(
     id='track-env-wind',
     entry_point='environments.track_env:TrackEnvWind'
)

gym.envs.register(
     id='track-env-no-wind',
     entry_point='environments.track_env:TrackEnvNoWind'
)