import gym.envs.registration

gym.envs.register(
     id='track-env-wind-v0',
     entry_point='environments.track_env:TrackEnvWind'
)

gym.envs.register(
     id='track-env-no-wind-v0',
     entry_point='environments.track_env:TrackEnvNoWind'
)