import gym
import numpy as np
from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def act(self, state) -> np.ndarray:
        ...

    @abstractmethod
    def observe(self, state, action, reward, done):
        ...


class RandomAgent(Agent):
    def __init__(self, action_space: gym.Space):
        self.action_space = action_space

    def act(self, state=None):
        return self.action_space.sample()

    def observe(self, state, action, reward, done):
        pass


class ConstantAgent(Agent):
    def __init__(self, constant_action_deg):
        self.constant_action = np.array([np.cos(np.radians(constant_action_deg)),
                                         np.sin(np.radians(constant_action_deg))])

    def act(self, _):
        return self.constant_action

    def observe(self, state, action, reward, done):
        pass