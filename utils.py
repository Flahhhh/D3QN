import gymnasium
import numpy as np
import torch
from gymnasium import ObservationWrapper

def save_model(model, path):
    torch.save(model.state_dict(), path)

class PyTorchObsWrapper(ObservationWrapper):
    def observation(self, observation: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(observation).float()

def make_lunar_env():
    env = gymnasium.make("CartPole-v1")
    env = PyTorchObsWrapper(env)

    return env
