import gymnasium
import numpy as np
import torch
from gymnasium import ObservationWrapper
from gymnasium.wrappers import RecordVideo

def save_model(model, path):
    torch.save(model.state_dict(), path)

class PyTorchObsWrapper(ObservationWrapper):
    def observation(self, observation: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(observation).float()

def make_cartpole_env(test=False):
    env = gymnasium.make("CartPole-v1", render_mode="rgb_array" if test else None)
    if test:
        env = RecordVideo(env, video_folder="./videos", name_prefix="test", fps=30)

    env = PyTorchObsWrapper(env)

    return env
