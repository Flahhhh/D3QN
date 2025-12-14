import torch
from torch import optim

import gymnasium as gym
import ale_py

from utils import make_cartpole_env as make_env
from agent import DuellingDDQN
from net import Net

gym.register_envs(ale_py)


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
net = Net()
optimizer = optim.AdamW(net.parameters(), lr=0.00005, amsgrad=True)

agent = DuellingDDQN(net, optimizer, device=device)
env = make_env()

agent.train(env, 6400)
