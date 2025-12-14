import torch

from net import Net
from utils import make_cartpole_env as make_env
from agent import DuellingDDQN

if __name__ == '__main__':
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    net = Net().to(device)
    net.load_state_dict(torch.load('logs/D3QN_2025-12-14_20-55/Models/model.pt', weights_only=False))

    env = make_env(test=True)
    agent = DuellingDDQN(net, device=device)

    agent.test(env)