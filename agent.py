import os
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from buffer import PERBuffer
from utils import save_model

from torch.utils.tensorboard import SummaryWriter


class DuellingDDQN:
    def __init__(self, model, optimizer, device=None):
        if device is None:
            self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = device

        self.action_space = 2
        self.model = model.to(self.device)
        self.target_model = deepcopy(self.model).to(self.device)

        self.optimizer = optimizer

        self.loss_fn = nn.HuberLoss()

        self.log_dir = "logs"
        self.model_dir = "models"

        self.discount = 0.99

        self.start_eps = 0.9
        self.eps = self.start_eps
        self.min_eps = 0.05

        self.start_train = 2000
        self.buffer = PERBuffer(size=100000, batch_size=128, device=self.device)

        self.target_update_steps = 128
        self.tau = 0.01

    def update_eps(self, step, eps_decay_frames):
        self.eps = self.start_eps - (self.start_eps - self.min_eps) * min(1.0, step / eps_decay_frames)

    def get_action(self, state):
        if np.random.rand() <= self.eps:
            return np.random.choice(self.action_space)
        else:
            with torch.no_grad():
                return self.model(state.to(self.device).unsqueeze(0)).argmax().item()

    def train(self, env, num_epochs):

        dir_name = f"D3QN_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
        log_dir = os.path.join("logs", dir_name)
        writer = SummaryWriter(log_dir=log_dir)
        model_dir = os.path.join(log_dir, "Models")

        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)

        step = 0

        for i in tqdm(range(num_epochs)):
            state, _ = env.reset()
            done = False

            ep_length, sum_reward, sum_loss = 0, 0, 0

            while not done:
                action = self.get_action(state)
                next_state, reward, done, truncated, _ = env.step(action)

                done = done or truncated
                self.buffer.add_sample(state, action, next_state, reward, done)
                step += 1

                ep_length += 1
                state = next_state
                sum_reward += reward

                if len(self.buffer) >= self.start_train:
                    ids, states, actions, next_states, rewards, dones, weights = self.buffer.sample()

                    predictions = torch.gather(self.model(states), dim=1, index=actions.unsqueeze(1)).squeeze()

                    next_actions = self.model(next_states).argmax(1)
                    target_predictions = self.target_model(next_states)
                    target_values = torch.gather(target_predictions, dim=1, index=next_actions.unsqueeze(1)).squeeze()
                    targets = rewards + self.discount * (1 - dones) * target_values.detach()

                    loss = self.loss_fn(predictions, targets)

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                    self.buffer.update_probs(ids, (targets - predictions).abs())
                    sum_loss += loss.item()

                if step % self.target_update_steps == 0:
                    with torch.no_grad():
                        for model_param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
                            target_param.data.copy_(model_param * self.tau + target_param * (1 - self.tau))

            self.update_eps(i, int(0.9*num_epochs))
            writer.add_scalar("Loss/sum_loss", sum_loss, i)
            writer.add_scalar("Loss/mean_loss", sum_loss / ep_length, i)
            writer.add_scalar("Reward/sum_reward", sum_reward, i)
            writer.add_scalar("Reward/mean_reward", sum_reward / ep_length, i)
            writer.add_scalar("Length", ep_length, i)
            writer.add_scalar("EPS", self.eps, i)

        if model_dir:
            save_model(self.model, os.path.join(model_dir, "model.pt"))
        writer.close()
