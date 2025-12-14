import torch
EPS = 1e-10


class PERBuffer:
    def __init__(self, size, batch_size, device=None):
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        self.size = size
        self.batch_size = batch_size
        self.entries = 0

        self.states = torch.zeros((size, 4), dtype=torch.float32)
        self.actions = torch.zeros(size, dtype=torch.long)
        self.rewards = torch.zeros(size, dtype=torch.float32)
        self.next_states = torch.zeros((size, 4), dtype=torch.float32)
        self.dones = torch.zeros(size, dtype=torch.float32)
        self.weights = torch.ones(size, dtype=torch.float32)

        self.alpha = 1

    def __len__(self):
        return self.entries

    def add_sample(self, state, action, next_state, reward, done):
        i = self.__len__() % self.size
        weight = 1.
        if self.__len__() > 0:
            weight = self.weights[:self.entries].max()

        self.states[i] = state
        self.actions[i] = action
        self.next_states[i] = next_state
        self.rewards[i] = reward
        self.dones[i] = done
        self.weights[i] = weight

        self.entries = min(self.size, self.entries + 1)

    def update_probs(self, ids, losses):
        self.weights[ids] = losses.detach().cpu()

    def _get_probs(self):
        return (self.weights[:self.__len__()] + EPS) ** self.alpha / (
                    (self.weights[:self.__len__()] + EPS) ** self.alpha).sum()

    def sample(self):
        probs = self._get_probs()
        ids = torch.multinomial(probs, self.batch_size, replacement=False)

        states = self.states[ids].to(self.device)
        actions = self.actions[ids].to(self.device)
        next_states = self.next_states[ids].to(self.device)
        rewards = self.rewards[ids].to(self.device)
        dones = self.dones[ids].to(self.device)
        probs = probs[ids].to(self.device)

        return ids, states, actions, \
            next_states, rewards, \
            dones, probs
