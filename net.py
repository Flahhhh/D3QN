from torch import nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.activation = nn.ReLU()
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 128)

        self.fc_val = nn.Linear(128, 1)
        self.fc_adv = nn.Linear(128, 2)

    def forward(self, x):
        x = self.flatten(x)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))

        val = self.fc_val(x)
        adv = self.fc_adv(x)

        return val + (adv - adv.mean(1, keepdim=True))
