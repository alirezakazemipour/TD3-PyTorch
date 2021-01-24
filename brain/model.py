from abc import ABC
import torch
from torch import nn
from torch.nn import functional as F


class Actor(nn.Module, ABC):
    def __init__(self, n_states, n_actions):
        super(Actor, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions

        self.fc1 = nn.Linear(self.n_states, 400)
        self.fc2 = nn.Linear(400, 300)
        self.output = nn.Linear(300, self.n_actions)

        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        self.fc1.bias.data.zero_()
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="relu")
        self.fc2.bias.data.zero_()

        nn.init.xavier_uniform_(self.output.weight)
        self.output.bias.data.zero_()

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.output(x))


class Critic(nn.Module, ABC):
    def __init__(self, n_states, n_actions):
        super(Critic, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions

        self.fc1 = nn.Linear(self.n_states + self.n_actions, 400)
        self.fc2 = nn.Linear(400, 300)
        self.value = nn.Linear(300, 1)

        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        self.fc1.bias.data.zero_()
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="relu")
        self.fc2.bias.data.zero_()

        nn.init.xavier_uniform_(self.value.weight)
        self.value.bias.data.zero_()

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.value(x)
