from .model import Actor
from .memory import Memory, Transition
import torch
import numpy as np
from torch import from_numpy


class Agent:
    def __init__(self, **config):
        self.config = config
        self.batch_size = self.config["batch_size"]
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.memory = Memory(self.config["mem_size"])

        self.policy = Actor(self.config["n_states"], self.config["n_actions"]).to(self.device)
        self.target_policy = Actor(self.config["n_states"], self.config["n_actions"]).to(self.device)
        self.target_policy.load_state_dict(self.policy.state_dict())

    def choose_action(self, state):
        state = np.expand_dims(state, 0)
        state = from_numpy(state).float().to(self.device)
        with torch.no_grad():
            actions = self.policy(state)
        exp_noise = np.random.normal(scale=0.1, size=self.config["n_actions"])

        actions = (actions.cpu().numpy() + exp_noise) * self.config["action_bounds"][1]
        return np.clip(actions, self.config["action_bounds"][0], self.config["action_bounds"][1])

    def store(self, state, action, reward, done, next_state):
        state = torch.Tensor(state).to("cpu")
        action = torch.HalfTensor(action).to("cpu")
        reward = torch.Tensor([reward]).to("cpu")
        done = torch.BoolTensor([done]).to("cpu")
        next_state = torch.Tensor(next_state).to("cpu")

        self.memory.add(state, action, reward, done, next_state)

    def unpack(self, batch):
        batch = Transition(*zip(*batch))

        states = torch.cat(batch.state).view(self.batch_size, self.config["n_states"]).to(self.device)
        actions = torch.cat(batch.action).view((self.batch_size, self.config["n_states"])).to(self.device)
        rewards = torch.cat(batch.reward).view((self.batch_size, 1)).to(self.device)
        next_states = torch.cat(batch.next_state).view(self.batch_size, self.config["n_states"]).to(self.device)
        dones = torch.cat(batch.done).view((-1, 1)).to(self.device)

        return states, actions, rewards, dones, next_states

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        batch = self.memory.sample(self.batch_size)

        states, actions, rewards, dones, next_states = self.unpack(batch)
        with torch.no_grad():
            next_actions = self.target_policy(next_states)

        smoothing_target = torch.normal(0, 0.2, size=next_actions.size(), device=self.device)
        smoothing_target.clamp_(-0.5, 0.5)
        next_actions = (next_actions + smoothing_target) * self.config["action_bounds"][1]
        next_actions.clamp_(self.config["action_bounds"][0], self.config["action_bounds"][1])
        