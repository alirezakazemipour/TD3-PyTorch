from .model import Actor, Critic
from .memory import Memory, Transition
import torch
import numpy as np
from torch import from_numpy, nn


class Agent:
    def __init__(self, **config):
        self.config = config
        self.batch_size = self.config["batch_size"]
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.memory = Memory(self.config["mem_size"])

        self.policy = Actor(self.config["n_states"], self.config["n_actions"]).to(self.device)
        self.target_policy = Actor(self.config["n_states"], self.config["n_actions"]).to(self.device)
        self.hard_copy_target_nets(self.policy, self.target_policy)
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=self.config["lr"])

        self.critic1 = Critic(self.config["n_states"], self.config["n_actions"]).to(self.device)
        self.target_critic1 = Critic(self.config["n_states"], self.config["n_actions"]).to(self.device)
        self.hard_copy_target_nets(self.critic1, self.target_critic1)

        self.critic2 = Critic(self.config["n_states"], self.config["n_actions"]).to(self.device)
        self.target_critic2 = Critic(self.config["n_states"], self.config["n_actions"]).to(self.device)
        self.hard_copy_target_nets(self.critic2, self.target_critic2)

        self.critic_opt = torch.optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()),
                                           lr=self.config["lr"])

        self.mse_loss_fn = torch.nn.MSELoss()

        self.train_counter = 0

    def choose_action(self, state, eval=False):
        state = np.expand_dims(state, 0)
        state = from_numpy(state).float().to(self.device)
        with torch.no_grad():
            actions = self.policy(state)
        actions = actions.cpu().numpy()

        if not eval:
            exp_noise = np.random.normal(scale=0.1, size=self.config["n_actions"])
            actions = actions + exp_noise

        actions = actions * self.config["action_bounds"][1]
        return np.clip(actions, self.config["action_bounds"][0], self.config["action_bounds"][1]).squeeze(0)

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
        actions = torch.cat(batch.action).view((self.batch_size, self.config["n_actions"])).to(self.device)
        rewards = torch.cat(batch.reward).view((self.batch_size, 1)).to(self.device)
        next_states = torch.cat(batch.next_state).view(self.batch_size, self.config["n_states"]).to(self.device)
        dones = torch.cat(batch.done).view((-1, 1)).to(self.device)

        return states, actions, rewards, dones, next_states

    def train(self):
        self.train_counter += 1
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

        with torch.no_grad():
            next_q1 = self.target_critic1(next_states, next_actions)
            next_q2 = self.target_critic2(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2)
        target_q_value = rewards + self.config["gamma"] * next_q * (~dones)

        q1_value = self.critic1(states, actions)
        q2_value = self.critic2(states, actions)
        critic_loss = self.mse_loss_fn(q1_value, target_q_value) + self.mse_loss_fn(q2_value, target_q_value)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        if self.train_counter % self.config["policy_update_period"] == 0:
            policy_loss = - self.critic1(states, self.policy(states))
            policy_loss = policy_loss.mean()

            self.policy_opt.zero_grad()
            policy_loss.backward()
            self.policy_opt.step()

            self.soft_copy_target_nets(self.critic1, self.target_critic1, self.config["tau"])
            self.soft_copy_target_nets(self.critic2, self.target_critic2, self.config["tau"])
            self.soft_copy_target_nets(self.policy, self.target_policy, self.config["tau"])

    @staticmethod
    def hard_copy_target_nets(online_net, target_net):
        target_net.load_state_dict(online_net.state_dict())
        target_net.eval()

    @staticmethod
    def soft_copy_target_nets(online_net, target_net, tau):
        for target_param, online_param in zip(target_net.parameters(), online_net.parameters()):
            target_param.data.copy_(target_param.data * (1 - tau) + online_param * tau)
        target_net.eval()

    def save_weights(self):
        torch.save(self.policy.state_dict(), "weights.pth")

    def load_weights(self):
        self.policy.load_state_dict(torch.load("weights.pth"))
        self.policy.eval()
