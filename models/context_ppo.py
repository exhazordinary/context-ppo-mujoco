import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, context_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim + context_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh()
        )
        self.actor_mean = nn.Linear(128, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc(x)
        return x

    def get_action_dist(self, x):
        x = self.fc(x)
        mean = self.actor_mean(x)
        std = self.actor_log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def get_value(self, x):
        x = self.fc(x)
        return self.critic(x)


class PPOAgent:
    def __init__(self, obs_dim, action_dim, context_dim, device):
        self.model = ActorCritic(obs_dim, action_dim, context_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.device = device

        self.buffer = []
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.entropy_coef = 0.01

    def get_action(self, obs, context):
        obs = torch.tensor(obs, dtype=torch.float32)
        context = torch.tensor(context, dtype=torch.float32)
        input_tensor = torch.cat([obs, context]).to(self.device)

        dist = self.model.get_action_dist(input_tensor)
        action = dist.sample()
        return action.cpu().numpy()

    def store_transition(self, obs, action, reward, next_obs, done, context):
        self.buffer.append((obs, action, reward, next_obs, done, context))

    def learn(self):
        if not self.buffer:
            return

        obs, actions, rewards, next_obs, dones, contexts = zip(*self.buffer)

        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        contexts = torch.tensor(contexts, dtype=torch.float32).to(self.device)

        inputs = torch.cat([obs, contexts], dim=1)
        dists = self.model.get_action_dist(inputs)
        values = self.model.get_value(inputs).squeeze()
        returns = self.compute_returns(rewards, dones)

        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        log_probs = dists.log_prob(actions).sum(dim=1)
        entropy = dists.entropy().sum(dim=1).mean()

        actor_loss = -(log_probs * advantages).mean()
        critic_loss = (returns - values).pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.buffer.clear()

    def compute_returns(self, rewards, dones):
        returns = []
        G = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            G = r + self.gamma * G * (1 - d)
            returns.insert(0, G)
        return torch.tensor(returns, dtype=torch.float32).to(self.device)
