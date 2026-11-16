import torch
import torch.nn.functional as F
from torch import nn
import math
import torch.distributions as td

class VIB(nn.Module):
    # state belief Guassian fitting
    def __init__(self, state_dim, action_dim, max_action, latent_dim, beta, beta2, neg, device, hidden_dim=750):
        super(VIB, self).__init__()
        self.e1 = nn.Linear(state_dim, hidden_dim)
        self.e2 = nn.Linear(hidden_dim, hidden_dim)
        self.e3 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_std = nn.Linear(hidden_dim, latent_dim)

        self.d1 = nn.Linear(latent_dim, hidden_dim)
        self.d2 = nn.Linear(hidden_dim, hidden_dim)
        self.d3 = nn.Linear(hidden_dim, state_dim)

        self.e_1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.e_2 = nn.Linear(hidden_dim, hidden_dim)
        self.e_3 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_ = nn.Linear(hidden_dim, latent_dim)
        self.log_std_ = nn.Linear(hidden_dim, latent_dim)

        self.relu = nn.ReLU()
        self.beta = beta
        self.beta2 = beta2
        self.neg = neg
        self.max_action = max_action
        self.latent_dim = latent_dim
        self.device = device

    def forward(self, state, action):
        mean, std, log_std = self.encode_sa(state, action)
        z = mean + std * torch.randn_like(std)
        recon = self.decode(z)
        return recon, mean, std, log_std, z

    def train_vib(self, state, action, next_state, reward=None):
        neg_action = torch.empty_like(action).uniform_(-self.max_action, self.max_action)
        mean, std, log_std = self.encode_sa(state, action)
        mean_neg, std_neg, log_std_neg = self.encode_sa(state, neg_action)

        z = mean + std * torch.randn_like(std)
        recon = self.decode(z)
        recon_loss = F.mse_loss(recon, next_state)
        mean_s, std_s, log_std_s = self.encode_s(state)

        KL_loss = -0.5 * (1 + 2 * (log_std - log_std_s) - ((mean - mean_s).pow(2) + std.pow(2)) / std_s.pow(2)).mean()
        s_loss = -0.5 * (1 + 2 * log_std_s - mean_s.pow(2) - std_s.pow(2)).mean() * self.beta2
        neg_loss = -0.5 * (1 + 2 * (log_std_neg - log_std_s.detach()) - ((mean_neg - mean_s.detach()).pow(2) + std_neg.pow(2)) / std_s.detach().pow(2))
        neg_loss = torch.clamp(neg_loss, max=4.0).mean()

        vib_loss = recon_loss + self.beta * (KL_loss + s_loss) - self.neg * neg_loss
        return vib_loss, log_std.max(), log_std.min(), log_std_s.max(), log_std_s.min(), neg_loss

    def kl_loss(self, state, pi):
        mean, std, log_std = self.encode_sa(state, pi)
        mean_s, std_s, log_std_s = self.encode_s(state)
        vib_loss = -0.5 * (1 + 2 * (log_std - log_std_s) - ((mean - mean_s).pow(2) + std.pow(2)) / std_s.pow(2)).mean(-1)
        return vib_loss

    def encode_s(self, state):
        z = self.relu(self.e1(state))
        z = self.relu(self.e2(z))
        z = self.relu(self.e3(z))
        mean = self.mean(z)
        log_std = self.log_std(z).clamp(-5, 10)
        std = torch.exp(log_std)
        return mean, std, log_std

    def encode_sa(self, state, action):
        z = self.relu(self.e_1(torch.cat([state, action], -1)))
        z = self.relu(self.e_2(z))
        z = self.relu(self.e_3(z))
        mean = self.mean_(z)
        log_std = self.log_std_(z).clamp(-6, 10)         # Clamped for numerical stability
        std = torch.exp(log_std)
        return mean, std, log_std

    def decode(self, z):
        z = self.relu(self.d1(z))
        z = self.relu(self.d2(z))
        s = self.d3(z)
        return s
