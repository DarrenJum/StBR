import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os
from utils import ReplayBuffer, calculate_norm
import numpy as np


def weights_init_(m, init_w=3e-3):
    m.weight.data.uniform_(-init_w, init_w)
    m.bias.data.uniform_(-init_w, init_w)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256, num_layers=2, init_w=None):
        super(Actor, self).__init__()

        self.trunk = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU())
        for i in range(num_layers):
            self.trunk.add_module(f"{2 + i * 2}", nn.Linear(hidden_dim, hidden_dim))
            self.trunk.add_module(f"{3 + i * 2}", nn.ReLU())

        self.l = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action
        self.action_dim = action_dim

        if init_w:
            weights_init_(self.l, init_w=init_w)

    def forward(self, state):
        action = self.l(self.trunk(state))

        return action if self.max_action is None else self.max_action * torch.tanh(action)


class Flow_policy(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256, num_layers=2, init_w=None):
        super(Flow_policy, self).__init__()

        self.trunk = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU())
        for i in range(num_layers):
            self.trunk.add_module(f"{2 + i * 2}", nn.Linear(hidden_dim, hidden_dim))
            self.trunk.add_module(f"{3 + i * 2}", nn.ReLU())

        self.l = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action
        self.action_dim = action_dim

        if init_w:
            weights_init_(self.l, init_w=init_w)

    def forward(self, state, noise=None):
        noise = torch.randn(state.shape[0], self.action_dim, device=state.device, dtype=state.dtype).clamp(-1, 1) if noise is None else noise
        action = self.l(self.trunk(torch.cat([state, noise], -1)))

        return action if self.max_action is None else self.max_action * torch.tanh(action)


class parallelized_layer(nn.Module):
    def __init__(self, in_dim, out_dim, ensemble=2):
        super().__init__()
        w_init, b_init = [], []
        for _ in range(ensemble):
            l = nn.Linear(in_dim, out_dim)
            w_init.append(l.weight.data.transpose(0, 1))
            b_init.append(l.bias.data.unsqueeze(0))
        w_init = torch.stack(w_init, dim=0)
        b_init = torch.stack(b_init, dim=0)

        self.weight = nn.Parameter(w_init, requires_grad=True)
        self.bias = nn.Parameter(b_init, requires_grad=True)

    def forward(self, x):
        return x @ self.weight + self.bias


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_layers=2, init_w=None, ensemble=2):
        super(Critic, self).__init__()
        # parallelized architecture
        self.ensemble = ensemble
        self.t = nn.Sequential(parallelized_layer(state_dim + action_dim, hidden_dim, ensemble),
                               nn.ReLU(),
                               )
        for i in range(num_layers):
            self.t.add_module(f"{2 + i * 2}", parallelized_layer(hidden_dim, hidden_dim, ensemble))
            self.t.add_module(f"{3 + i * 2}", nn.ReLU())
        self.l = parallelized_layer(hidden_dim, 1, ensemble)

        if init_w:
            weights_init_(self.l, init_w=init_w)

    def forward(self, state, action):
        sa = torch.cat([state, action], -1).repeat(self.ensemble, 1, 1)
        q = self.l(self.t(sa))

        return q[0], q[1]

    def Q1(self, state, action):
        sa = torch.cat([state, action], -1).repeat(self.ensemble, 1, 1)
        q = self.l(self.t(sa))

        return q[0]


class StBR(object):
    def __init__(
            self,
            device,
            env,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            vib_timesteps=1e5,
            # TD3
            lr=3e-4,
            actor_lr=None,
            num_layers=2,
            actor_hidden_dim=256,
            critic_hidden_dim=256,
            actor_init_w=None,
            critic_init_w=None,
            Vmax=float('inf'),
            Vmin=-float('inf'),
            # VIB
            vib=None,
            vib_optimizer=None,
            latent_dim=None,
            lambd=1.0,
            flow_policy=False,
            # finetune
            lambd_cool=False,
            lambd_end=0.2,
    ):
        self.total_it = 0
        self.device = device
        self.is_adroit = any(k in env for k in ('pen', 'door', 'hammer', 'relocate'))
        self.is_antmaze = 'antmaze' in env
        # Actor
        if flow_policy:
            self.actor = Flow_policy(state_dim, action_dim, max_action, actor_hidden_dim, num_layers, actor_init_w).to(device)
        else:
            self.actor = Actor(state_dim, action_dim, max_action, actor_hidden_dim, num_layers, actor_init_w).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr or lr)
        # Critic
        self.critic = Critic(state_dim, action_dim, critic_hidden_dim, num_layers, critic_init_w).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # TD3
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.latent_dim = latent_dim
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self._Vmax = Vmax
        self._Vmin = Vmin

        # VIB
        self.s_std = None
        self.flow_policy = flow_policy
        self.vib_timesteps = vib_timesteps
        self.vib = vib
        self.vib_optimizer = vib_optimizer
        self.lambd = lambd
        self.norm = calculate_norm(tau, 1e5, device)

        # fine-tuning
        self.lambd_cool = lambd_cool
        self.lambd_end = lambd_end

    @torch.no_grad()
    def select_action(self, state):
        self.actor.eval()
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        self.actor.train()
        return action

    def train(self, replay_buffer: ReplayBuffer, batch_size=256, logger=None):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to target policy
            if not self.flow_policy:
                noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            else:
                next_action = self.actor(next_state)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = (target_Q1 + target_Q2) / 2 if self.is_antmaze else torch.min(target_Q1, target_Q2)
            target_Q = (reward + not_done * self.discount * target_Q).clamp(min=self._Vmin, max=self._Vmax)

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        if self.total_it <= self.vib_timesteps:
            self.vib_optimizer.zero_grad()
            vib_loss, theta_max, theta_min, phi_max, phi_min, neg = self.vib.train_vib(state, action, next_state)
            critic_loss += vib_loss

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it <= self.vib_timesteps:
            self.vib_optimizer.step()
            logger.log('train/vib_loss', vib_loss, self.total_it)
            logger.log('train/a_IB_max', theta_max, self.total_it)
            logger.log('train/a_IB_min', theta_min, self.total_it)
            logger.log('train/s_IB_max', phi_max, self.total_it)
            logger.log('train/s_IB_min', phi_min, self.total_it)
            logger.log('train/neg_sample_loss', neg, self.total_it)

        # Log
        logger.log('train/critic_loss', critic_loss, self.total_it)

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            pi = self.actor(state)
            Q, Q2 = self.critic(state, pi)
            Q = (Q + Q2) / 2 if self.is_antmaze else Q
            neg_kl_p = self.vib.kl_loss(state, pi)

            if self.is_adroit:
                q_weight, reg_weight = self.lambd / 1e2, 1.0 - self.lambd # Constant Q-norm
            else:
                self.norm.update(Q.abs().mean().detach())
                q_weight, reg_weight = 1.0 / self.norm.mu, self.lambd # MA Q-norm

            reg_loss = neg_kl_p.mean()
            if self.flow_policy:  # Entropy
                noise = torch.empty_like(action).uniform_(-self.max_action, self.max_action)
                entropy = F.mse_loss(pi, noise)
                reg_loss = reg_loss + 0.2 * entropy
                logger.log('train/entropy', entropy, self.total_it)

            actor_loss = -q_weight * Q.mean() + reg_weight * reg_loss

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Log
            logger.log('train/Q', Q.mean(), self.total_it)
            logger.log('train/actor_loss', actor_loss, self.total_it)
            logger.log('train/neg_kl_p', neg_kl_p.mean(), self.total_it)
            logger.log('train/neg_kl_p_max', neg_kl_p.max(), self.total_it)

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            if not self.flow_policy:
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train_online(self, replay_buffer: ReplayBuffer, vib_optimizer, batch_size=256, logger=None):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            if not self.flow_policy:
                noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            else:
                next_action = self.actor(next_state)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = (target_Q1 + target_Q2) / 2 if self.is_antmaze else torch.min(target_Q1, target_Q2)
            target_Q = (reward + not_done * self.discount * target_Q).clamp(min=self._Vmin, max=self._Vmax)

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        if not self.lambd_cool:
            self.vib_optimizer.zero_grad()
            vib_loss, theta_max, theta_min, phi_max, phi_min, neg = self.vib.train_vib(state, action, next_state)
            critic_loss += vib_loss

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if not self.lambd_cool:
            self.vib_optimizer.step()
            logger.log('train/vib_loss', vib_loss, self.total_it)
            logger.log('train/a_IB_max', theta_max, self.total_it)
            logger.log('train/a_IB_min', theta_min, self.total_it)
            logger.log('train/s_IB_max', phi_max, self.total_it)
            logger.log('train/s_IB_min', phi_min, self.total_it)
            logger.log('train/neg_sample_loss', neg, self.total_it)

        # Log
        logger.log('train/critic_loss', critic_loss, self.total_it)

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            pi = self.actor(state)
            Q, Q2 = self.critic(state, pi)
            Q = (Q + Q2) / 2 if self.is_antmaze else Q

            neg_kl_p = self.vib.kl_loss(state, pi)

            if self.lambd_cool:
                lambd = self.lambd * max(self.lambd_end, (1 - self.total_it / 1e6))
                logger.log('train/lambd', lambd, self.total_it)
            else:
                lambd = self.lambd

            if self.is_adroit:
                q_weight, reg_weight = lambd / 1e2, 1.0 - lambd # Constant Q-norm
            else:
                self.norm.update(Q.abs().mean().detach())
                q_weight, reg_weight = 1.0 / self.norm.mu, lambd # MA Q-norm

            reg_loss = neg_kl_p.mean()
            if self.flow_policy:  # Entropy
                noise = torch.empty_like(action).uniform_(-self.max_action, self.max_action)
                entropy = F.mse_loss(pi, noise)
                reg_loss = reg_loss + 0.2 * entropy
                logger.log('train/entropy', entropy, self.total_it)

            actor_loss = -q_weight * Q.mean() + reg_weight * reg_loss

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Log
            logger.log('train/Q', Q.mean(), self.total_it)
            logger.log('train/actor_loss', actor_loss, self.total_it)
            logger.log('train/neg_kl_p', neg_kl_p.mean(), self.total_it)
            logger.log('train/neg_kl_p_max', neg_kl_p.max(), self.total_it)

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            if not self.flow_policy:
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, model_dir, type=None):
        type = type or self.total_it
        torch.save(self.critic.state_dict(), os.path.join(model_dir, f"critic_s{str(type)}.pth"))
        torch.save(self.critic_target.state_dict(), os.path.join(model_dir, f"critic_target_s{str(type)}.pth"))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(model_dir, f"critic_optimizer_s{str(type)}.pth"))

        torch.save(self.actor.state_dict(), os.path.join(model_dir, f"actor_s{str(type)}.pth"))
        torch.save(self.actor_target.state_dict(), os.path.join(model_dir, f"actor_target_s{str(type)}.pth"))
        torch.save(self.actor_optimizer.state_dict(), os.path.join(model_dir, f"actor_optimizer_s{str(type)}.pth"))
        torch.save(self.vib.state_dict(), os.path.join(model_dir, f"vib_s{str(type)}.pth"))
        torch.save(self.vib_optimizer.state_dict(), os.path.join(model_dir, f"vib_optimizer_s{str(type)}.pth"))

    def load(self, model_dir, step=1000000):
        self.critic.load_state_dict(
            torch.load(os.path.join(model_dir, f"critic_s{str(step)}.pth"), map_location=self.device))
        self.critic_target.load_state_dict(
            torch.load(os.path.join(model_dir, f"critic_target_s{str(step)}.pth"), map_location=self.device))
        self.critic_optimizer.load_state_dict(
            torch.load(os.path.join(model_dir, f"critic_optimizer_s{str(step)}.pth"), map_location=self.device))

        self.actor.load_state_dict(
            torch.load(os.path.join(model_dir, f"actor_s{str(step)}.pth"), map_location=self.device))
        self.actor_target.load_state_dict(
            torch.load(os.path.join(model_dir, f"actor_target_s{str(step)}.pth"), map_location=self.device))
        self.actor_optimizer.load_state_dict(
            torch.load(os.path.join(model_dir, f"actor_optimizer_s{str(step)}.pth"), map_location=self.device))
        self.vib.load_state_dict(
            torch.load(os.path.join(model_dir, f"vib_s{str(step)}.pth"), map_location=self.device))
        self.vib.load_state_dict(
            torch.load(os.path.join(model_dir, f"vib_optimizer_s{str(step)}.pth"), map_location=self.device))
