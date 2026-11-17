import numpy as np
import torch
import os
import random
import imageio
from tqdm import trange


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, device, max_size=int(1e6)):
        self.max_size = max_size
        self.size = 0
        self.ptr = 0

        self.state = np.empty((max_size, state_dim))
        self.action = np.empty((max_size, action_dim))
        self.next_state = np.empty((max_size, state_dim))
        self.reward = np.empty((max_size, 1))
        self.not_done = np.empty((max_size, 1))

        self.device = device

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):  # RL training sample
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),)

    def convert_D4RL(self, dataset, standardize=False):
        self.state = dataset['observations']
        self.action = dataset['actions']
        self.next_state = dataset['next_observations']
        rewards = dataset['rewards']
        if standardize:
            self.reward = ((dataset['rewards'] - dataset['rewards'].min()) / (
                        dataset['rewards'].max() - dataset['rewards'].min())).reshape(-1, 1)
        else:
            self.reward = rewards.reshape(-1, 1)
        self.not_done = 1. - dataset['terminals'].reshape(-1, 1)
        self.size = self.state.shape[0]

    def convert_D4RL_finetune(self, dataset, standardize=False):
        self.ptr = dataset['observations'].shape[0]
        self.size = dataset['observations'].shape[0]
        self.state[:self.ptr] = dataset['observations']
        self.action[:self.ptr] = dataset['actions']
        self.next_state[:self.ptr] = dataset['next_observations']
        rewards = dataset['rewards']
        if standardize:
            self.reward = ((dataset['rewards'] - dataset['rewards'].min()) / (
                        dataset['rewards'].max() - dataset['rewards'].min())).reshape(-1, 1)
        else:
            self.reward = rewards.reshape(-1, 1)
        self.not_done[:self.ptr] = 1. - dataset['terminals'].reshape(-1, 1)

    def normalize_states(self, eps=1e-3):
        mean = self.state[:self.size].mean(0, keepdims=True)
        std = self.state[:self.size].std(0, keepdims=True) + eps
        self.state = (self.state - mean) / std
        self.next_state = (self.next_state - mean) / std
        return mean, std

    def clip_to_eps(self, eps=1e-6): # due to tanh
        lim = 1 - eps
        self.action = np.clip(self.action, -lim, lim)


def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def set_seed_everywhere(env, args):
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        # Align performance with old cuda devices (e.g. 2080ti)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def snapshot_src(src, target, exclude_from):
    make_dir(target)
    os.system(f"rsync -rv --exclude-from={exclude_from} {src} {target}")


class calculate_norm:
    def __init__(self, alpha, init_mu=None, device='cpu'):
        self.alpha = alpha
        self.mu = torch.tensor(init_mu, device=device)
        self.device = device

    def update(self, current_mu):
        self.mu = self.alpha * current_mu + (1 - self.alpha) * self.mu


class VideoRecorder(object):
    def __init__(self, dir_name, height=512, width=512, camera_id=0, fps=30):
        self.dir_name = dir_name
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.dir_name is not None and enabled

    def record(self, env):
        if self.enabled:
            frame = env.sim.render(
                # mode='rgb_array',
                height=self.height,
                width=self.width,
                # camera_id=self.camera_id
            )
            self.frames.append(frame[::-1,:,:])

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.dir_name, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)
