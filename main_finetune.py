import numpy as np
import torch
import gym
import argparse
import json
import yaml
import StBR
import utils
from utils import VideoRecorder
from tqdm import trange
from coolname import generate_slug
from log import Logger
from vib import VIB
from eval import eval_policy
import os
import d4rl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--policy", default="StBR", type=str)       # Policy name
    parser.add_argument("--env", type=str)                          # OpenAI gym environment name
    parser.add_argument("--seed", default=1, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_freq", default=1e4, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
    parser.add_argument("--save_model_best", default=False, action="store_true")
    parser.add_argument('--save_model_final', default=True, action='store_true')
    parser.add_argument('--eval_episodes', default=10, type=int)
    parser.add_argument('--save_video', default=False, action='store_true')
    # TD3
    parser.add_argument("--expl_noise", default=0.1, type=float)    # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)         # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)    # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--actor_lr', default=None, type=float)
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--actor_hidden_dim', default=256, type=int)
    parser.add_argument('--critic_hidden_dim', default=256, type=int)
    parser.add_argument("--normalize", default=True, action='store_true')
    parser.add_argument("--clip_v", default=True, action='store_true')
    # SBGF
    parser.add_argument('--latent_dim', default=None, type=int)
    parser.add_argument('--lambd', default=1.0, type=float)
    parser.add_argument('--flow_policy', default=False, action='store_true')
    parser.add_argument("--scheduler", default=False, action='store_true')
    parser.add_argument("--beta", default=0.001, type=float)
    parser.add_argument("--beta2", default=0.1, type=float)
    parser.add_argument("--neg_sample", default=0.0, type=float)
    # Antmaze and Adroit
    parser.add_argument('--antmaze_center_reward', default=0.0, type=float)
    parser.add_argument('--antmaze_no_normalize', default=False, action='store_true')
    parser.add_argument('--reward_standardize', default=False, action='store_true')
    # Finetune
    parser.add_argument('--pretrain_model', default=None, type=str)
    parser.add_argument('--pretrain_step', default=1e6, type=int)
    parser.add_argument('--buffer_size', default=2e6, type=int)
    parser.add_argument('--start_times', default=0, type=int)
    parser.add_argument('--lambd_cool', default=False, action='store_true')
    parser.add_argument('--lambd_end', default=0.2, type=float)
    # Config
    parser.add_argument('--work_dir', type=str)
    parser.add_argument('--config', default='configs/online_finetune/antmaze-large-diverse.yml', type=str)

    args = parser.parse_args()
    # log config
    if args.config is not None:
        with open(args.config, 'r') as f:
            parser.set_defaults(**yaml.load(f.read(), Loader=yaml.FullLoader))
        args = parser.parse_args()
    args.cooldir = generate_slug(2)

    # Build work dir
    if args.policy == 'StBR':
        base_dir = 'runs'
        utils.make_dir(base_dir)
        base_dir = os.path.join(base_dir, args.work_dir)
        utils.make_dir(base_dir)
        args.work_dir = os.path.join(base_dir, args.env)
        utils.make_dir(args.work_dir)
    else:
        raise NotImplementedError

    # make directory
    # ts = time.gmtime()
    # ts = time.strftime("%m-%d-%H:%M", ts)
    exp_name = str(args.env)
    exp_name += '-lad' + str(args.lambd) + '-cool' + str(args.lambd_cool) + '_fpi' + f"{args.flow_policy}"
    exp_name += '-' + args.cooldir

    args.work_dir = args.work_dir + '/' + exp_name
    utils.make_dir(args.work_dir)
    args.model_dir = os.path.join(args.work_dir, 'model')
    utils.make_dir(args.model_dir)
    args.video_dir = os.path.join(args.work_dir, 'video')
    utils.make_dir(args.video_dir)

    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    # utils.snapshot_src('.', os.path.join(args.work_dir, 'src'), '.gitignore')

    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    env = gym.make(args.env)

    # Set seeds
    utils.set_seed_everywhere(env, args)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # State Belief Gaussian Fitting (SBGF)
    latent_dim = int(state_dim * 1.2) if args.latent_dim is None else int(args.latent_dim)
    vib = VIB(state_dim, action_dim, max_action, latent_dim, args.beta, args.beta2, args.neg_sample, device).to(device)
    vib_optimizer = torch.optim.Adam(vib.parameters(), lr=3e-4, weight_decay=0)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=vib_optimizer, gamma=0.95)

    kwargs = {
        "device": device,
        "env": args.env,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        # TD3
        "policy_noise": args.policy_noise * max_action,
        "noise_clip": args.policy_noise * 2.5 * max_action,
        "policy_freq": args.policy_freq,
        "lr": args.lr,
        "actor_lr": args.actor_lr,
        "num_layers": args.num_layers,
        "actor_hidden_dim": args.actor_hidden_dim,
        "critic_hidden_dim": args.critic_hidden_dim,
        # VIB
        "vib": vib,
        "vib_optimizer": vib_optimizer,
        "latent_dim": latent_dim,
        "lambd": args.lambd,
        "flow_policy": args.flow_policy,
        # finetune
        "lambd_cool": args.lambd_cool,
        "lambd_end": args.lambd_end,
    }

    # Initialize policy
    if args.policy == 'StBR':
        policy = StBR.StBR(**kwargs)
        UTD = 1
    else:
        raise NotImplementedError

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device, max_size=int(args.buffer_size))
    replay_buffer.convert_D4RL_finetune(d4rl.qlearning_dataset(env), args.reward_standardize)

    if 'antmaze' in args.env and args.antmaze_center_reward is not None:
        # Center reward for Ant-Maze
        # See https://github.com/aviralkumar2907/CQL/blob/master/d4rl/examples/cql_antmaze_new.py#L22
        replay_buffer.reward = np.where(replay_buffer.reward == 1.0, args.antmaze_center_reward, -1.0)

    if args.normalize and not ('antmaze' in args.env and args.antmaze_no_normalize):
        mean, std = replay_buffer.normalize_states()
    else:
        print("No normalize")
        mean, std = 0, 1

    if args.clip_v:
        policy._Vmax = max(0.0, replay_buffer.reward[:replay_buffer.size].max() / (1 - args.discount))
        policy._Vmin = min(0.0, replay_buffer.reward[:replay_buffer.size].min() / (1 - args.discount), policy._Vmax - 1.0 / (1 - args.discount))

    logger = Logger(args.work_dir, use_tb=True, train_log_interval=500)
    video = VideoRecorder(dir_name=args.video_dir)

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    episode_state = [state]

    # load offline pretrained model
    if args.pretrain_model is None:
        print("No pretrained model")
        exit(0)
    else:
        policy.load(args.pretrain_model, int(args.pretrain_step))
        if args.lambd_cool:
            vib.eval()

    best_d4rl_score = 0
    for t in trange(int(args.max_timesteps)):
        episode_timesteps += 1

        # Select action randomly or according to policy
        action = (policy.select_action(state) + np.random.normal(0, max_action * args.expl_noise, size=action_dim)).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        reward_original = reward
        if 'antmaze' in args.env and args.antmaze_center_reward is not None:
            reward = args.antmaze_center_reward if done_bool else -1.0

        # Store data in replay buffer
        if args.normalize and not ('antmaze' in args.env and args.antmaze_no_normalize):
            state = (state - mean) / std
            next_state = (next_state - mean) / std
        replay_buffer.add(state, action, next_state, reward, done_bool)
        logger.log('train/buffer_ptr', replay_buffer.ptr, policy.total_it)

        state = next_state
        episode_reward += reward_original
        episode_state.append(state)

        if t > args.start_times:
            [policy.train_online(replay_buffer, vib_optimizer, args.batch_size, logger=logger) for _ in range(UTD)]

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f} Last Reward: {reward_original:.3f}")
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            episode_state = [state]

        # Evaluate episode
        if t == 0 or (t + 1) % args.eval_freq == 0:
            eval_episodes = 100 if t + 1 == int(args.max_timesteps) and 'antmaze' in args.env else args.eval_episodes
            d4rl_score = eval_policy(args, t + 1, video, logger, policy, args.env, args.seed, mean, std, eval_episodes=eval_episodes)

            if args.save_model_best and d4rl_score > best_d4rl_score:
                best_d4rl_score = d4rl_score
                policy.save(args.model_dir, 'best')

    if args.save_model_final:
        policy.save(args.model_dir)

    logger._sw.close()
