import os
import random
import time
import argparse
from itertools import product

import numpy as np
import torch

from td3_agent import TD3Agent, gru_type
from process_discovery_environment import ProcessDiscoveryEnvironment
from utils import Order, Algo, save_results

def agent_test(env, agent):
    """Test agent on the environment and log results."""
    env.multi_reward = False
    env.multi_log = False
    state, _ = env.reset()
    episode_reward = []

    while True:
        with torch.no_grad():
            action = agent.make_action(state)
            scaled_action = agent.scale_action(np.expand_dims(action, 0), np.array(env.action_space))
            state, reward, done, _ = env.step(np.squeeze(scaled_action, 0))

        episode_reward.append(reward)
        if done:
            print("Test log:", env.log_name, np.array(env.evaluations[1:])[:, 2].mean(), len(env.drift_moments) - 1)
            save_results(env=env, agent_name=type(agent).__name__)
            break

    env.multi_reward = True
    env.multi_log = True


def inference(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using {device} device')

    logs = [(args.log_name, args.log_cut)]
    memory_sizes = [args.memory_size]
    sampling_rates = [args.sampling_rate]
    orders = [Order[args.order]]
    algos = [Algo[args.algo]]
    tops = [None]
    filtering_settings = [args.filtering]
    frequency_settings = [args.frequency]
    update_settings = [args.update]
    update_param_settings = [args.update_param]
    history_windows = [args.history_window]
    observation_windows = [args.observation_window]

    for (log_name, cut), order, algo, top, filtering, frequency, update, update_param, memory_size, sampling_rate, history_window, observation_window in \
        product(logs, orders, algos, tops, filtering_settings, frequency_settings, update_settings, update_param_settings, memory_sizes, sampling_rates, history_windows, observation_windows):

        env = ProcessDiscoveryEnvironment(
            log_name, cut, algo, order, top, filtering, frequency, update, update_param, memory_size, sampling_rate,
            args.max_memory_size, args.min_memory_size, args.max_sampling_rate, args.min_sampling_rate,
            history_window, observation_window, args.drift_punish, args.memory_size_punish, args.reward_value
        )
        start_time = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
        td3_agent = TD3Agent(
            env=env, device=device, state_dim=env.state_dim, action_dim=env.action_dim, learning_rate=args.learning_rate,
            start_time=start_time, replay_buffer_size=args.replay_buffer_size, batch_size=args.batch_size,
            total_episodes=args.total_episodes
        )
        td3_agent.load(args.model_dir)
        agent_test(env, td3_agent)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_name', type=str, default='BPIC2013Incidents')
    parser.add_argument('--log_cut', type=int, default=800)
    parser.add_argument('--memory_size', type=int, default=200)
    parser.add_argument('--sampling_rate', type=float, default=0.8)
    parser.add_argument('--order', type=str, choices=['FRQ', 'MAX', 'MIN'], default='FRQ')
    parser.add_argument('--algo', type=str, choices=['IND', 'ILP'], default='IND')
    parser.add_argument('--filtering', type=bool, default=True)
    parser.add_argument('--frequency', type=bool, default=True)
    parser.add_argument('--update', type=bool, default=True)
    parser.add_argument('--update_param', type=bool, default=True)
    parser.add_argument('--history_window', type=int, default=10)
    parser.add_argument('--observation_window', type=int, default=10)
    parser.add_argument('--drift_punish', type=float, default=1.0)
    parser.add_argument('--memory_size_punish', type=float, default=0.001)
    parser.add_argument('--reward_value', type=str, choices=['absolute', 'relative'], default='absolute')

    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--replay_buffer_size', type=int, default=1_000_000)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--total_episodes', type=int, default=200)
    parser.add_argument('--gru_type', type=str, choices=['full', 'fusion'], default='full')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_dir', type=str, default='/home/waleed/DEMO/RLSPD/model_dir')

    parser.add_argument('--max_memory_size', type=int, default=500)
    parser.add_argument('--min_memory_size', type=int, default=10)
    parser.add_argument('--max_sampling_rate', type=float, default=1.0)
    parser.add_argument('--min_sampling_rate', type=float, default=0.1)

    args = parser.parse_args()
    inference(args)