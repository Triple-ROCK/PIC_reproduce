import time

import numpy as np
import os.path as osp
import torch
from logger import EpochLogger


def load_policy(fpath, final=False):
    """ Load a pytorch policy saved with Spinning Up Logger."""

    file_name = 'final_model.pt' if final else 'model.pt'
    fname = osp.join(fpath, file_name)
    print('\n\nLoading from %s.\n\n' % fname)

    agent = torch.load(fname, map_location='cpu')
    if hasattr(agent, 'ac'):
        agent.ac.cpu()
    elif hasattr(agent, 'actor'):
        agent.actor.cpu()
    else:
        raise NotImplementedError
    agent.args.device = 'cpu'

    # make function for producing an action given a single state
    def get_action(obs_n):
        action_n = agent.chooce_action(torch.Tensor(obs_n), deterministic=True)
        return action_n

    return get_action


def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True):

    logger = EpochLogger()
    obs_n, ep_ret, ep_len, episode = env.reset(), 0, 0, 0
    while episode < num_episodes:
        if render:
            env.render()
            time.sleep(1e-1)
        action_n = get_action(obs_n)
        obs_n, rew_n, done_n, _ = env.step(action_n)
        ep_ret += np.sum(rew_n)
        ep_len += 1
        done = all(done_n)

        if done or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpLen %d' % (episode, ep_ret, ep_len))
            obs_n, ep_ret, ep_len = env.reset(), 0, 0
            episode += 1

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    from utils import make_env

    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=50)
    parser.add_argument("--scenario", type=str, help="name of the scenario script")
    parser.add_argument('--episodes', '-n', type=int, default=10)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--discrete_action_space', '-d', default=False, action='store_true')
    parser.add_argument('--final', default=False, action="store_true")
    args = parser.parse_args()
    env = make_env(args.scenario, args)
    get_action = load_policy(args.fpath, args.final)

    run_policy(env, get_action, args.len, args.episodes, not args.norender)