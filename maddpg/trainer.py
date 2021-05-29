import time
from copy import deepcopy

from arguments import get_variable_args, get_default_args, get_td3_args
from logger import EpochLogger
from agents.td3 import TD3
from utils import make_env, onehot
from agents.ddpg import MADDPG
from buffer import ReplayBuffer

import utils
import torch
import numpy as np
import os
import random


class RL_trainer:
    def __init__(self, args):
        self.args = args
        # torch.set_num_threads(1)

        self.env = make_env(args.scenario, args)
        self.test_env = make_env(args.scenario, args)

        # random seed
        self.env.seed(args.seed)
        self.test_env.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

        if args.discrete_action_space:
            args.act_shape = utils.n_actions(self.env.action_space)[0]
        else:
            args.act_shape = self.env.action_space[0].shape[0]
        args.act_limit = self.env.action_space[0].high[0]
        args.obs_shape = self.env.observation_space[0].shape[0]
        args.save_dir = os.path.join(args.save_dir, args.exp_name)
        args.n_agents = self.env.n
        args.replay_start = max(args.batch_size, args.replay_start)
        args.device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
        self.device = args.device
        self.best_eval_reward = -100000000

        if args.alg == 'ddpg':
            self.agent = MADDPG(args)
            var_counts = tuple(utils.count_vars(module) for module in [self.agent.actor, self.agent.critic])
        elif args.alg == 'td3':
            self.agent = TD3(args)
            var_counts = tuple(utils.count_vars(module) for module in [self.agent.ac.pi, self.agent.ac.q1])
        else:
            raise NotImplementedError
        self.buffer = ReplayBuffer(args)

        # set up logger
        self.logger = EpochLogger(output_dir=args.save_dir, exp_name=args.exp_name)
        self.logger.save_config(args)

        # Count variables
        self.logger.log('\nNumber of parameters: \t actor: %d, \t critic: %d\n' % var_counts)

        # set up saver
        self.logger.setup_pytorch_saver(self.agent)

    def train(self):
        time_steps, start_time, episodes = 0, time.time(), 0
        while episodes < self.args.num_episodes:
            obs_n, ep_ret, ep_len, done = self.env.reset(), 0, 0, False
            while not done and ep_len < self.args.max_episode_len:
                if time_steps > self.args.start_steps:
                    action_n = self.agent.chooce_action(torch.tensor(obs_n, dtype=torch.float32).to(self.device))
                else:
                    action_n = [self.env.action_space[i].sample() for i in range(self.args.n_agents)]
                    if self.args.discrete_action_space:
                        action_n = onehot(action_n, self.args.act_shape)

                new_obs_n, rew_n, done_n, info_n = self.env.step(deepcopy(action_n))
                done = all(done_n)
                self.buffer.store(np.array(obs_n), np.array(action_n),
                                  np.mean(rew_n), np.array(new_obs_n), done)  # It is necessary to use np.mean(rew_n)

                obs_n = new_obs_n
                ep_ret += np.sum(rew_n)  # use sum for better comparison with the paper
                ep_len += 1

                if time_steps % self.args.update_every == 0 and time_steps > self.args.replay_start:
                    # collect update_every steps, then update network
                    for i in range(self.args.update_times):
                        batch = self.buffer.sample(self.args.batch_size)
                        self.agent.learn(batch, self.logger, i)

                if time_steps % self.args.evaluate_cycle == 0 and time_steps > self.args.replay_start:
                    self.evaluate()
                    test_ret = self.logger.get_stats('TestEpRet')[0]
                    if test_ret > self.best_eval_reward:
                        self.logger.save_state({})
                        self.best_eval_reward = test_ret
                    self.log_diagnostics(time_steps, start_time, episodes)

                time_steps += 1

            episodes += 1
            self.logger.store(EpRet=ep_ret)
            if not self.args.fixed_lr:
                self.agent.adjust_lr(episodes)
            if self.args.noise_scale_schedule:
                self.agent.adjust_noise_scale(episodes)
        self.logger.save_final_agent()
        return self.best_eval_reward

    def evaluate(self, render=False):
        for i in range(32):
            obs_n, ep_ret, ep_len, done = self.test_env.reset(), 0, 0, False
            while not done and ep_len < self.args.max_episode_len:
                if render:
                    self.test_env.render()
                    time.sleep(1e-1)
                action_n = self.agent.chooce_action(torch.tensor(obs_n, dtype=torch.float32).to(self.device),
                                                    deterministic=True)
                obs_n, rew_n, done_n, info_n = self.test_env.step(deepcopy(action_n))
                done = all(done_n)
                ep_ret += np.sum(rew_n)
                ep_len += 1
            self.logger.store(TestEpRet=ep_ret)

    def log_diagnostics(self, current_steps, start_time, episodes):
        logger = self.logger
        logger.log_tabular('Episodes', episodes)
        logger.log_tabular('TotalEnvInteracts', current_steps)
        logger.log_tabular('Time', time.time() - start_time)
        logger.log_tabular('EpRet')
        logger.log_tabular('TestEpRet')
        logger.log_tabular('pi_lr', self.agent.actor_optimizer.param_groups[0]['lr'])
        logger.log_tabular('noise_scale', self.agent.noise_scale)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossQ', average_only=True)
        logger.log_tabular('QVals', with_min_and_max=True)
        logger.log_tabular('pi_grad_norm', average_only=True)
        logger.log_tabular('q_grad_norm', average_only=True)
        logger.dump_tabular()


def main():
    args = get_variable_args()
    args = get_default_args(args)
    if args.alg == 'td3':
        args = get_td3_args(args)
    trainer = RL_trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
