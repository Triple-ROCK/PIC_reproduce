from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

from models import model_factory
from utils import adjust_lr_, adjust_noise_scale


class Actor(nn.Module):
    def __init__(self, hidden_size, num_inputs, num_outputs, act_limit):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.mu = nn.Linear(hidden_size, num_outputs)
        self.act_limit = act_limit

    def forward(self, inputs):
        x = inputs
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        mu = self.mu(x)
        mu = torch.tanh(mu) * self.act_limit
        return mu


class Critic(nn.Module):
    def __init__(self, hidden_size, num_inputs, num_outputs, num_agents, critic_type):
        super(Critic, self).__init__()

        self.num_agents = num_agents
        self.critic_type = critic_type
        sa_dim = num_inputs + num_outputs
        self.net_fn = model_factory.get_model_fn(critic_type)
        self.net = self.net_fn(sa_dim, num_agents, hidden_size)

    def forward(self, inputs, actions):
        # Assume dimension is: (batch_size, num_agents, specific_dim)
        x = torch.cat((inputs, actions), dim=2)
        V = self.net(x)
        return V


class ActorCritic(nn.Module):

    def __init__(self, mlp_hidden_dim, obs_shape, act_shape, act_limit, num_agents, critic_type):
        super().__init__()

        # build policy and value functions
        self.pi = Actor(mlp_hidden_dim, obs_shape, act_shape, act_limit)
        self.q1 = Critic(mlp_hidden_dim, obs_shape, act_shape, num_agents, critic_type)
        self.q2 = Critic(mlp_hidden_dim, obs_shape, act_shape, num_agents, critic_type)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).cpu().numpy()


class TD3:
    def __init__(self, args):
        self.args = args
        self.obs_shape = args.obs_shape
        self.act_shape = args.act_shape
        self.act_limit = args.act_limit
        self.n_agents = args.n_agents
        self.critic_type = args.critic_type
        self.device = args.device

        self.ac = ActorCritic(args.mlp_hidden_dim, self.obs_shape, self.act_shape,
                              self.act_limit, self.n_agents, self.critic_type).to(self.device)

        # create target actor-critic
        self.ac_target = deepcopy(self.ac).to(self.device)
        for p in self.ac_target.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = list(self.ac.q1.parameters()) + list(self.ac.q2.parameters())

        # optimizers
        self.actor_optimizer = Adam(self.ac.pi.parameters(), lr=args.actor_lr)
        self.critic_optimizer = Adam(self.q_params, lr=args.critic_lr)

        # training configurations
        self.gamma = args.gamma
        self.polyak = args.polyak
        self.noise_scale = args.noise_init
        self.target_noise = args.target_noise
        self.noise_clip = args.noise_clip
        self.policy_delay = args.policy_delay

    def chooce_action(self, obs, deterministic=False):
        a = self.ac.act(obs)
        if not deterministic:
            a += self.noise_scale * np.random.randn(*a.shape)
        return np.clip(a, -self.act_limit, self.act_limit)

    def learn(self, batch, logger, train_steps):
        for key in batch.keys():  # 把batch里的数据转化成tensor
            batch[key] = torch.tensor(batch[key], dtype=torch.float32).to(self.device)
        o, u, r, o_next, done = batch['o'], batch['u'], batch['r'], batch['o_next'], batch['done']

        # Bellman backup for Q functions
        with torch.no_grad():
            # 維度(batch_size, n_agents, act_shape)
            target_u_next = self.ac_target.pi(o_next)

            # Target policy smoothing
            epsilon = torch.randn_like(target_u_next) * self.target_noise
            epsilon = torch.clamp(epsilon, -self.noise_clip, self.noise_clip).to(self.device)
            u2 = target_u_next + epsilon
            u2 = torch.clamp(u2, -self.act_limit, self.act_limit)

            # Target Q-values
            q1_pi_targ = self.ac_target.q1(o_next, u2)
            q2_pi_targ = self.ac_target.q2(o_next, u2)
            q_target = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - done) * q_target  # 維度(batch_size, 1)

        # update Q-network
        self.critic_optimizer.zero_grad()

        q1 = self.ac.q1(o, u)  # 維度(batch_size, 1)
        q2 = self.ac.q2(o, u)
        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2
        loss_q.backward()
        q_grad_norm = clip_grad_norm_(self.q_params, max_norm=self.args.grad_norm_clipping)
        self.critic_optimizer.step()

        logger.store(LossQ=loss_q.item(), QVals=q1.mean().item(), q_grad_norm=q_grad_norm)

        # Possibly update pi and target networks
        if train_steps % self.policy_delay == 0:
            # update Pi-network
            # freeze params in q-network
            for params in self.q_params:
                params.requires_grad = False

            self.actor_optimizer.zero_grad()
            u_current = self.ac.pi(o)
            loss_pi = -self.ac.q1(o, u_current).mean()
            pi_reg = (u_current ** 2).mean()
            loss_pi_total = loss_pi + pi_reg * self.args.actor_reg

            loss_pi_total.backward()
            pi_grad_norm = clip_grad_norm_(self.ac.pi.parameters(), max_norm=self.args.grad_norm_clipping)
            self.actor_optimizer.step()

            # unfreeze params after updating policy
            for params in self.q_params:
                params.requires_grad = True

            # update target network by polyak averaging
            self.soft_update(self.ac, self.ac_target)
            logger.store(LossPi=loss_pi.item(), pi_grad_norm=pi_grad_norm)

    def adjust_lr(self, i_episode):
        start_episode = self.args.start_steps // self.args.max_episode_len
        adjust_lr_(self.actor_optimizer, self.args.actor_lr, i_episode, self.args.num_episodes, start_episode)
        adjust_lr_(self.critic_optimizer, self.args.critic_lr, i_episode, self.args.num_episodes, start_episode)

    def adjust_noise_scale(self, i_episode):
        start_episode = self.args.num_episodes // 2
        self.noise_scale = adjust_noise_scale(self.args.noise_init, self.args.noise_final, i_episode,
                                              self.args.num_episodes, start_episode)

    def soft_update(self, source, target):
        with torch.no_grad():
            for p, p_targ in zip(source.parameters(), target.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

