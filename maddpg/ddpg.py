from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

from maddpg.models import model_factory
from maddpg.utils import adjust_lr


class Actor(nn.Module):
    def __init__(self, hidden_size, num_inputs, num_outputs):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.mu = nn.Linear(hidden_size, num_outputs)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.1)

    def forward(self, inputs):
        x = inputs
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        mu = self.mu(x)
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


class MADDPG:
    def __init__(self, args):
        self.args = args
        self.obs_shape = args.obs_shape
        self.act_shape = args.act_shape
        self.n_agents = args.n_agents
        self.critic_type = args.critic_type

        self.actor = Actor(args.mlp_hidden_dim, self.obs_shape, self.act_shape)
        self.critic = Critic(args.mlp_hidden_dim, self.obs_shape, self.act_shape, self.n_agents, self.critic_type)

        # create target-Q and target-actor
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)
        for p in list(self.actor_target.parameters()) + list(self.critic_target.parameters()):
            p.requires_grad = False

        # optimizers
        self.actor_optimizer = Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=args.critic_lr)

        # training configurations
        self.gamma = args.gamma
        self.polyak = args.polyak
        self.noise_scale = args.noise_scale

    def chooce_action(self, obs, deterministic=False, use_target=False, require_grad=False):
        assert len(obs.shape) > 1, 'you should provide a batch-axis'

        if use_target:
            mu = self.actor_target(torch.tensor(obs, dtype=torch.float32))
        else:
            mu = self.actor(torch.tensor(obs, dtype=torch.float32))

        if self.args.discrete_action_space:
            if not deterministic:
                noise = np.log(-np.log(np.random.uniform(0, 1, mu.shape)))
                mu -= torch.tensor(noise, dtype=torch.float32)
            action = F.softmax(mu, dim=1)
        else:
            mu = torch.tanh(mu)
            if not deterministic:
                noise = self.noise_scale * np.random.randn(*mu.shape)
                mu -= torch.tensor(noise, dtype=torch.float32)
            action = mu.clamp(-1, 1)

        if require_grad:
            return action, mu
        else:
            return action.detach().numpy()

    def learn(self, batch):
        for key in batch.keys():  # 把batch里的数据转化成tensor
            batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        o, u, r, o_next, done = batch['o'], batch['u'], batch['r'], batch['o_next'], batch['done']

        # 維度(batch_size, n_agents, act_shape)
        target_u_next = self.chooce_action(o_next, deterministic=True, use_target=True)

        # update Q-network
        self.critic_optimizer.zero_grad()
        q_target = self.critic_target(o_next, torch.tensor(target_u_next, dtype=torch.float32))  # 維度(batch_size, 1)
        backup = r + self.gamma * (1 - done) * q_target

        q = self.critic(o, u)  # 維度(batch_size, 1)
        loss_q = ((backup - q) ** 2).mean()
        loss_q.backward()
        q_grad_norm = clip_grad_norm_(self.critic.parameters(), max_norm=self.args.grad_norm_clipping)
        self.critic_optimizer.step()

        # update Pi-network
        # freeze params in q-network
        for params in self.critic.parameters():
            params.requires_grad = False

        self.actor_optimizer.zero_grad()
        u_current, mu = self.chooce_action(o, deterministic=True, require_grad=True)
        loss_pi = -self.critic(o, u_current).mean()
        pi_reg = (mu ** 2).mean()
        loss_pi_total = loss_pi + pi_reg * self.args.actor_reg

        loss_pi_total.backward()
        pi_grad_norm = clip_grad_norm_(self.actor.parameters(), max_norm=self.args.grad_norm_clipping)
        self.actor_optimizer.step()

        # unfreeze params after updating policy
        for params in self.critic.parameters():
            params.requires_grad = True

        # update target network by polyak averaging
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)

        return [loss_pi.item(), loss_q.item(), q.mean().item(), pi_grad_norm, q_grad_norm]

    def adjust_lr(self, i_episode):
        start_episode = self.args.start_steps // self.args.max_episode_len
        adjust_lr(self.actor_optimizer, self.args.actor_lr, i_episode, self.args.num_episodes, start_episode)
        adjust_lr(self.critic_optimizer, self.args.critic_lr, i_episode, self.args.num_episodes, start_episode)

    def soft_update(self, source, target):
        with torch.no_grad():
            for p, p_targ in zip(source.parameters(), target.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

