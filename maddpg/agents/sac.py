from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.distributions.normal import Normal

from models import model_factory
from utils import adjust_lr_


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, hidden_size, num_inputs, num_outputs, act_limit):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(num_inputs, hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU())
        self.mu_layer = nn.Linear(hidden_size, num_outputs)
        self.log_std_layer = nn.Linear(hidden_size, num_outputs)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()  # r stands for reparameterization trick

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(dim=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(dim=-1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


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
        self.pi = SquashedGaussianMLPActor(mlp_hidden_dim, obs_shape, act_shape, act_limit)
        self.q1 = Critic(mlp_hidden_dim, obs_shape, act_shape, num_agents, critic_type)
        self.q2 = Critic(mlp_hidden_dim, obs_shape, act_shape, num_agents, critic_type)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.cpu().numpy()


class SAC:
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

        # automatically adjust temperature
        self.alpha = torch.as_tensor(0.2, dtype=torch.float32).to(self.device)
        if not self.args.fixed_alpha:
            self.alpha = torch.nn.Parameter(self.alpha, requires_grad=True)
            self.dual_optimizer = Adam([self.alpha], lr=args.actor_lr)

        # training configurations
        self.gamma = args.gamma
        self.polyak = args.polyak
        self.target_ent = -self.act_shape

    def chooce_action(self, obs, deterministic=False):
        return self.ac.act(obs, deterministic)

    def learn(self, batch, logger, train_steps):
        for key in batch.keys():  # 把batch里的数据转化成tensor
            batch[key] = torch.tensor(batch[key], dtype=torch.float32).to(self.device)
        o, u, r, o_next, done = batch['o'], batch['u'], batch['r'], batch['o_next'], batch['done']

        loss_q, q1, q_grad_norm = self.update_critic(o, u, r, o_next, done)
        loss_pi, logp_pi, pi_grad_norm = self.update_actor(o)

        if not self.args.fixed_alpha:
            self.update_dual(logp_pi)

        # update target network by polyak averaging
        self.soft_update(self.ac, self.ac_target)
        logger.store(LossQ=loss_q.item(), QVals=q1.mean().item(), q_grad_norm=q_grad_norm)
        logger.store(LossPi=loss_pi.item(), pi_grad_norm=pi_grad_norm)
        logger.store(entropy=-logp_pi.mean().item(), alpha=self.alpha.item())

    def update_critic(self, o, u, r, o_next, done):
        # Bellman backup for Q functions
        with torch.no_grad():
            # 維度(batch_size, n_agents, act_shape)
            u2, logp_u2 = self.ac.pi(o_next)

            # Target Q-values
            q1_pi_targ = self.ac_target.q1(o_next, u2)
            q2_pi_targ = self.ac_target.q2(o_next, u2)
            q_target = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - done) * (q_target - self.alpha * logp_u2)  # 維度(batch_size, 1)

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

        return loss_q, q1, q_grad_norm

    def update_actor(self, o):
        # update Pi-network
        # freeze params in q-network
        for params in self.q_params:
            params.requires_grad = False

        self.actor_optimizer.zero_grad()
        pi, logp_pi = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()
        pi_reg = (pi ** 2).mean()
        loss_pi_total = loss_pi + pi_reg * self.args.actor_reg

        loss_pi_total.backward()
        pi_grad_norm = clip_grad_norm_(self.ac.pi.parameters(), max_norm=self.args.grad_norm_clipping)
        self.actor_optimizer.step()

        # unfreeze params after updating policy
        for params in self.q_params:
            params.requires_grad = True

        return loss_pi, logp_pi, pi_grad_norm

    def update_dual(self, logp):
        # The dual update is quite intuitive:
        # if target_ent > -logp, then increase alpha to punish violation of the constraint,
        # else decrease alpha to encourage optimize base variable, that is, rewards.
        loss_dual = -self.alpha * (logp.detach() + self.target_ent).mean()
        self.dual_optimizer.zero_grad()
        loss_dual.backward()
        self.dual_optimizer.step()

    def adjust_lr(self, i_episode):
        start_episode = self.args.start_steps // self.args.max_episode_len
        adjust_lr_(self.actor_optimizer, self.args.actor_lr, i_episode, self.args.num_episodes, start_episode)
        adjust_lr_(self.critic_optimizer, self.args.critic_lr, i_episode, self.args.num_episodes, start_episode)
        if not self.args.fixed_alpha:
            adjust_lr_(self.dual_optimizer, self.args.actor_lr, i_episode, self.args.num_episodes, start_episode)

    def soft_update(self, source, target):
        with torch.no_grad():
            for p, p_targ in zip(source.parameters(), target.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

