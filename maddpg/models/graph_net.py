import torch
import torch.nn as nn
from torch.nn.modules.module import Module
import torch.nn.functional as F


class GraphConvLayer(Module):
    """Implements a GCN layer."""

    def __init__(self, input_dim, output_dim):
        super(GraphConvLayer, self).__init__()
        self.lin_layer = nn.Linear(input_dim, output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, input_feature, input_adj):
        feat = self.lin_layer(input_feature)
        out = torch.matmul(input_adj, feat)
        return out

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.input_dim) + ' -> ' \
               + str(self.output_dim) + ')'


class GraphNet(nn.Module):
    """
    A graph net that is used to pre-process actions and states, and solve the order issue.
    """

    def __init__(self, sa_dim, n_agents, hidden_size, pool_type='avg'):
        super(GraphNet, self).__init__()
        self.sa_dim = sa_dim
        self.n_agents = n_agents
        self.pool_type = pool_type
        self.gc1 = GraphConvLayer(sa_dim, hidden_size)
        self.nn_gc1 = nn.Linear(sa_dim, hidden_size)
        self.gc2 = GraphConvLayer(hidden_size, hidden_size)
        self.nn_gc2 = nn.Linear(hidden_size, hidden_size)

        self.V = nn.Linear(hidden_size, 1)
        self.V.weight.data.mul_(0.1)
        self.V.bias.data.mul_(0.1)

        # Assumes a fully connected graph.
        self.register_buffer('adj', (torch.ones(n_agents, n_agents) - torch.eye(n_agents)) / self.n_agents)

    def forward(self, x):
        """
        :param x: [batch_size, self.n_agent, self.sa_dim] tensor
        :return: [batch_size, 1] tensor
        """

        feat = F.relu(self.gc1(x, self.adj))
        feat += F.relu(self.nn_gc1(x))
        feat /= (1. * self.n_agents)
        out = F.relu(self.gc2(feat, self.adj))
        out += F.relu(self.nn_gc2(feat))
        out /= (1. * self.n_agents)

        # Pooling
        if self.pool_type == 'avg':
            ret = out.mean(1)  # Pooling over the agent dimension.
        elif self.pool_type == 'max':
            ret, _ = out.max(1)
        elif self.pool_type == 'sum':
            ret = out.sum(1)  # This is different from original implementation
        else:
            raise NotImplementedError(self.pool_type + ' is not implemented!')

        # Compute V
        V = self.V(ret)
        return V
