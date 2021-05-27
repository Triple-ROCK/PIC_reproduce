import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.graph_net import GraphConvLayer


class AttentionModule(nn.Module):
    """ Applies attention mechanism on the `context` using the `query`.

    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`
    """

    def __init__(self, dimensions, attention_type='general'):
        super().__init__()

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query):
        """
        Self attention
            bs, n_agents, emb_feat_dim = query.shape
        """

        if self.attention_type in ['general', 'dot']:
            context = query.transpose(-2, -1).contiguous()  # (batch_size, emb_feat_dim, n_agents)
            if self.attention_type == 'general':
                query = self.linear_in(query)
            attention_scores = torch.matmul(query, context)  # (batch_size, n_agents, n_agents)
            attention_weights = self.softmax(attention_scores)

        elif self.attention_type == 'identity':
            # 只看自己
            n_agents = query.shape[-2]
            attention_weights = torch.zeros(query.shape[:-2] + (n_agents, n_agents))
            attention_weights.reshape(-1, n_agents, n_agents)
            for i in range(n_agents):
                if len(query.shape) > 2:
                    attention_weights[:, i, i] = 1
                else:
                    attention_weights[i, i] = 1
            attention_weights = \
                attention_weights.reshape(query.shape[:-2] + (n_agents, n_agents))

        elif self.attention_type == 'uniform':
            # 全连接图且连接自己
            n_agents = query.shape[-2]
            attention_weights = torch.ones(query.shape[:-2] + (n_agents, n_agents))
            attention_weights = attention_weights / n_agents

        else:
            raise NotImplementedError

        return attention_weights


class EncoderLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_dim)

    def forward(self, inputs):
        x = F.relu(self.linear1(inputs))
        x = F.relu(self.linear2(x))
        return x


class DICGBase(nn.Module):
    def __init__(self,
                 sa_dim,
                 n_agents,
                 encoder_hidden_sizes=(128,),
                 embedding_dim=64,
                 attention_type='general',
                 n_gcn_layers=2,):

        super().__init__()

        self._embedding_dim = embedding_dim  # dev

        self.n_gcn_layers = n_gcn_layers  # dev

        self.n_agents = n_agents

        self.layers = []

        self.encoder = EncoderLayer(input_dim=sa_dim,
                                    output_dim=self._embedding_dim,
                                    hidden_size=encoder_hidden_sizes)
        self.layers.append(self.encoder)

        self.attention_layer = AttentionModule(dimensions=self._embedding_dim,
                                               attention_type=attention_type)
        self.layers.append(self.attention_layer)

        self.gcn_layers = [GraphConvLayer(input_dim=self._embedding_dim,
                                          output_dim=self._embedding_dim) for i in range(self.n_gcn_layers)
                           ]
        self.layers.extend(self.gcn_layers)

    def grad_norm(self):
        return np.sqrt(
            np.sum([p.grad.norm(2).item() ** 2 for p in self.parameters()]))

    def forward(self, x):
        # Partially decentralize, treating agents as being independent

        # (batch_size,  n_agents, emb_feat_dim)
        embeddings_collection = []
        embeddings_0 = self.encoder(x)
        embeddings_collection.append(embeddings_0)

        # (batch_size, n_agents, n_agents)
        attention_weights = self.attention_layer.forward(embeddings_0)

        for i_layer, gcn_layer in enumerate(self.gcn_layers):
            # (batch_size, n_agents, emb_feat_dim)
            embeddings_gcn = F.relu(gcn_layer(embeddings_collection[i_layer], attention_weights))
            embeddings_collection.append(embeddings_gcn)

        return embeddings_collection, attention_weights


class DICGNet(DICGBase):
    def __init__(self,
                 sa_dim,
                 n_agents,
                 hidden_size,
                 attention_type='general',
                 n_gcn_layers=2,
                 residual=True,
                 pool_type='vdn'):
        super().__init__(
            sa_dim=sa_dim,
            n_agents=n_agents,
            encoder_hidden_sizes=hidden_size,
            embedding_dim=hidden_size,
            attention_type=attention_type,
            n_gcn_layers=n_gcn_layers,
        )
        self.residual = residual
        self.pool_type = pool_type
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        :param x: [batch_size, self.n_agent, self.sa_dim] tensor
        :return: [batch_size, 1] tensor
        """
        embeddings_collection, attention_weights = super().forward(x)
        if self.residual:
            emb = embeddings_collection[0] + embeddings_collection[-1]
        else:
            emb = embeddings_collection[-1]

        # Pooling
        if self.pool_type == 'avg':
            ret = emb.mean(1)  # Pooling over the agent dimension.
        elif self.pool_type == 'max':
            ret, _ = emb.max(1)
        elif self.pool_type == 'sum':
            ret = emb.sum(1)  # This is different from original implementation
        elif self.pool_type == 'vdn':
            ret = self.V(emb)
            return ret.sum(-2)
        else:
            raise NotImplementedError(self.pool_type + ' is not implemented!')

        # Compute V
        V = self.V(ret)
        return V
