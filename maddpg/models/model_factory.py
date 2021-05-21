"""Implements a model factory."""

import functools
from maddpg.models import graph_net

MODEL_MAP = {
    'gcn_mean': functools.partial(graph_net.GraphNet, pool_type='avg'),
    'gcn_max': functools.partial(graph_net.GraphNet, pool_type='max'),
    'gcn_sum': functools.partial(graph_net.GraphNet, pool_type='sum'),
}


def get_model_fn(name):
    assert name in MODEL_MAP
    return MODEL_MAP[name]
