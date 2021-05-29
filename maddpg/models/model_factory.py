"""Implements a model factory."""

import functools
from models import graph_net, dicg_net

MODEL_MAP = {
    'gcn_avg': functools.partial(graph_net.GraphNet, pool_type='avg'),
    'gcn_max': functools.partial(graph_net.GraphNet, pool_type='max'),
    'gcn_sum': functools.partial(graph_net.GraphNet, pool_type='sum'),
    'gcn_vdn': functools.partial(graph_net.GraphNet, pool_type='vdn'),

    'dicg_avg': functools.partial(dicg_net.DICGNet, pool_type='avg'),
    'dicg_max': functools.partial(dicg_net.DICGNet, pool_type='max'),
    'dicg_sum': functools.partial(dicg_net.DICGNet, pool_type='sum'),
    'dicg_vdn': functools.partial(dicg_net.DICGNet, pool_type='vdn'),
}


def get_model_fn(name):
    assert name in MODEL_MAP
    return MODEL_MAP[name]
