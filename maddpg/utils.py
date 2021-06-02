import csv
import numpy as np


def adjust_lr_(optimizer, init_lr, episode_i, num_episode, start_episode):
    if episode_i < start_episode:
        return init_lr
    lr = init_lr * (1 - (episode_i - start_episode) / (num_episode - start_episode))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_noise_scale(init_scale, final_scale, episode_i, num_episode, start_episode):
    if episode_i < start_episode:
        return init_scale
    elif episode_i > start_episode + num_episode:
        return final_scale
    fraction = (episode_i - start_episode) / (num_episode - start_episode)
    return init_scale + fraction * (final_scale - init_scale)


def dict2csv(output_dict, f_name):
    with open(f_name, mode='w') as f:
        writer = csv.writer(f, delimiter=",")
        for k, v in output_dict.items():
            v = [k] + v
            writer.writerow(v)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def make_env(scenario_name, arglist, benchmark=False):
    if arglist.webots:
        env = make_webots_env(scenario_name, arglist, benchmark)
    else:
        env = make_simple_env(scenario_name, arglist, benchmark)
    return env


def make_simple_env(scenario_name, arglist, benchmark):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data,
                            seed_callback=scenario.seed, cam_range=scenario.world_radius,
                            discrete_action_space=arglist.discrete_action_space)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                            seed_callback=scenario.seed, cam_range=scenario.world_radius,
                            discrete_action_space=arglist.discrete_action_space)
    return env


def make_webots_env(scenario_name, arglist, benchmark):
    from webotSim.environment import MultiRobotEnv
    from webotSim.scenarios.spread_n15 import Scenario

    scenario = Scenario(mode='diff', control_durations=arglist.control_durations)
    env = MultiRobotEnv(scenario, reward_callback=scenario.reward, observation_callback=scenario.observation,
                        info_callback=scenario.get_info, reset_callback=scenario.reset_world,
                        done_callback=scenario.is_done)
    return env


def n_actions(action_spaces):
    """
    :param action_space: list
    :return: n_action: list
    """
    n_actions = []
    from gym import spaces
    for action_space in action_spaces:
        if isinstance(action_space, spaces.discrete.Discrete):
            n_actions.append(action_space.n)
        else:
            raise NotImplementedError
    return n_actions


def onehot(action_n, n_actions):
    actions = []
    for action in action_n:
        action_onehot = np.zeros(n_actions)
        action_onehot[action] = 1
        actions.append(action_onehot)
    return actions
