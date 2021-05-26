import argparse


def get_variable_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--scenario', type=str, default='simple_spread_n6', help='the scenario of the particle-env')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--max_episode_len', '-len', type=int, default=25, help="max length of one episode")
    parser.add_argument('--num_episodes', type=int, default=60000)

    # training configuration
    parser.add_argument('--critic_type', type=str, default='gcn_max', help='the type of the critic')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')
    parser.add_argument('--noise_init', type=float, default=0.1, help='init noise scale')
    parser.add_argument('--noise_final', type=float, default=0, help='final noise scale')
    parser.add_argument('--actor_lr', type=float, default=1e-2)
    parser.add_argument('--critic_lr', type=float, default=1e-2)
    parser.add_argument('--fixed_lr', default=False, action='store_true')
    parser.add_argument('--noise_scale_schedule', '-nss', default=False, action='store_true',
                        help='start annealing noise scale in the middle of the training')
    parser.add_argument('--mlp_hidden_dim', type=int, default=128)
    parser.add_argument('--polyak', type=float, default=0.99)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--discrete_action_space', '-d', default=False, action='store_true')
    parser.add_argument('--cuda', default=False, action='store_true')

    # warm start
    parser.add_argument('--start_steps', type=int, default=int(1e4), help="random action steps")
    parser.add_argument('--replay_start', type=int, default=int(2e3), help="replay when you have enough data")

    # checkpoint
    parser.add_argument('--save_dir', type=str, default="../data", help='model directory of the policy')
    parser.add_argument('--exp_name', type=str, default='test', help='name of the experiment')

    args = parser.parse_args()
    return args


# arguments of maddpg
def get_default_args(args):

    # collect update_every steps of data, then update network for update_times steps
    args.update_every = 100
    args.update_times = 8

    # experience replay
    args.buffer_size = int(1e6)

    # how often to evaluate the model
    args.evaluate_cycle = int(5e3)

    # regularization and gradients norm clipping
    args.actor_reg = 1e-3
    args.grad_norm_clipping = 0.5

    return args