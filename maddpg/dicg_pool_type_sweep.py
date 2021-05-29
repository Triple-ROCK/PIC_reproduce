from trainer import RL_trainer
from arguments import get_td3_args, get_variable_args, get_default_args
from utils import dict2csv
import os


def main():
    performances = {}
    args = get_variable_args()
    args = get_default_args(args)
    save_dir = args.save_dir
    if args.alg == 'td3':
        args = get_td3_args(args)
    for pool_type in ['avg', 'sum', 'vdn']:
        args.critic_type = 'dicg_' + pool_type
        args.save_dir = save_dir
        args.exp_name = args.scenario + '_' + args.critic_type + '_' + args.alg
        trainer = RL_trainer(args)
        best_eval_reward = trainer.train()
        performances[args.critic_type] = best_eval_reward
    print(performances)
    dict2csv(performances, os.path.join(args.save_dir, 'gcn_sweep.csv'))


if __name__=='__main__':
    main()