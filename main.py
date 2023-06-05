
import argparse
from numpy import e
import torch
import time
import utils.pytorch_util as ptu
from utils.env_utils import domain_to_epoch 
from utils.variant_util import build_variant
from utils.rng import set_global_pkg_rng_state
from launcher_util import run_experiment_here
from rl_algorithm import BatchRLAlgorithm

#import git


# ray.init(
#     # If true, then output from all of the worker processes on all nodes will be directed to the driver.
#     log_to_driver=True,
#     logging_level=logging.WARNING,
#
#     # # The amount of memory (in bytes)
#     # object_store_memory=1073741824, # 1g
#     # redis_max_memory=1073741824 # 1g
# )

# def get_current_branch(dir):
#     from git import Repo
#
#     repo = Repo(dir)
#     return repo.active_branch.name


def experiment(variant, prev_exp_state=None):
    built_variant = build_variant(variant)
    trainer = built_variant['trainer']
    expl_path_collector = built_variant['expl_path_collector']
    remote_eval_path_collector = built_variant['remote_eval_path_collector']
    replay_buffer = built_variant['replay_buffer']

    algorithm = BatchRLAlgorithm(
        trainer=trainer,
        exploration_data_collector=expl_path_collector,
        remote_eval_data_collector=remote_eval_path_collector,
        replay_buffer=replay_buffer,
        optimistic_exp_hp=variant['optimistic_exp'],
        deterministic=variant['alg'] == 'p-oac',
        **variant['algorithm_kwargs']
    )

    algorithm.to(ptu.device)

    if prev_exp_state is not None:
        expl_path_collector.restore_from_snapshot(
            prev_exp_state['exploration'])

        remote_eval_path_collector.restore_from_snapshot(
            prev_exp_state['evaluation_remote'])

        replay_buffer.restore_from_snapshot(prev_exp_state['replay_buffer'])

        trainer.restore_from_snapshot(prev_exp_state['trainer'])

        set_global_pkg_rng_state(prev_exp_state['global_pkg_rng_state'])

    start_epoch = prev_exp_state['epoch'] + \
                  1 if prev_exp_state is not None else 0

    algorithm.train(start_epoch)

def get_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--domain', type=str, default='mountain')
    parser.add_argument('--dim', type=int, default=25)
    parser.add_argument('--pac', action="store_true")
    parser.add_argument('--ensemble', action="store_true")
    parser.add_argument('--n_policies', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--alg', type=str, default='oac', choices=[
        'oac', 'p-oac', 'sac', 'g-oac', 'g-tsac', 'p-tsac', 'ddpg', 'oac-w', 'gs-oac'
    ])
    parser.add_argument('--no_gpu', default=False, action='store_true')
    parser.add_argument('--base_log_dir', type=str, default='./data')
    parser.add_argument('--load_dir', type=str, default='')
    parser.add_argument('--save_heatmap', action="store_true")
    parser.add_argument('--comp_MADE', action="store_true")
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--layer_size', type=int, default=256)
    parser.add_argument('--fake_policy', action="store_true")
    parser.add_argument('--random_policy', action="store_true")
    parser.add_argument('--expl_policy_std', type=float, default=0)
    parser.add_argument('--target_paths_qty', type=float, default=0)
    parser.add_argument('--dont_use_target_std', action="store_true")
    parser.add_argument('--n_estimators', type=int, default=2)
    parser.add_argument('--share_layers', action="store_true")
    parser.add_argument('--mean_update', action="store_true")
    parser.add_argument('--counts', action="store_true", help="count the samples in replay buffer")
    parser.add_argument('--std_inc_prob', type=float, default=0.)
    parser.add_argument('--prv_std_qty', type=float, default=0.)
    parser.add_argument('--prv_std_weight', type=float, default=1.)
    parser.add_argument('--std_inc_init', action="store_true")
    parser.add_argument('--log_dir', type=str, default='./data')
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--max_path_length', type=int, default=1000) # SAC: 1000
    parser.add_argument('--replay_buffer_size', type=float, default=1e6)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256) # SAC: 256
    parser.add_argument('--r_min', type=float, default=0.)
    parser.add_argument('--r_max', type=float, default=1.)
    parser.add_argument('--r_mellow_max', type=float, default=1.)
    parser.add_argument('--mellow_max', action="store_true")
    parser.add_argument('--priority_sample', action="store_true")
    parser.add_argument('--global_opt', action="store_true")
    parser.add_argument('--save_sampled_data', default=False, action='store_true')
    parser.add_argument('--n_components', type=int, default=1)
    parser.add_argument('--snapshot_gap', type=int, default=10)
    parser.add_argument('--keep_first', type=int, default=-1)
    parser.add_argument('--save_fig', action='store_true')
    parser.add_argument(
        '--snapshot_mode',
        type=str,
        default='none',
        choices=['last_every_gap','all','last','gap','gap_and_last','none']
    )
    parser.add_argument(
        '--difficulty', 
        type=str, 
        default='hard', 
        choices=[
            'empty', 'easy', 'medium', 'hard', 'harder', 'maze',
            'maze_easy', 'maze_med', 'maze_simple', 'double_L', 'double_I', 'para', 'maze_hard'
        ],
        help='only for point environment'
    )
    parser.add_argument('--policy_lr', type=float, default=3E-4)
    parser.add_argument('--qf_lr', type=float, default=3E-4)
    parser.add_argument('--std_lr', type=float, default=3E-4)
    parser.add_argument('--target_policy_lr', type=float, default=0)
    parser.add_argument('--sigma_noise', type=float, default=0.0)
    parser.add_argument('--deterministic_rs', action="store_true", help="make riverswim deterministic")
    parser.add_argument('--policy_grad_steps', type=int, default=1)
    parser.add_argument('--fixed_alpha', type=float, default=0)
    parser.add_argument('--stable_critic', action='store_true')

    parser.add_argument('--std_soft_update', action="store_true")
    parser.add_argument('--clip_state', action="store_true", help='only for point environment')
    parser.add_argument('--terminal', action="store_true", help='only for point environment')
    parser.add_argument('--max_state', type=float, default=500., help='only for point environment')
    parser.add_argument('--sparse_reward', action="store_true", help='only for point environment')

    # optimistic_exp_hyper_param
    parser.add_argument('--beta_UB', type=float, default=0.0) # humanoid: 4.66
    parser.add_argument('--trainer_UB', action='store_true')
    parser.add_argument('--delta', type=float, default=0.0) # humanoid: 23.53
    parser.add_argument('--delta_oac', type=float, default=20.53)
    parser.add_argument('--deterministic_optimistic_exp', action='store_true')
    parser.add_argument('--no_resampling', action="store_true",
                        help="Samples are removed from replay buffer after being used once")

    # Training param
    parser.add_argument('--num_expl_steps_per_train_loop',
                        type=int, default=1000) # OAC default 1000
    parser.add_argument('--num_trains_per_train_loop', type=int, default=1000)
    parser.add_argument('--num_train_loops_per_epoch', type=int, default=1)
    parser.add_argument('--num_eval_steps_per_epoch', type=int, default=5000)
    parser.add_argument('--min_num_steps_before_training', type=int, default=1000)
    parser.add_argument('--clip_action', dest='clip_action', action='store_true')
    parser.add_argument('--no_clip_action', dest='clip_action', action='store_false')
    parser.set_defaults(clip_action=True)
    parser.add_argument('--policy_activation', type=str, default='ReLU')
    parser.add_argument('--policy_output', type=str, default='TanhGaussian')
    parser.add_argument('--policy_weight_decay', type=float, default=0)
    parser.add_argument('--reward_scale', type=float, default=1.0)
    parser.add_argument('--entropy_tuning', dest='entropy_tuning', action='store_true')
    parser.add_argument('--no_entropy_tuning', dest='entropy_tuning', action='store_false')
    parser.set_defaults(entropy_tuning=True)
    parser.add_argument('--load_from', type=str, default='')
    parser.add_argument('--train_bias', dest='train_bias', action='store_true')
    parser.add_argument('--no_train_bias', dest='train_bias', action='store_false')
    parser.add_argument('--should_use',  action='store_true')
    parser.add_argument('--stochastic',  action='store_true')
    parser.set_defaults(train_bias=True)
    parser.add_argument('--soft_target_tau', type=float, default=5E-3)
    parser.add_argument('--ddpg', action='store_true', help='use a ddpg version of the algorithms')
    parser.add_argument('--ddpg_noisy', action='store_true', help='use noisy exploration policy')
    parser.add_argument('--std', type=float, default=0.1, help='use noisy exploration policy for ddpg')
    parser.add_argument('--use_target_policy', action='store_true', help='use a target policy in ddpg')
    parser.add_argument('--rescale_targets_around_mean', action='store_true', help='use a target policy in ddpg')

    args = parser.parse_args()

    return args


def get_log_dir(args, should_include_base_log_dir=True, should_include_seed=True, should_include_domain=True):
    start_time = time.time()
    if args.load_dir != '':
        log_dir = args.load_dir
    else:
        if args.n_policies > 1:
            el = str(args.n_policies)
        elif args.n_components > 1:
            el = str(args.n_components)
        else:
            el = ''
        log_dir = args.log_dir + '/' + args.domain + '/' + \
                  (args.difficulty + '/' if args.domain == 'point' else '') + \
                  ('terminal' + '/' if args.terminal and  args.domain == 'point' else '') + \
                  (str(args.dim) + '/' if args.domain == 'riverswim' else '') + \
                  ('global/' if args.global_opt else '') + \
                  ('ddpg/' if args.ddpg else '') + \
                  ('mean_update_' if args.mean_update else '') + \
                  ('_priority_' if args.priority_sample else '') + \
                  ('counts/' if args.counts else '') + \
                  ('/' if args.mean_update and not args.counts else '') + \
                   args.alg + ('_std' if args.std_soft_update else '') + '_' + el + '/' +\
                   args.suffix + '/' # + str(int(start_time))
        if args.log_dir == './data/debug':
            log_dir = log_dir + str(int(start_time))


    return log_dir


if __name__ == "__main__":
    # Parameters for the experiment are either listed in variant below
    # or can be set through cmdline args and will be added or overrided
    # the corresponding attributein variant

    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(1e6), # default 1e6
        algorithm_kwargs=dict(
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=None,
            num_expl_steps_per_train_loop=None,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=32,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
        ),
        optimistic_exp={}
    )

    args = get_cmd_args()

    variant['log_dir'] = get_log_dir(args)
    if args.load_from != '':
        variant['log_dir'] = args.load_from

    variant['seed'] = args.seed
    #variant['git_hash'] = git.Repo().head.object.hexsha
    variant['domain'] = args.domain
    variant['num_layers'] = args.num_layers
    variant['layer_size'] = args.layer_size
    variant['share_layers'] = args.share_layers
    variant['n_estimators'] = args.n_estimators if args.alg in ['p-oac', 'p-tsac'] else 2
    variant['replay_buffer_size'] = int(args.replay_buffer_size)
    variant['algorithm_kwargs']['num_epochs'] = domain_to_epoch(args.domain) if args.epochs <= 0 else args.epochs
    variant['algorithm_kwargs']['num_trains_per_train_loop'] = args.num_trains_per_train_loop
    variant['algorithm_kwargs']['num_expl_steps_per_train_loop'] = args.num_expl_steps_per_train_loop
    variant['algorithm_kwargs']['max_path_length'] = args.max_path_length
    variant['algorithm_kwargs']['num_eval_steps_per_epoch'] = args.num_eval_steps_per_epoch
    variant['algorithm_kwargs']['min_num_steps_before_training'] = args.min_num_steps_before_training
    variant['algorithm_kwargs']['batch_size'] = args.batch_size
    variant['algorithm_kwargs']['save_sampled_data'] = args.save_sampled_data
    variant['algorithm_kwargs']['num_train_loops_per_epoch'] = args.num_train_loops_per_epoch
    variant['algorithm_kwargs']['trainer_UB'] = args.trainer_UB
    variant['algorithm_kwargs']['fake_policy'] = args.fake_policy
    variant['algorithm_kwargs']['random_policy'] = args.random_policy
    variant['algorithm_kwargs']['domain'] = args.domain
    variant['algorithm_kwargs']['target_paths_qty'] = args.target_paths_qty
    # variant['algorithm_kwargs']['log_dir'] = args.log_dir

    variant['delta'] = args.delta
    variant['std'] = args.std
    variant['optimistic_exp']['should_use'] = args.beta_UB > 0 or args.delta > 0 and not args.alg in [
        'p-oac', 'sac','g-oac', 'g-tsac','p-tsac', 'gs-oac', 'ddpg'
    ]
    if not variant['optimistic_exp']['should_use']:
        variant['optimistic_exp']['should_use'] = args.should_use
    variant['optimistic_exp']['beta_UB'] = args.beta_UB if args.alg in ['oac', 'oac-w'] else 0
    variant['optimistic_exp']['delta'] = args.delta if args.alg in ['p-oac', 'oac', 'g-oac', 'oac-w', 'gs-oac'] else 0
    variant['optimistic_exp']['share_layers'] = False
    if args.alg in ['p-oac']:
        variant['optimistic_exp']['share_layers'] = args.share_layers
    if args.should_use and args.alg in ['p-oac']:
        variant['optimistic_exp']['delta'] = args.delta_oac
    variant['optimistic_exp']['deterministic'] = args.deterministic_optimistic_exp
    if args.alg not in ['ddpg']:
        variant['trainer_kwargs']['use_automatic_entropy_tuning'] = args.entropy_tuning
    variant['trainer_kwargs']['discount'] = args.gamma
    variant['trainer_kwargs']['policy_lr'] = args.policy_lr
    variant['trainer_kwargs']['qf_lr'] = args.qf_lr
    if not args.target_policy_lr == 0:
        variant['trainer_kwargs']['target_policy_lr'] = args.target_policy_lr
    variant['trainer_kwargs']['policy_lr'] = args.policy_lr
    variant['ensemble'] = args.ensemble
    variant['n_policies'] = args.n_policies if args.ensemble else 1
    variant['n_components'] = args.n_components
    variant['priority_sample'] = False
    variant['clip_action'] = args.clip_action
    variant['policy_activation'] = args.policy_activation
    variant['policy_output'] = args.policy_output
    variant['stochastic'] = args.stochastic
    if args.domain == 'lqg':
        variant['clip_action'] = True
    if not args.fixed_alpha == 0:
        variant['trainer_kwargs']['fixed_alpha'] = args.fixed_alpha
    # if args.alg in ['g-oac']:
    #     variant['trainer_kwargs']['expl_policy_lr'] = args.expl_policy_lr
    if args.alg in ['p-oac', 'g-oac', 'g-tsac', 'p-tsac', 'oac-w', 'gs-oac']:
        variant['trainer_kwargs']['std_soft_update'] = args.std_soft_update
        variant['trainer_kwargs']['counts'] = args.counts
        variant['trainer_kwargs']['prv_std_qty'] = args.prv_std_qty
        variant['trainer_kwargs']['prv_std_weight'] = args.prv_std_weight
        variant['trainer_kwargs']['dont_use_target_std'] = args.dont_use_target_std
    if args.alg in ['gs-oac', 'oac-w']:
        variant['trainer_kwargs']['train_bias'] = args.train_bias # duplicate
    if args.alg in ['gs-oac']:
        variant['trainer_kwargs']['mean_update'] = args.mean_update # duplicate
        variant['trainer_kwargs']['stable_critic'] = args.stable_critic # 
    if args.alg in ['p-oac', 'g-oac', 'g-tsac', 'p-tsac']:
        variant['trainer_kwargs']['share_layers'] = args.share_layers
        variant['trainer_kwargs']['mean_update'] = args.mean_update 
        variant['trainer_kwargs']['std_inc_prob'] = args.std_inc_prob
        variant['trainer_kwargs']['std_inc_init'] = args.std_inc_init
        variant['trainer_kwargs']['fake_policy'] = args.fake_policy
        variant['priority_sample'] = args.priority_sample
        variant['trainer_kwargs']['global_opt'] = args.global_opt
        variant['trainer_kwargs']['policy_grad_steps'] = args.policy_grad_steps
        if args.alg in ['p-oac', 'g-oac']:
            variant['trainer_kwargs']['r_mellow_max'] = args.r_mellow_max
            variant['trainer_kwargs']['mellow_max'] = args.mellow_max
            variant['algorithm_kwargs']['global_opt'] = args.global_opt
            variant['algorithm_kwargs']['save_fig'] = args.save_fig
            variant['algorithm_kwargs']['expl_policy_std'] = args.expl_policy_std
            variant['trainer_kwargs']['train_bias'] = args.train_bias
            variant['trainer_kwargs']['rescale_targets_around_mean'] = args.rescale_targets_around_mean
            variant['trainer_kwargs']['use_target_policy'] = args.use_target_policy
        if args.alg in ['g-oac', 'oac-w', 'gs-oac']:
            variant['trainer_kwargs']['std_lr'] = args.std_lr
    variant['algorithm_kwargs']['save_heatmap'] = args.save_heatmap
    variant['algorithm_kwargs']['comp_MADE'] = args.comp_MADE
    variant['trainer_kwargs']['policy_weight_decay'] = args.policy_weight_decay
    variant['trainer_kwargs']['reward_scale'] = args.reward_scale

    variant['alg'] = args.alg
    variant['dim'] = args.dim
    variant['difficulty'] = args.difficulty
    variant['max_state'] = args.max_state
    variant['clip_state'] = args.clip_state
    variant['terminal'] = args.terminal
    variant['sparse_reward'] = args.sparse_reward
    variant['pac'] = args.pac
    variant['no_resampling'] = args.no_resampling
    variant['r_min'] = args.r_min
    variant['r_max'] = args.r_max
    variant['sigma_noise'] = args.sigma_noise
    variant['deterministic_rs'] = args.deterministic_rs # added


    variant['trainer_kwargs']['soft_target_tau'] = args.soft_target_tau
    variant['algorithm_kwargs']['ddpg_noisy'] = args.ddpg_noisy
    if args.alg in ['p-oac', 'g-oac', 'g-tsac', 'p-tsac']: # default False
        N_expl = variant['algorithm_kwargs']['num_expl_steps_per_train_loop']
        N_train = variant['algorithm_kwargs']['num_trains_per_train_loop']
        B = variant['algorithm_kwargs']['batch_size']
        N_updates = (N_train * B) / N_expl
        std_soft_update_prob = 2 / (N_updates * (N_updates + 1))
        variant['trainer_kwargs']['std_soft_update_prob'] = std_soft_update_prob
    if args.ddpg or args.alg == 'ddpg':
        variant['algorithm_kwargs']['num_trains_per_train_loop'] = 1
        variant['algorithm_kwargs']['num_expl_steps_per_train_loop'] = 4
        variant['algorithm_kwargs']['num_train_loops_per_epoch'] = args.num_expl_steps_per_train_loop // 4
        variant['trainer_kwargs']['use_target_policy'] = args.use_target_policy
        variant['algorithm_kwargs']['ddpg'] = args.ddpg
        if args.alg == 'ddpg':
            variant['algorithm_kwargs']['ddpg_noisy'] = True
        else:
            variant['algorithm_kwargs']['ddpg_noisy'] = args.ddpg_noisy

    #print("Prob %s" % variant['trainer_kwargs']['std_soft_update_prob'])

    if args.no_resampling:
        variant['algorithm_kwargs']['num_trains_per_train_loop'] = 500
        variant['algorithm_kwargs']['num_expl_steps_per_train_loop'] = 500 * args.batch_size
        variant['algorithm_kwargs']['min_num_steps_before_training'] = 1000 
        # 4 * 500 * args.batch_size # SAC: 10000
        variant['algorithm_kwargs']['batch_size'] = args.batch_size
        variant['replay_buffer_size'] = 5 * 500 * args.batch_size
    if torch.cuda.is_available():
        gpu_id = int(args.seed % torch.cuda.device_count())
    else:
        gpu_id = None
    if not args.no_gpu:
        try:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        except:
            pass
    run_experiment_here(
        experiment,
        variant,
        seed=args.seed,
        use_gpu=not args.no_gpu and torch.cuda.is_available(),
        gpu_id=gpu_id,

        # Save the params every snapshot_gap and override previously saved result
        snapshot_gap=args.snapshot_gap,
        snapshot_mode=args.snapshot_mode,
        keep_first=args.keep_first,

        log_dir=variant['log_dir']
    )
