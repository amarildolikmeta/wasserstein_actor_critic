from trainer.particle_trainer import ParticleTrainer
import json
from utils.variant_util import env_producer, get_policy_producer, get_q_producer
from envs.point import PointEnv
from utils.env_utils import env_producer
from trainer.trainer import SACTrainer
from utils.core import np_ify, torch_ify
import numpy as np
from utils.variant_util import build_variant
from utils.pythonplusplus import load_gzip_pickle
from time import sleep
import os

'''
Run point simulations using this files

1. defines path with pkl file
2. produces point env
3. extracts policy from trainer
4. renders point and acts according to policy
'''

policy = 'expl' # expl, target, input, fixed

if not policy in ['input', 'fixed']:
    # path = './data/point/gifs/point1-M/SAC'
    # path = './data/point/gifs/point1-M/OAC'
    # path = './data/point/gifs/point1-M/WACmu'
    # path = './data/point/gifs/point2-DI/SAC'
    # path = './data/point/gifs/point2-DI/OAC'
    # path = './data/point/gifs/point2-DI/WACmu'
    # path = './data/point/gifs/point3-MS/SAC'
    # path = './data/point/gifs/point3-MS/OAC'
    # path = './data/point/gifs/point3-MS/WACmu'
    # path = './data/point/gifs/point4-DL/SAC'
    # path = './data/point/gifs/point4-DL/OAC'
    path = './data/point/gifs/point4-DL/WACmu'

    base_dir = path
    #experiment = os.path.join(path, 'itr_240.zip_pkl')
    experiment = os.path.join(path, 'itr.zip_pkl')
    variant = json.load(open(os.path.join(path, 'variant.json'), 'r'))

    trainer = build_variant(variant, return_replay_buffer=False, return_collectors=False)['trainer']

    # experiment = base_dir + 'params.zip_pkl'
    exp = load_gzip_pickle(experiment)
    trainer.restore_from_snapshot(exp['trainer'])

env_args = {
    'clip_state': True,
    'terminal': True,
    'max_state': 500.0,
    'z_dim': False,
    'reward_scale': False
}
# env = env_producer('point', 0, difficulty='medium', **env_args)
# env = env_producer('point', 0, difficulty='double_I', **env_args)
# env = env_producer('point', 0, difficulty='maze_simple', **env_args)
env = env_producer('point', 0, difficulty='double_L', **env_args)


ob = env.reset()
print(env.action_space)

n = 10000
for i in range(n):
    env.render()


    if i < 250:
        ac = np.array([0,0])

    elif policy == 'expl':
        ac, *_ = trainer.policy(obs=torch_ify(ob))
        ac = np_ify(ac)
    elif policy == 'target':
        ac, *_ = trainer.target_policy(obs=torch_ify(ob))
        ac = np_ify(ac)
    elif policy == 'input':
        if i % 10 == 0: # steps number
            right = float(input("right: "))
            up = float(input("up: "))
            ac = np.array([right,up])
    elif policy == 'fixed':
        ac = np.array([1,0])
        # sleep(0.2)

    #print(ac)
    ob, r, done, *_ = env.step(ac)
    print(ob)
    print(r)
    if ob[1] < -1:
        debug = 0
    #print(r)
    if done:
        print("Reached Goal!")
        break
env.render()
print('end')









