import numpy as np
from gym import utils
from envs.mujoco_env import MujocoEnv
import math

from gym import spaces

diff_to_path = {
    'empty': 'point_empty.xml',
    'easy': 'point.xml',
    'medium': 'point_medium.xml',
    'hard': 'point_hard.xml',
    'harder': 'point_harder.xml',
    'maze': 'maze.xml',
    'maze_easy': 'maze_easy.xml',
    'maze_simple': 'maze_simple.xml',
    'maze_med': 'maze_med.xml',
    'maze_hard': 'maze_hard.xml',
    'double_L': 'double_L.xml',
    'double_I': 'double_I.xml',
    'para': 'point_para.xml'
}


class PointEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, difficulty=None, max_state=500, clip_state=True, terminal=False, z_dim=False, reward_scale=True, sparse_reward=False):
        if difficulty is None:
            difficulty = 'easy'
        model = diff_to_path[difficulty]
        self.max_state = max_state
        self.clip_state = clip_state
        self.bounds = [[0, -9.7, 0], [25, 9.7, 0]]
        self.vbounds = [[-50, -50, 0], [50, 50, 0]]
        self.terminal = terminal
        self.z_dim = z_dim
        self.goal = [25.0, 0.0]
        self.reward_scale = reward_scale
        self.sparse_reward = sparse_reward
        if self.reward_scale:
            self.abs_min_reward = np.linalg.norm(self.goal - np.array([0, 9.7]))

        MujocoEnv.__init__(self, model, 1)
        utils.EzPickle.__init__(self)

        low = np.array(self.bounds[0] + self.vbounds[0])
        high = np.array(self.bounds[1] + self.vbounds[1])
        if self.z_dim:
            self.observation_space = spaces.Box(low, high, dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low[[0,1,3,4]], high[[0,1,3,4]], dtype=np.float32)

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.do_simulation(action, self.frame_skip)
        next_obs = self._get_obs(z=True)

        qpos = next_obs[:3]
        if self.clip_state:
            qvel = next_obs[3:]
            qpos_clipped = np.clip(qpos, a_min=self.bounds[0], a_max=self.bounds[1])
            qvel_clipped = np.clip(qvel, a_min=self.vbounds[0], a_max=self.vbounds[1])
            self.set_state(qpos_clipped, qvel_clipped)
            qpos = qpos_clipped
        next_obs = self._get_obs(self.z_dim)
        reward = -np.linalg.norm(self.goal - qpos[:2])
        done = False
        if reward >= -2. and self.terminal:
            done = True
        if self.sparse_reward:
            reward = -1
        if self.reward_scale and not self.sparse_reward:
            reward = reward/self.abs_min_reward
        return next_obs, reward, done, {}

    def _get_obs(self, z=False):
        if z:
            return np.concatenate([
                self.sim.data.qpos.flat,
                self.sim.data.qvel.flat,
            ])
        else: 
            return np.concatenate([
                self.sim.data.qpos.flat[:2],
                self.sim.data.qvel.flat[:2],
            ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1

        if self.clip_state:
            qpos_clipped = np.clip(qpos, a_min=self.bounds[0], a_max=self.bounds[1])
            qvel_clipped = np.clip(qvel, a_min=self.vbounds[0], a_max=self.vbounds[1])
            self.set_state(qpos_clipped, qvel_clipped)
        else:
            self.set_state(qpos, qvel)
            
        return self._get_obs(z=self.z_dim)

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent


if __name__ == "__main__":
    env = PointEnv(difficulty='para')
    ob = env.reset()
    print(env.action_space)
    done = False
    while not done:
        env.render()
        command = input()
        try:
            x, y = [float(a) for a in command.split(' ')]
        except:
            x, y = 0, 0
        ac = np.array([[x, y]])
        print(ac)
        env.step(ac)
    env.render()