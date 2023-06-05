import math
import numpy as np
from gym import utils
from envs.mujoco_env import MujocoEnv
# from gym.envs.mujoco.mujoco_env import MujocoEnv

diff_to_path = {
    'empty': 'ant_maze_gym.xml',
}

class AntMazeEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, difficulty=None, clip_state=False, terminal=True, reward_scale=True):
        if difficulty is None:
            difficulty = 'empty'
        model = diff_to_path[difficulty]        

        self.clip_state = clip_state
        self.terminal = terminal
        self.reward_scale = reward_scale
        self.goal = [25.0, 0.0]

        self.dist_reward = True
        self.stable_reward = True
        if self.reward_scale and self.dist_reward:
            self.abs_min_reward = np.linalg.norm(self.goal - np.array([0, 9.7]))

        self.bounds = np.array([
            [
                # coordinates
                0, -9.7, 0, 
                # torso orientation
                -math.pi, -math.pi, -math.pi, -math.pi, 
                # links angles
                -math.pi, -math.pi, -math.pi, -math.pi, 
                -math.pi, -math.pi, -math.pi, -math.pi,
                # torso coordinates velocities
                -50, -50, -50,
                # links angular velocities
                -2*math.pi, -2*math.pi, -2*math.pi,
                # torso angular velocities
                -2*math.pi, -2*math.pi, -2*math.pi, -2*math.pi,
                -2*math.pi, -2*math.pi, -2*math.pi, -2*math.pi, 
            ],
            [
                # coordinates
                25, 9.7, 5, 
                # torso orientation
                math.pi, math.pi, math.pi, math.pi, 
                # links angles
                math.pi, math.pi, math.pi, math.pi, 
                math.pi, math.pi, math.pi, math.pi,
                # torso coordinates velocities
                50, 50, 50,
                # links angular velocities
                2*math.pi, 2*math.pi, 2*math.pi,
                # torso angular velocities
                2*math.pi, 2*math.pi, 2*math.pi, 2*math.pi,
                2*math.pi, 2*math.pi, 2*math.pi, 2*math.pi, 
            ]
        ])

        MujocoEnv.__init__(self, model, 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        
        done = False
        ob = self._get_obs()

        qpos = ob[:15]
        if self.clip_state:
            qvel = ob[15:]
            qpos_clipped = np.zeros_like(qpos)
            qpos_clipped[:3] = np.clip(qpos[:3], a_min=self.bounds[0][:3], a_max=self.bounds[1][:3])
            qpos_clipped[3:] = [ np.arctan2(np.sin(a), np.cos(a)) for a in qpos[3:] ]
            qvel_clipped = np.clip(qvel, a_min=self.bounds[0][15:], a_max=self.bounds[1][15:])
            self.set_state(qpos_clipped, qvel_clipped)
            qpos = qpos_clipped
            ob = self._get_obs()

        reward = 0
        if self.dist_reward:
            reward = -np.linalg.norm(self.goal - qpos[:2])
            done = False
            if reward >= -1. and self.terminal:
                done = True
            if self.reward_scale:
                reward = reward/self.abs_min_reward
        if self.stable_reward:
            forward_reward = (xposafter - xposbefore) / self.dt
            ctrl_cost = 0.5 * np.square(a).sum()
            contact_cost = (
                0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
            )
            survive_reward = 1.0
            reward += forward_reward - ctrl_cost - contact_cost + survive_reward
            state = self.state_vector()
            
            # notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
            # done = not notdone
        
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            # self.sim.data.qpos.flat[2:],
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5


if __name__ == "__main__":
    env = AntMazeEnv(difficulty="empty", reward_scale=True)
    ob = env.reset()
    print(env.action_space)
    done = False
    pause_step = 10
    while not done:
        for i in range(pause_step):
            env.render()
            ac = np.random.uniform(-1, 1, 8)
            #print(ac)
            ob, reward, done, _ = env.step(ac)
            print(reward)
        input()
    env.render()

