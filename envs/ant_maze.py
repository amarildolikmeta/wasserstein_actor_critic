
import math
import numpy as np
import mujoco_py
from gym import utils
from gym import spaces
from envs.mujoco_env import MujocoEnv

diff_to_path = {
    'empty': 'ant_empty.xml',
}

def q_inv(a):
  return [a[0], -a[1], -a[2], -a[3]]

# x coordinate
# y coordinate
# z coordinate (already in the list)
# https://www.gymlibrary.ml/environments/mujoco/ant/

def q_mult(a, b): # multiply two quaternion
  w = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
  i = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2]
  j = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1]
  k = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]
  return [w, i, j, k]

class AntMazeEnv(MujocoEnv, utils.EzPickle):
  FILE = "ant_empty.xml"
  ORI_IND = 3

  def __init__(self, difficulty=None, expose_all_qpos=True,
               expose_body_coms=None, expose_body_comvels=None,
               clip_state=True, terminal=True, reward_scale=False):
    if difficulty is None:
        difficulty = 'empty'
    model = diff_to_path[difficulty]           
    self._expose_all_qpos = expose_all_qpos
    self._expose_body_coms = expose_body_coms
    self._expose_body_comvels = expose_body_comvels
    self._body_com_indices = {}
    self._body_comvel_indices = {}

    self.clip_state = clip_state
    self.terminal = terminal
    self.reward_scale = reward_scale
    self.goal = [25.0, 0.0]

    self.dist_reward = True
    self.stable_reward = True
    if self.reward_scale and self.dist_reward:
        self.abs_min_reward = np.linalg.norm(self.goal - np.array([0, 9.7]))

    self.bounds = np.array(
      [
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
      ]
    )

    MujocoEnv.__init__(self, model, 5)
    utils.EzPickle.__init__(self)

    # self.observation_space = spaces.Box(self.bounds[0], self.bounds[1], dtype=np.float32)

  @property
  def physics(self):
    # check mujoco version is greater than version 1.50 to call correct physics
    # model containing PyMjData object for getting and setting position/velocity
    # check https://github.com/openai/mujoco-py/issues/80 for updates to api
    if mujoco_py.get_version() >= '1.50':
      return self.sim
    else:
      return self.model

  def _step(self, a):
    return self.step(a)

  def step(self, a):
    xposbefore = self.get_body_com("torso")[0]
    self.do_simulation(a, self.frame_skip)
    xposafter = self.get_body_com("torso")[0]
    # forward_reward = (xposafter - xposbefore) / self.dt
    # ctrl_cost = .5 * np.square(a).sum()
    # survive_reward = 1.0
    # reward = forward_reward - ctrl_cost + survive_reward
    # state = self.state_vector()
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
      ctrl_cost = .5 * np.square(a).sum()
      survive_reward = 1.0
      reward += forward_reward - ctrl_cost + survive_reward

    #   contact_cost = (
    #       0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
    #   )
    #   reward -= contact_cost
      state = self.state_vector()
    # notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
    return ob, reward, done, {} 
    dict(
        reward_forward=forward_reward,
        reward_ctrl=-ctrl_cost,
        reward_survive=survive_reward
    )

  def _get_obs(self):
    # No cfrc observation
    if self._expose_all_qpos:
      obs = np.concatenate([
          self.physics.data.qpos.flat[:15],  # Ensures only ant obs.
          self.physics.data.qvel.flat[:14],
      ])
    else:
      obs = np.concatenate([
          self.physics.data.qpos.flat[2:15],
          self.physics.data.qvel.flat[:14],
      ])

    if self._expose_body_coms is not None:
      for name in self._expose_body_coms:
        com = self.get_body_com(name)
        if name not in self._body_com_indices:
          indices = range(len(obs), len(obs) + len(com))
          self._body_com_indices[name] = indices
        obs = np.concatenate([obs, com])

    if self._expose_body_comvels is not None:
      for name in self._expose_body_comvels:
        comvel = self.get_body_comvel(name)
        if name not in self._body_comvel_indices:
          indices = range(len(obs), len(obs) + len(comvel))
          self._body_comvel_indices[name] = indices
        obs = np.concatenate([obs, comvel])
    return obs

  def reset_model(self):
    qpos = self.init_qpos + self.np_random.uniform(
        size=self.model.nq, low=-.1, high=.1)
    qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1

    # Set everything other than ant to original position and 0 velocity.
    qpos[15:] = self.init_qpos[15:]
    qvel[14:] = 0.
    self.set_state(qpos, qvel)
    return self._get_obs()

  def viewer_setup(self):
    self.viewer.cam.distance = self.model.stat.extent * 0.5

  def get_ori(self):
    ori = [0, 1, 0, 0]
    rot = self.physics.data.qpos[self.__class__.ORI_IND:self.__class__.ORI_IND + 4]  # take the quaternion
    ori = q_mult(q_mult(rot, ori), q_inv(rot))[1:3]  # project onto x-y plane
    ori = math.atan2(ori[1], ori[0])
    return ori

  def set_xy(self, xy):
    qpos = np.copy(self.physics.data.qpos)
    qpos[0] = xy[0]
    qpos[1] = xy[1]

    qvel = self.physics.data.qvel
    self.set_state(qpos, qvel)

  def get_xy(self):
    return self.physics.data.qpos[:2]

if __name__ == "__main__":
    env = AntMazeEnv(difficulty="empty", reward_scale=False)
    ob = env.reset()
    print(env.action_space)
    done = False
    pause_step = 100
    while not done:
        for i in range(pause_step):
            env.render()
            ac = np.random.uniform(-10, 10, 8)
            #print(ac)
            ob, reward, done, _ = env.step(ac)
            print(reward)
            print
        input()
    env.render()




