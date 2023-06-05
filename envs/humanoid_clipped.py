import numpy as np
from gym.envs.mujoco import HumanoidEnv
from gym import spaces


class HumanoidClipped(HumanoidEnv):

    def __init__(self):
        self.bounds = [-120, 120]
        super().__init__()
        s = self.reset()
        low = np.ones_like(s) * self.bounds[0]
        high = np.ones_like(s) * self.bounds[1]
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        

    def _get_obs(self):
        data = self.sim.data
        res = np.concatenate(
            [
                data.qpos.flat[2:],
                data.qvel.flat,
                data.cinert.flat,
                data.cvel.flat,
                data.qfrc_actuator.flat,
                data.cfrc_ext.flat,
            ]
        )
        #return np.clip(res, self.bounds[0], self.bounds[1])
        return res






