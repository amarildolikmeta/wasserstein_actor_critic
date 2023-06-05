import numpy as np
from gym.envs.mujoco import HumanoidEnv

class HumanoidSaveBounds(HumanoidEnv):

    def __init__(self):
        self.save_bounds = True
        self.mins = np.ones(shape=[376], dtype=np.float64) * np.inf
        self.maxs = np.ones(shape=[376], dtype=np.float64) * - np.inf
        self.random = np.random.randint(0,1e4)
        super().__init__()
        

    def _get_obs(self):
        data = self.sim.data
        res = np.concatenate( # 376
            [
                data.qpos.flat[2:],
                data.qvel.flat,
                data.cinert.flat,
                data.cvel.flat,
                data.qfrc_actuator.flat,
                data.cfrc_ext.flat,
            ]
        )
        if self.save_bounds:
            test = np.stack((res, self.mins), axis=0, out=None)
            self.mins = np.amin(test, axis=0)
            test = np.stack((res, self.maxs), axis=0, out=None)
            self.maxs = np.amax(test, axis=0)

        return res

    def save_bounds_file(self):
        with open('bounds/bounds' + str(self.random) + '.txt', 'w+') as bounds:
            bounds.write(str(self.mins))
            bounds.write(str(self.maxs))








