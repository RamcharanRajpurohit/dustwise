"""DustWise v2: the v1 env with real forecasts, action masks and a configurable water supply."""
import numpy as np

from env import BLOCKS, RAIN_CLEAN, REFILL_L, TANK_L, WASH_L, DustWiseEnv


class DustWiseV2(DustWiseEnv):

    def __init__(self, weather, days=90, refill=REFILL_L):
        super().__init__(weather, days)
        self.sun_fc, self.rain_fc = weather[["sun_fc", "rain_fc"]].to_numpy().T
        self.refill = refill  # litres delivered per day (drought study)

    def _obs(self):
        """Same layout as v1, but the agent sees the forecast instead of the observed weather."""
        obs = super()._obs()
        obs[-3:-1] = self.sun_fc[self.t] / 7, min(self.rain_fc[self.t], 12) / RAIN_CLEAN
        return obs

    def step(self, action):
        tank = self.tank
        obs, reward, terminated, truncated, info = super().step(action)
        self.tank = min(tank - info["water"] + self.refill, TANK_L)
        return self._obs(), reward, terminated, truncated, info

    def action_masks(self):
        """Water wash is only allowed when the tank holds enough for one block."""
        mask = np.ones(1 + 2 * BLOCKS, dtype=bool)
        mask[1 + BLOCKS:] = self.tank >= WASH_L
        return mask
