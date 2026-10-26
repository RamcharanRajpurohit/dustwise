"""DustWise v3: the v2 env plus the season (day of year), so the agent can anticipate the dust season."""
import numpy as np
from gymnasium import spaces

from env import BLOCKS
from v2.env_v2 import DustWiseV2


class DustWiseV3(DustWiseV2):

    def __init__(self, weather, days=90, **kwargs):
        super().__init__(weather, days, **kwargs)
        angle = 2 * np.pi * weather.index.day_of_year.to_numpy() / 365.25
        self.season = np.c_[1 + np.sin(angle), 1 + np.cos(angle)] / 2  # scaled to [0, 1]
        self.observation_space = spaces.Box(0, np.inf, (2 * BLOCKS + 6,), np.float32)

    def _obs(self):
        return np.append(super()._obs(), self.season[self.t]).astype(np.float32)
