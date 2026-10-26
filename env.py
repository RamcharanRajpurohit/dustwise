"""DustWise: daily cleaning decisions for 10 solar panel blocks in Jaisalmer."""
import gymnasium as gym
import numpy as np

BLOCKS = 10
MODULES = 1820            # 550 W modules per 1 MWp block
KWP = 1000                # kWp per block
PR = 0.78                 # performance ratio
TARIFF = 2.5              # INR/kWh, Rajasthan solar auctions 2024
SOIL_RATE = 0.004         # energy loss per day at average dust (Gandhinagar: 0.37 %/day)
DUST_MEAN = 950           # ug/m3, mean CAMS dust in Jaisalmer 2022-25
MAX_LOSS = 0.30           # soiling cap (pvlib Kimber)
RAIN_CLEAN = 6            # mm/day that washes panels (pvlib Kimber)
DRY_EFF = 0.9             # share of loose dust removed by dry brush
LABOUR = 0.2 * MODULES    # INR per block per cleaning
WASH_L = 2 * MODULES      # litres per block wash (2 L/module)
WATER_PRICE = 0.24        # INR/L tanker water
TANK_L = 10_000           # site water tank
REFILL_L = 1_000          # water delivered per day


class DustWiseEnv(gym.Env):
    """Action 0 = idle, 1-10 = dry brush block i, 11-20 = water wash block i."""

    def __init__(self, weather, days=90):
        self.sun, self.rain, self.dust, self.sun_fc, self.rain_fc = (
            weather[["sun", "rain", "dust", "sun_fc", "rain_fc"]].to_numpy().T)
        angle = 2 * np.pi * weather.index.day_of_year.to_numpy() / 365.25
        self.season = np.c_[1 + np.sin(angle), 1 + np.cos(angle)] / 2  # scaled to [0, 1]
        self.days = days
        self.action_space = gym.spaces.Discrete(1 + 2 * BLOCKS)
        self.observation_space = gym.spaces.Box(0, np.inf, (2 * BLOCKS + 6,), np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = int(self.np_random.integers(len(self.sun) - self.days))  # keep a next day for the last obs
        self.end = self.t + self.days
        self.loose = self.np_random.uniform(0, 0.05, BLOCKS)  # removable by brush
        self.stuck = np.zeros(BLOCKS)                         # cemented, needs water
        self.tank = TANK_L
        return self._obs(), {}

    def _obs(self):
        """The agent sees the forecast made a day earlier, not the weather that will happen."""
        t = self.t
        weather = [self.sun_fc[t] / 7, min(self.rain_fc[t], 12) / RAIN_CLEAN, self.dust[t] / DUST_MEAN / 10]
        return np.float32([*self.loose * 10, *self.stuck * 10, self.tank / TANK_L, *weather, *self.season[t]])

    def step(self, action):
        cost, water = self._clean(int(action))

        # energy lost to soiling today
        loss = np.minimum(self.loose + self.stuck, MAX_LOSS)
        lost_kwh = KWP * self.sun[self.t] * PR * loss.sum()

        # weather updates the soiling for tomorrow
        rain = self.rain[self.t]
        if rain >= RAIN_CLEAN:
            self.loose[:], self.stuck[:] = 0, 0
        elif rain > 0:  # light rain cements loose dust
            self.stuck += self.loose
            self.loose[:] = 0
        self.loose += SOIL_RATE * self.dust[self.t] / DUST_MEAN
        self.tank = min(self.tank + REFILL_L, TANK_L)

        cost += lost_kwh * TARIFF
        self.t += 1
        truncated = self.t == self.end
        info = {"lost_kwh": lost_kwh, "water": water, "cost": cost}
        return self._obs(), -cost / 10_000, False, truncated, info

    def _clean(self, action):
        """Returns (cost INR, water L) of the chosen cleaning."""
        if action == 0:
            return 0, 0
        wash, block = divmod(action - 1, BLOCKS)
        if not wash:
            self.loose[block] *= 1 - DRY_EFF
            return LABOUR, 0
        if self.tank < WASH_L:  # no water: crew stays idle
            return 0, 0
        self.tank -= WASH_L
        self.loose[block] = self.stuck[block] = 0
        return LABOUR + WASH_L * WATER_PRICE, WASH_L

    def action_masks(self):
        """Water wash is only allowed when the tank holds enough for one block (read by MaskablePPO)."""
        mask = np.ones(1 + 2 * BLOCKS, dtype=bool)
        mask[1 + BLOCKS:] = self.tank >= WASH_L
        return mask
