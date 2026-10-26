"""Train one algorithm on DustWise v2 (2022-24 weather).

    python -m v2.train <dqn|qrdqn|ppo|a2c|maskppo> <seed> [refill litres/day]
"""
import sys

import torch
from sb3_contrib import QRDQN, MaskablePPO
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.monitor import Monitor

from env import REFILL_L
from v2.env_v2 import DustWiseV2
from v2.forecast import load_weather_v2

POLICY = dict(net_arch=[128, 128])
REPLAY = dict(learning_rate=3e-4, buffer_size=100_000, learning_starts=10_000, batch_size=64,
              target_update_interval=1_000, exploration_fraction=0.2)
ALGOS = {
    "dqn": (DQN, dict(REPLAY, n_steps=3)),  # v1 settings
    "qrdqn": (QRDQN, dict(REPLAY, policy_kwargs=dict(POLICY, n_quantiles=50))),  # default 200 is ~16x slower
    "ppo": (PPO, {}),
    "a2c": (A2C, {}),
    "maskppo": (MaskablePPO, {}),  # reads env.action_masks()
}


def name(algo, seed, refill):
    return f"{algo}_s{seed}_w{refill}"


if __name__ == "__main__":
    algo, seed = sys.argv[1], int(sys.argv[2])
    refill = int(sys.argv[3]) if len(sys.argv) > 3 else REFILL_L
    torch.set_num_threads(1)  # small MLP: extra threads only add overhead

    run = name(algo, seed, refill)
    env = Monitor(DustWiseV2(load_weather_v2().loc[:"2024"], refill=refill), f"v2/logs/{run}")
    Algo, kwargs = ALGOS[algo]
    model = Algo("MlpPolicy", env, gamma=0.9, seed=seed, **{"policy_kwargs": POLICY, **kwargs})
    model.learn(total_timesteps=500_000)
    model.save(f"v2/models/{run}")
