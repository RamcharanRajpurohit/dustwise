"""Train the DustWise agent (MaskablePPO) on 2022-24 weather; 2025 is kept for testing.

    python train.py <seed>
"""
import sys

import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor

from data import load_weather
from env import DustWiseEnv

if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    torch.set_num_threads(1)  # small MLP: extra threads only add overhead

    env = Monitor(DustWiseEnv(load_weather().loc[:"2024"]), f"logs/maskppo_s{seed}")
    model = MaskablePPO(
        "MlpPolicy", env,
        gamma=0.9,
        ent_coef=0.01,  # keeps exploring; best on 2024 validation
        policy_kwargs=dict(net_arch=[128, 128]),
        seed=seed,
    )
    model.learn(total_timesteps=1_000_000)
    model.save(f"models/maskppo_s{seed}")
