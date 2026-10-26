"""Train the v3 agent (MaskablePPO) on 2022-24:  python -m v3.train <seed>"""
import sys

import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor

from v2.forecast import load_weather_v2
from v3.env_v3 import DustWiseV3

if __name__ == "__main__":
    seed = int(sys.argv[1])
    torch.set_num_threads(1)  # small MLP: extra threads only add overhead

    env = Monitor(DustWiseV3(load_weather_v2().loc[:"2024"]), f"v3/logs/maskppo_s{seed}")
    model = MaskablePPO(
        "MlpPolicy", env,
        gamma=0.9,
        ent_coef=0.01,  # keeps exploring; best on 2024 validation
        policy_kwargs=dict(net_arch=[128, 128]),
        seed=seed,
    )
    model.learn(total_timesteps=1_000_000)
    model.save(f"v3/models/maskppo_s{seed}")
