"""Train the DustWise DQN on 2022-2024 weather (2025 is kept for testing)."""
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from data import load_weather
from env import DustWiseEnv

SEED = 0
torch.set_num_threads(1)  # small MLP: extra threads only add overhead

env = Monitor(DustWiseEnv(load_weather().loc[:"2024"]), "logs/dqn")
model = DQN(
    "MlpPolicy", env,
    learning_rate=3e-4,
    buffer_size=100_000,
    learning_starts=10_000,
    batch_size=64,
    gamma=0.9,
    n_steps=3,
    target_update_interval=1_000,
    exploration_fraction=0.2,
    policy_kwargs=dict(net_arch=[128, 128]),
    seed=SEED,
    verbose=1,
)
model.learn(total_timesteps=500_000, log_interval=100)
model.save("models/dustwise_dqn")
