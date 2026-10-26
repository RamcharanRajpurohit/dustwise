"""Compare the trained DQN with baseline cleaning rules on unseen 2025 weather."""
import itertools

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy, window_func

from data import load_weather
from env import BLOCKS, DustWiseEnv


def run(policy, weather):
    """Plays the whole test year; returns cost (lakh INR), energy lost (MWh), water (kL)."""
    env = DustWiseEnv(weather, days=len(weather) - 1)
    obs, _ = env.reset(seed=0)
    totals, done = np.zeros(3), False
    while not done:
        obs, _, _, done, info = env.step(policy(obs))
        totals += info["cost"], info["lost_kwh"], info["water"]
    return totals / (1e5, 1e3, 1e3)


def rotation(first):
    """Fixed schedule: one block per day in turn."""
    day = itertools.count()
    return lambda obs: first + next(day) % BLOCKS


def greedy_dry(obs):
    """Brush the block with most loose dust; skip if heavy rain is forecast."""
    return 0 if obs[-2] >= 1 else 1 + np.argmax(obs[:BLOCKS])


model = DQN.load("models/dustwise_dqn")
policies = {
    "No cleaning": lambda obs: 0,
    "Rotation dry": rotation(1),
    "Rotation wash": rotation(1 + BLOCKS),
    "Greedy dry": greedy_dry,
    "DQN": lambda obs: model.predict(obs, deterministic=True)[0],
}

test = load_weather().loc["2025"]
results = {name: run(policy, test) for name, policy in policies.items()}
print(f"{'Policy':14} {'Cost (lakh INR)':>15} {'Lost (MWh)':>11} {'Water (kL)':>11}")
for name, (cost, lost, water) in results.items():
    print(f"{name:14} {cost:15.2f} {lost:11.0f} {water:11.0f}")

fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
x, y = ts2xy(load_results("logs"), "timesteps")
left.plot(*window_func(x, y, 200, np.mean))
left.set(title="DQN training (90-day episodes, 2022-24)", xlabel="Timesteps", ylabel="Mean episode reward (10k INR)")
right.bar(results.keys(), [r[0] for r in results.values()])
right.set(title="Total cost on unseen 2025", ylabel="Lost revenue + cleaning (lakh INR)")
right.tick_params(axis="x", rotation=20)
fig.tight_layout()
fig.savefig("results.png", dpi=150)
