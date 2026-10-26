"""Compare the trained agent (3 seeds) with baseline cleaning rules on unseen 2025 weather."""
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO

from data import load_weather
from env import BLOCKS, DustWiseEnv

SEEDS = [0, 1, 2]


def run(policy, weather):
    """Plays the whole test year; returns cost (lakh INR), energy lost (MWh), water (kL)."""
    env = DustWiseEnv(weather, days=len(weather) - 1)
    obs, _ = env.reset(seed=0)
    totals, done = np.zeros(3), False
    while not done:
        obs, _, _, done, info = env.step(policy(obs, env))
        totals += info["cost"], info["lost_kwh"], info["water"]
    return totals / (1e5, 1e3, 1e3)


def rotation(first):
    """Fixed schedule: one block per day in turn."""
    day = itertools.count()
    return lambda obs, env: first + next(day) % BLOCKS


def greedy_dry(obs, env):
    """Brush the block with most loose dust; skip if heavy rain is forecast."""
    return 0 if obs[2 * BLOCKS + 2] >= 1 else 1 + np.argmax(obs[:BLOCKS])


def agent(seed):
    model = MaskablePPO.load(f"models/maskppo_s{seed}")
    return lambda obs, env: model.predict(obs, action_masks=env.action_masks(), deterministic=True)[0]


test = load_weather().loc["2025"]
results = {
    "No cleaning": np.array([run(lambda obs, env: 0, test)]),
    "Rotation dry": np.array([run(rotation(1), test)]),
    "Rotation wash": np.array([run(rotation(1 + BLOCKS), test)]),
    "Greedy dry": np.array([run(greedy_dry, test)]),
    "MaskablePPO": np.array([run(agent(s), test) for s in SEEDS]),
}

print(f"{'Policy':14} {'Cost (lakh INR)':>16} {'Lost (MWh)':>11} {'Water (kL)':>11}")
for name, r in results.items():
    print(f"{name:14} {r[:, 0].mean():9.2f} ± {r[:, 0].std():4.2f} {r[:, 1].mean():11.0f} {r[:, 2].mean():11.0f}")

fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
for s in SEEDS:
    log = pd.read_csv(f"logs/maskppo_s{s}.monitor.csv", skiprows=1)
    left.plot(log.l.cumsum(), log.r.rolling(200).mean(), label=f"seed {s}")
left.set(title="Training (90-day episodes, 2022-24, 200-episode mean)", xlabel="Timesteps",
         ylabel="Episode reward (10k INR)")
left.legend()
right.bar(results.keys(), [r[:, 0].mean() for r in results.values()], yerr=[r[:, 0].std() for r in results.values()])
right.set(title="Total cost on unseen 2025, real forecasts", ylabel="Lost revenue + cleaning (lakh INR)")
right.tick_params(axis="x", rotation=20)
fig.tight_layout()
fig.savefig("results.png", dpi=150)
