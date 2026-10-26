"""v3 vs v2 MaskablePPO vs greedy on 2025 with real forecasts (3 seeds each)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO

from env import BLOCKS
from v2.env_v2 import DustWiseV2
from v2.forecast import load_weather_v2
from v3.env_v3 import DustWiseV3

SEEDS = [0, 1, 2]


def run(policy, env):
    """Plays the whole test year; returns cost (lakh INR), energy lost (MWh), water (kL)."""
    obs, _ = env.reset(seed=0)
    totals, done = np.zeros(3), False
    while not done:
        obs, _, _, done, info = env.step(policy(obs, env))
        totals += info["cost"], info["lost_kwh"], info["water"]
    return totals / (1e5, 1e3, 1e3)


def agent(path):
    model = MaskablePPO.load(path)
    return lambda obs, env: model.predict(obs, action_masks=env.action_masks(), deterministic=True)[0]


def greedy_dry(obs, env):
    """Brush the block with most loose dust; skip if heavy rain is forecast."""
    return 0 if obs[2 * BLOCKS + 2] >= 1 else 1 + np.argmax(obs[:BLOCKS])


test = load_weather_v2().loc["2025"]
days = len(test) - 1
results = {
    "greedy": np.array([run(greedy_dry, DustWiseV2(test, days))]),
    "v2 maskppo": np.array([run(agent(f"v2/models/maskppo_s{s}_w1000"), DustWiseV2(test, days)) for s in SEEDS]),
    "v3 maskppo": np.array([run(agent(f"v3/models/maskppo_s{s}"), DustWiseV3(test, days)) for s in SEEDS]),
}

print(f"{'Policy':11} {'Cost (lakh INR)':>16} {'Lost (MWh)':>11} {'Water (kL)':>11}")
for name, r in results.items():
    print(f"{name:11} {r[:, 0].mean():9.2f} ± {r[:, 0].std():4.2f} {r[:, 1].mean():11.0f} {r[:, 2].mean():11.0f}")

fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
for s in SEEDS:
    log = pd.read_csv(f"v3/logs/maskppo_s{s}.monitor.csv", skiprows=1)
    left.plot(log.l.cumsum(), log.r.rolling(200).mean(), label=f"seed {s}")
left.set(title="v3 training (200-episode mean)", xlabel="Timesteps", ylabel="Episode reward (10k INR)")
left.legend()
right.bar(results.keys(), [r[:, 0].mean() for r in results.values()], yerr=[r[:, 0].std() for r in results.values()])
right.set(title="Total cost on 2025, real forecasts", ylabel="Lakh INR")
fig.tight_layout()
fig.savefig("v3/results_v3.png", dpi=150)
