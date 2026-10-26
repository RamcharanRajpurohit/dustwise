"""Compare RL algorithms on 2025 with real forecasts, then the drought study."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from env import BLOCKS, REFILL_L
from v2.env_v2 import DustWiseV2
from v2.forecast import load_weather_v2
from v2.train import ALGOS, name

SEEDS = [0, 1, 2]
REFILLS = [250, 500, 1000, 2000]  # litres/day for the drought study
DROUGHT_ALGOS = ["dqn", "maskppo"]


def run(policy, weather, refill=REFILL_L):
    """Plays the whole test year; returns cost (lakh INR), energy lost (MWh), water (kL)."""
    env = DustWiseV2(weather, days=len(weather) - 1, refill=refill)
    obs, _ = env.reset(seed=0)
    totals, done = np.zeros(3), False
    while not done:
        obs, _, _, done, info = env.step(policy(obs, env))
        totals += info["cost"], info["lost_kwh"], info["water"]
    return totals / (1e5, 1e3, 1e3)


def agent(algo, seed, refill=REFILL_L):
    model = ALGOS[algo][0].load(f"v2/models/{name(algo, seed, refill)}")
    if algo == "maskppo":
        return lambda obs, env: model.predict(obs, action_masks=env.action_masks(), deterministic=True)[0]
    return lambda obs, env: model.predict(obs, deterministic=True)[0]


def greedy_dry(obs, env):
    """Brush the block with most loose dust; skip if heavy rain is forecast."""
    return 0 if obs[-2] >= 1 else 1 + np.argmax(obs[:BLOCKS])


test = load_weather_v2().loc["2025"]
greedy = run(greedy_dry, test)
results = {algo: np.array([run(agent(algo, s), test) for s in SEEDS]) for algo in ALGOS}

print(f"{'Policy':10} {'Cost (lakh INR)':>16} {'Lost (MWh)':>11} {'Water (kL)':>11}")
print(f"{'greedy':10} {greedy[0]:16.2f} {greedy[1]:11.0f} {greedy[2]:11.0f}")
for algo, r in results.items():
    print(f"{algo:10} {r[:, 0].mean():9.2f} ± {r[:, 0].std():4.2f} {r[:, 1].mean():11.0f} {r[:, 2].mean():11.0f}")

drought = {algo: np.array([run(agent(algo, 0, w), test, w) for w in REFILLS]) for algo in DROUGHT_ALGOS}
print(f"\nDrought study (seed 0), greedy cost {greedy[0]:.2f}")
print(f"{'L/day':>6} " + " ".join(f"{a + ' cost':>13} {a + ' kL':>11}" for a in DROUGHT_ALGOS))
for i, w in enumerate(REFILLS):
    print(f"{w:6} " + " ".join(f"{drought[a][i, 0]:13.2f} {drought[a][i, 2]:11.0f}" for a in DROUGHT_ALGOS))

fig, ax = plt.subplots(2, 2, figsize=(11, 8))
for algo in ALGOS:
    log = pd.read_csv(f"v2/logs/{name(algo, 0, REFILL_L)}.monitor.csv", skiprows=1)
    ax[0, 0].plot(log.l.cumsum(), log.r.rolling(200).mean(), label=algo)
ax[0, 0].set(title="Training (seed 0, 200-episode mean)", xlabel="Timesteps", ylabel="Episode reward (10k INR)")
ax[0, 0].legend()

ax[0, 1].bar(results.keys(), [r[:, 0].mean() for r in results.values()], yerr=[r[:, 0].std() for r in results.values()])
ax[0, 1].axhline(greedy[0], color="gray", ls="--", label="greedy dry")
ax[0, 1].set(title="Total cost on 2025, real forecasts (3 seeds)", ylabel="Lakh INR")
ax[0, 1].legend()

for algo in DROUGHT_ALGOS:
    ax[1, 0].plot(REFILLS, drought[algo][:, 0], "o-", label=algo)
    ax[1, 1].plot(REFILLS, drought[algo][:, 2], "o-", label=algo)
ax[1, 0].axhline(greedy[0], color="gray", ls="--", label="greedy dry")
ax[1, 0].set(title="Drought: total cost", xlabel="Water delivered (L/day)", ylabel="Lakh INR")
ax[1, 1].set(title="Drought: water used", xlabel="Water delivered (L/day)", ylabel="kL in 2025")
ax[1, 0].legend()
ax[1, 1].legend()
fig.tight_layout()
fig.savefig("v2/results_v2.png", dpi=150)
