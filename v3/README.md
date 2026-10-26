# DustWise v3

v3 is the best agent from a validation study. The study trained on 2022–23 and scored each change on 2024 with real forecasts. The greedy baseline cost 30.86 lakh INR on that year. 2025 was not used to choose anything.

| Change (MaskablePPO, seed 0) | Cost on 2024 (lakh INR) at 500k → 1M steps |
|---|---|
| v2 MaskablePPO | 19.16 → 18.22 |
| + season (day of year, sin/cos) | 18.00 → 18.03 |
| **+ season + entropy bonus 0.01** | **17.67 → 17.55** |
| + season + tuned PPO settings from An (2026)¹ | 18.50 → 17.70 |
| + season + 2- and 3-day rain forecasts (as suggested by Terhag et al. 2019)² | 17.93 → 17.90 |
| DQN (for reference) | 22.17 → 27.19 |

So v3 is v2's MaskablePPO plus the season input and an entropy bonus of 0.01, trained for 1M steps. The extra rain forecasts gained only about 0.1 lakh, which matches the small forecast gain reported by Terhag et al. That gain wasn't worth another API call, so they were left out. DQN became unstable with longer training.

¹ H. An, "Reinforcement learning-based dynamic cleaning scheduling framework for solar energy system," arXiv:2603.07518, 2026.
² F. Terhag et al., "Optimization of cleaning strategies based on ANN algorithms assessing the benefit of soiling rate forecasts," AIP Conf. Proc. 2126, 220005, 2019.

## Run (from the repo root)

```bash
python -m v3.train 0        # seeds 0, 1, 2; about 1 h each on 1 CPU thread
python -m v3.evaluate       # table + v3/results_v3.png
```

## Results (2025 test year, real forecasts, 3 seeds)

| Policy | Cost (lakh INR) | Energy lost (MWh) | Water (kL) |
|---|---|---|---|
| Greedy dry (baseline) | 19.63 | 735 | 0 |
| v2 MaskablePPO | 14.18 ± 0.19 | 513 | 204 |
| **v3 MaskablePPO** | **13.00 ± 0.04** | **467** | **175** |

v3 is 34% cheaper than greedy and 8% cheaper than v2. It also uses 14% less water than v2 and is the most consistent across seeds.

![v3 results](results_v3.png)
