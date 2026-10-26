# DustWise

Dust cuts solar output in Rajasthan, but washing panels uses scarce water. DustWise trains an RL agent (**MaskablePPO**) that decides each day which of 10 panel blocks to clean, and whether to dry-brush or water-wash it. It uses real weather, dust and day-ahead forecast data for Jaisalmer.

On the unseen 2025 test year the agent costs **13.00 ± 0.04 lakh INR**, which is **34% cheaper** than the best rule-based baseline (19.62).

## RL formulation

| | |
|---|---|
| **State** (26) | loose dust loss of each block ×10, cemented dust loss of each block ×10, water tank level, day-ahead forecast of irradiance and rain, today's dust level, season (sin/cos of day of year) |
| **Action** (21) | `0` idle, `1–10` dry-brush block *i*, `11–20` water-wash block *i* (one crew, one block a day). Water washes are masked out when the tank holds less than one wash. |
| **Reward** | `−(revenue lost to soiling + labour + water) / 10,000 INR` per day |
| **Episode** | 90 random consecutive days from the training years |

**Dynamics (one step = one day):**
1. The crew cleans in the morning. A dry brush removes 90% of the loose dust. A water wash removes all the dust but needs 3,640 L from the tank.
2. The panels generate power with their remaining soiling loss, using the weather that actually happened.
3. The day's weather then updates the dust:
   - Rain of 6 mm or more washes every block clean.
   - Lighter rain cements the loose dust, and only water can remove cemented dust.
   - New dust settles in proportion to that day's CAMS dust level.
4. The tank refills by 1,000 L a day, up to 10,000 L.

**Algorithm:** MaskablePPO (sb3-contrib) with an MLP of 128×128, γ = 0.9 and an entropy bonus of 0.01, trained for 1M steps on 2022-08 to 2024. The test year, 2025, is never seen during training.

**Baselines:** no cleaning; daily rotation with a dry brush; daily rotation with a water wash; greedy (brush the dustiest block, but skip the day if heavy rain is forecast).

## Results (2025, unseen, real forecasts)

| Policy | Cost (lakh INR) | Energy lost (MWh) | Water (kL) |
|---|---|---|---|
| No cleaning | 47.79 | 1912 | 0 |
| Rotation wash | 25.29 | 961 | 371 |
| Rotation dry | 19.62 | 732 | 0 |
| Greedy dry | 19.63 | 735 | 0 |
| **MaskablePPO (3 seeds)** | **13.00 ± 0.04** | **467** | **175** |

The agent loses 36% less energy than the best baseline and still uses less than half the water of the wash rotation. It idles on about 111 days, brushes on about 205 and washes on about 48. 96% of its washes go to blocks with cemented dust, the only case a brush cannot fix.

![results](results.png)

## What we tried, in order

1. **DQN with perfect forecasts.** The agent saw the weather that actually happened. It beat greedy, 18.01 against 19.41 lakh INR.
2. **Real day-ahead forecasts.** The agent sees the forecast issued 24 h earlier. On 2024–25 it has irradiance correlation 0.89 and caught only 14 of 25 heavy-rain days, so the agent must act under real uncertainty.
3. **Five algorithms**, 3 seeds each, 500k steps: DQN 17.30, QR-DQN 18.80, PPO 15.39, A2C 16.22 lakh INR. The on-policy methods were better and more stable.
4. **Action masking (MaskablePPO): 14.18.** Without masks, DQN wasted crew-days trying to wash with an empty tank, badly so when water was scarce.
5. **Tuning on a 2024 validation year** (trained on 2022–23). Adding the season input and an entropy bonus helped most. Tuned PPO settings from An (2026)¹ and 2–3-day rain forecasts gained less, so they were left out. This is the final agent, 13.00.

The code for the earlier steps is in the git history.

¹ H. An, "Reinforcement learning-based dynamic cleaning scheduling framework for solar energy system," arXiv:2603.07518, 2026.

## Run

Python 3.12. Run from the repo root. The data is cached in `data/` and the trained agents are in `models/`, so `evaluate.py` works straight away.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python data.py        # forecast accuracy (downloads the data if data/jaisalmer.csv is missing)
python train.py 0     # seeds 0, 1, 2; about 40 min each on CPU -> models/maskppo_s<seed>.zip
python evaluate.py    # baselines vs agent on 2025 -> table + results.png
```

## Files

| File | Purpose |
|---|---|
| `data.py` | fetch and cache the Open-Meteo weather, dust and forecast data |
| `env.py` | `DustWiseEnv` Gymnasium environment, with action masks |
| `train.py` | MaskablePPO training (logs in `logs/`, models in `models/`) |
| `evaluate.py` | baselines vs the agent on 2025, plots |
| `report/` | 2-page report (`report.tex`, built to `report.pdf` with `pdflatex`) |

## Constants and sources

| Value | Source |
|---|---|
| Soiling 0.4 %/day | Valerino et al., *Solar Energy* 2020 (Gandhinagar, 0.37 %/day), https://doi.org/10.1016/j.solener.2020.03.118 |
| Rain ≥ 6 mm cleans the panels; soiling capped at 30% | pvlib Kimber model, https://pvlib-python.readthedocs.io/en/v0.9.3/reference/generated/pvlib.soiling.kimber.html |
| Light rain cements dust; dry brushing leaves cemented particles | Valerino 2020; https://doi.org/10.1016/J.RENENE.2025.124815 |
| Water 2 L per module, ₹240/kL; labour ₹0.20 per module | https://www.pv-magazine-india.com/2018/10/02/study-high-risk-through-water-use-in-solar-sector/ , BHEL tender https://tenders.bhel.com/sites/default/files/Corrected%20Pages.pdf |
| Tariff ₹2.5/kWh, PR 0.78 | https://pv-magazine-india.com/2024/02/09/sjvn-allocates-1-5-gw-of-solar-at-0-030-kwh |

## Data

Daily shortwave radiation and precipitation come from the [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) (ERA5). Hourly dust comes from the [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api) (CAMS) and is averaged to daily values. Day-ahead forecasts come from the [Open-Meteo Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api), archived from 2024-01-20. All are for Jaisalmer (26.91 N, 70.91 E), 2022-08-05 to 2025-12-31, and are cached in `data/jaisalmer.csv`.

Weather data by [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0). Dust data from the Copernicus Atmosphere Monitoring Service (CAMS).

## Limitations

- Before 2024-01-20 there is no forecast archive, so the training years use the observed weather as the forecast. Only 2024 and 2025 have real forecasts.
- The dust input is the CAMS value for the same day, not a forecast, so the agent effectively has a perfect one-day dust forecast. The greedy baseline gets the same information.
- The soiling and cleaning constants come from nearby Indian and desert studies, not measurements from Jaisalmer itself.
- MaskablePPO itself was picked by comparing algorithms on 2025; only the final additions were chosen on the 2024 validation year.
