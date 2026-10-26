"""v1 weather + real day-ahead forecasts (Open-Meteo Previous Runs API, archived from 2024)."""
from pathlib import Path

import pandas as pd

from data import get, load_weather

FORECAST_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
CACHE = Path("data/jaisalmer_forecast.csv")


def load_weather_v2():
    """Adds sun_fc (kWh/m2) and rain_fc (mm) as forecast 24 h earlier; observed values fill the days before the archive."""
    if CACHE.exists():
        return pd.read_csv(CACHE, index_col="time", parse_dates=True)

    hourly = get(FORECAST_URL, hourly="shortwave_radiation_previous_day1,precipitation_previous_day1")["hourly"]
    fc = pd.DataFrame(hourly, index=pd.to_datetime(hourly.pop("time"))).resample("D").sum(min_count=24)

    df = load_weather()
    df["sun_fc"] = (fc["shortwave_radiation_previous_day1"] / 1000).fillna(df["sun"])  # Wh/m2 -> kWh/m2
    df["rain_fc"] = fc["precipitation_previous_day1"].fillna(df["rain"])
    df.to_csv(CACHE)
    return df
