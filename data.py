"""Real daily weather + dust for Jaisalmer, Rajasthan (Open-Meteo, cached to CSV)."""
from pathlib import Path

import pandas as pd
import requests

LAT, LON = 26.91, 70.91
START, END = "2022-08-05", "2025-12-31"  # CAMS dust starts 2022-08-04
CACHE = Path("data/jaisalmer.csv")

WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
DUST_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
PARAMS = {"latitude": LAT, "longitude": LON, "start_date": START, "end_date": END, "timezone": "Asia/Kolkata"}


def get(url, **params):
    r = requests.get(url, params={**PARAMS, **params}, timeout=60)
    r.raise_for_status()
    return r.json()


def load_weather():
    """Daily columns: sun (kWh/m2), rain (mm), dust (ug/m3 daily mean)."""
    if CACHE.exists():
        return pd.read_csv(CACHE, index_col="time", parse_dates=True)

    daily = get(WEATHER_URL, daily="shortwave_radiation_sum,precipitation_sum")["daily"]
    hourly = get(DUST_URL, hourly="dust")["hourly"]

    df = pd.DataFrame(daily, index=pd.to_datetime(daily.pop("time")))
    dust = pd.DataFrame(hourly, index=pd.to_datetime(hourly.pop("time")))

    df = pd.DataFrame({
        "sun": df["shortwave_radiation_sum"] / 3.6,  # MJ/m2 -> kWh/m2
        "rain": df["precipitation_sum"],
        "dust": dust["dust"].resample("D").mean(),
    }).ffill()

    CACHE.parent.mkdir(exist_ok=True)
    df.to_csv(CACHE)
    return df


if __name__ == "__main__":
    print(load_weather().describe())
