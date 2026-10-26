"""Real daily weather, dust and day-ahead forecasts for Jaisalmer, Rajasthan (Open-Meteo, cached to CSV)."""
from pathlib import Path

import pandas as pd
import requests

LAT, LON = 26.91, 70.91
START, END = "2022-08-05", "2025-12-31"  # CAMS dust starts 2022-08-04
CACHE = Path("data/jaisalmer.csv")

WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
DUST_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
FORECAST_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
PARAMS = {"latitude": LAT, "longitude": LON, "start_date": START, "end_date": END, "timezone": "Asia/Kolkata"}


def get(url, **params):
    r = requests.get(url, params={**PARAMS, **params}, timeout=60)
    r.raise_for_status()
    return r.json()


def load_weather():
    """Daily columns: sun (kWh/m2), rain (mm), dust (ug/m3 daily mean), and sun_fc, rain_fc as forecast 24 h earlier.

    The forecast archive starts on 2024-01-20; observed values fill the days before it.
    """
    if CACHE.exists():
        return pd.read_csv(CACHE, index_col="time", parse_dates=True)

    daily = get(WEATHER_URL, daily="shortwave_radiation_sum,precipitation_sum")["daily"]
    hourly = get(DUST_URL, hourly="dust")["hourly"]
    forecast = get(FORECAST_URL, hourly="shortwave_radiation_previous_day1,precipitation_previous_day1")["hourly"]

    df = pd.DataFrame(daily, index=pd.to_datetime(daily.pop("time")))
    dust = pd.DataFrame(hourly, index=pd.to_datetime(hourly.pop("time")))
    fc = pd.DataFrame(forecast, index=pd.to_datetime(forecast.pop("time"))).resample("D").sum(min_count=24)

    df = pd.DataFrame({
        "sun": df["shortwave_radiation_sum"] / 3.6,  # MJ/m2 -> kWh/m2
        "rain": df["precipitation_sum"],
        "dust": dust["dust"].resample("D").mean(),
    }).ffill()
    df["sun_fc"] = (fc["shortwave_radiation_previous_day1"] / 1000).fillna(df["sun"])  # Wh/m2 -> kWh/m2
    df["rain_fc"] = fc["precipitation_previous_day1"].fillna(df["rain"])

    CACHE.parent.mkdir(exist_ok=True)
    df.to_csv(CACHE)
    return df


if __name__ == "__main__":
    df = load_weather().loc["2024-01-20":]  # days with a real archived forecast
    nrmse = ((df.sun - df.sun_fc) ** 2).mean() ** 0.5 / df.sun.mean()
    heavy, heavy_fc = df.rain >= 6, df.rain_fc >= 6
    print(f"{len(df)} days. Irradiance forecast: correlation {df.sun.corr(df.sun_fc):.2f}, nRMSE {nrmse:.0%}")
    print(f"Heavy rain forecast: caught {(heavy & heavy_fc).sum()} of {heavy.sum()} days, {(heavy_fc & ~heavy).sum()} false alarms")
