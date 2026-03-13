"""
Generate a synthetic daily climate dataset for testing climate_year_selection_tool.

Output: data/synthetic_climate_data.csv
Columns: Date, temp [°C], wind [m/s], solar [W/m²]
Period:  1900-01-01 – 1999-12-31  (100 years, daily)

Seasonal patterns included so year selection is non-trivial:
  temp  – cold winters, warm summers + interannual variability + noise
  wind  – stronger in winter, weaker in summer + interannual variability + noise
  solar – near-zero in winter, high in summer + interannual variability + noise
"""

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

OUTPUT_FILE = Path(__file__).parent.parent / "data" / "synthetic_climate_data.csv"

rng = np.random.default_rng(42)

dates = pd.date_range("1900-01-01", "1999-12-31", freq="D")
n = len(dates)

# Day-of-year angle for seasonal cycle (0 at Jan 1, 2π at Dec 31)
doy = dates.day_of_year.to_numpy()
angle = 2 * np.pi * (doy - 1) / 365.0

# --- interannual variability: one random amplitude per year ---
years = dates.year.to_numpy()
unique_years = np.unique(years)
year_to_idx = {y: i for i, y in enumerate(unique_years)}
yr_idx = np.array([year_to_idx[y] for y in years])

n_years = len(unique_years)
ia_temp = rng.normal(0, 1.5, n_years)[yr_idx]  # ±1.5 °C interannual shift
ia_wind = rng.normal(0, 0.5, n_years)[yr_idx]  # ±0.5 m/s interannual shift
ia_solar = rng.normal(0, 15.0, n_years)[yr_idx]  # ±15 W/m² interannual shift

# Temperature [°C]: mean 10, amplitude 12, interannual + daily noise
temp = (
    10
    - 12 * np.cos(angle)  # seasonal: minimum ~Jan, maximum ~Jul
    + ia_temp
    + rng.normal(0, 2.5, n)  # daily noise
)

# Wind speed [m/s]: mean 7, stronger in winter
wind = np.clip(
    7.0
    + 2.0 * np.cos(angle)  # seasonal: peak ~Jan, trough ~Jul
    + ia_wind
    + rng.normal(0, 1.5, n),  # daily noise
    0,
    None,
)

# Solar radiation [W/m²]: daily mean GHI, strongly seasonal
solar = np.clip(
    120
    - 110 * np.cos(angle)  # seasonal: peak ~Jun (~230 W/m²), trough ~Dec (~10 W/m²)
    + ia_solar
    + rng.normal(0, 25, n),  # daily noise (cloud cover variability)
    0,
    None,
)

df = pd.DataFrame({"Date": dates, "temp": temp, "wind": wind, "solar": solar})
df.to_csv(OUTPUT_FILE, index=False)
logger.info((f"Saved {len(df):,} rows to {OUTPUT_FILE}"))
logger.info(df.head())
logger.info(df.describe().round(3))
