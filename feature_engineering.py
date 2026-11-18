import pandas as pd
import numpy as np
from pathlib import Path
path = Path(__file__).parent
raw_file = path/ "nasa_power_dhaka_10yr.csv"
output_file = path/ "dhaka_features_engineered.csv"
df = pd.read_csv(raw_file, parse_dates=['date']) # treat date as a date object
df = df.sort_values('date').reset_index(drop=True) # so i can perform math on them later

print(df[['date', 'PRECTOTCORR']].head(3).to_string(index=False))# print the first 3 rows of date and PRECTOTCORR
# --- Cumulative rainfall windows ---
# This is the core flood signal — sustained rainfall, not just one heavy day
df['rain_3day']  = df['PRECTOTCORR'].rolling(3).sum()
df['rain_7day']  = df['PRECTOTCORR'].rolling(7).sum()
df['rain_14day'] = df['PRECTOTCORR'].rolling(14).sum()
df['rain_30day'] = df['PRECTOTCORR'].rolling(30).sum()

# --- Antecedent moisture condition ---
# Classic hydrology concept — how wet was the catchment before today's rain
# High evapotranspiration drains moisture, low means soil stays saturated
df['moisture_deficit'] = df['rain_7day'] - (df['EVPTRNS'].rolling(7).sum())

# --- Rainfall anomaly ---
# Is today's rain unusually heavy for this time of year?
df['rain_30day_avg'] = df['PRECTOTCORR'].rolling(30).mean()
df['rain_anomaly'] = df['PRECTOTCORR'] - df['rain_30day_avg']

# --- Season encoding ---
# Bangladesh monsoon is June-September — the flood season
df['month'] = df['date'].dt.month # extracts the month as a number (1-12) from the date
df['is_monsoon'] = df['month'].isin([6, 7, 8, 9]).astype(int) # converts T or F to (0 or 1)
df['day_of_year'] = df['date'].dt.dayofyear #(gives a number 1-365)

# Cyclical encoding of month — better than raw numbers for ML
# December and January are adjacent in reality but 12 and 1 are far apart numerically
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# --- Humidity trend ---
# Rising humidity signals approaching monsoon or storm season
df['rh_3day_avg'] = df['RH2M'].rolling(3).mean()
df['rh_trend'] = df['RH2M'] - df['RH2M'].rolling(7).mean()

# --- Temperature trend ---
df['temp_3day_avg'] = df['T2M'].rolling(3).mean()

# Drop rows with NaN from rolling windows
df = df.dropna().reset_index(drop=True)

print('Final shape:', df.shape)
print('Columns:', list(df.columns))
print(df.head(3).to_string()) # print the full row without truncation

df.to_csv(output_file, index=False) # index=False - don't write the row numbers as 
print('\nSaved to dhaka_features_engineered.csv')  # a column in the file, just the data
