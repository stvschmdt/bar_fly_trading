"""
Create a 1-month sample dataframe from all 37 all_data_*.csv files.

Outputs: unit_tests/sample_data_1month.csv
"""
import glob
import pandas as pd

DATA_GLOB = "./all_data_*.csv"
OUTPUT_PATH = "unit_tests/sample_data_1month.csv"
CUTOFF_DATE = "2026-01-03"  # Keep only data from this date onward (~1 month)

files = sorted(glob.glob(DATA_GLOB))
print(f"Found {len(files)} all_data files")

frames = []
for f in files:
    df = pd.read_csv(f, parse_dates=["date"])
    df = df[df["date"] >= pd.to_datetime(CUTOFF_DATE)]
    frames.append(df)
    print(f"  {f}: {len(df)} rows after filter")

combined = pd.concat(frames, ignore_index=True)
combined.sort_values(["symbol", "date"], inplace=True)
combined.to_csv(OUTPUT_PATH, index=False)

n_symbols = combined["symbol"].nunique()
n_dates = combined["date"].nunique()
print(f"\nSaved {OUTPUT_PATH}: {len(combined)} rows, {n_symbols} symbols, {n_dates} dates")
print(f"Date range: {combined['date'].min()} to {combined['date'].max()}")
