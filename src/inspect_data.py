from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data" / "raw"

fedfunds = pd.read_csv(DATA_RAW / "fred_fedfunds.csv")
cpi = pd.read_csv(DATA_RAW / "fred_cpi.csv")
treasury_10y = pd.read_csv(DATA_RAW / "fred_10y_treasury.csv")
housing = pd.read_csv(DATA_RAW / "fred_housing_starts.csv")

print(fedfunds.head())
print(fedfunds.info())
