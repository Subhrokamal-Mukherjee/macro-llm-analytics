from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "data" / "raw"
CLEANED = PROJECT_ROOT / "data" / "cleaned"

CLEANED.mkdir(exist_ok=True)
fedfunds = pd.read_csv(RAW / "fred_fedfunds.csv")

fedfunds_clean = (
    fedfunds
    .rename(columns={
        "observation_date": "date",
        "FEDFUNDS": "fed_funds_rate"
    })
)

fedfunds_clean["date"] = pd.to_datetime(fedfunds_clean["date"])
fedfunds_clean = fedfunds_clean.sort_values("date")

fedfunds_clean.to_csv(CLEANED / "fed_funds_rate.csv", index=False)

cpi = pd.read_csv(RAW / "fred_cpi.csv")

cpi_clean = (
    cpi
    .rename(columns={
        "observation_date": "date",
        "CPIAUCSL": "cpi_index"
    })
)

cpi_clean["date"] = pd.to_datetime(cpi_clean["date"])
cpi_clean = cpi_clean.sort_values("date")

cpi_clean.to_csv(CLEANED / "cpi_index.csv", index=False)

treasury = pd.read_csv(RAW / "fred_10y_treasury.csv")

treasury_clean = (
    treasury
    .rename(columns={
        "observation_date": "date",
        "DGS10": "treasury_10y_rate"
    })
)

treasury_clean["date"] = pd.to_datetime(treasury_clean["date"])

# Convert "." to NaN if present
treasury_clean["treasury_10y_rate"] = pd.to_numeric(
    treasury_clean["treasury_10y_rate"],
    errors="coerce"
)

treasury_clean = treasury_clean.sort_values("date")

treasury_clean.to_csv(CLEANED / "treasury_10y_rate.csv", index=False)

housing = pd.read_csv(RAW / "fred_housing_starts.csv")

housing_clean = (
    housing
    .rename(columns={
        "observation_date": "date",
        "HOUST": "housing_starts"
    })
)

housing_clean["date"] = pd.to_datetime(housing_clean["date"])
housing_clean = housing_clean.sort_values("date")

housing_clean.to_csv(CLEANED / "housing_starts.csv", index=False)
