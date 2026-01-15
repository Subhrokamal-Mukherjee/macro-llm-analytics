from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEANED = PROJECT_ROOT / "data" / "cleaned"
FEATURES = PROJECT_ROOT / "data" / "features"

FEATURES.mkdir(exist_ok=True)

fed = pd.read_csv(CLEANED / "fed_funds_rate.csv", parse_dates=["date"])
cpi = pd.read_csv(CLEANED / "cpi_index.csv", parse_dates=["date"])
treasury = pd.read_csv(CLEANED / "treasury_10y_rate.csv", parse_dates=["date"])
housing = pd.read_csv(CLEANED / "housing_starts.csv", parse_dates=["date"])

treasury_monthly = (
    treasury
    .set_index("date")
    .resample("MS")
    .mean()
    .reset_index()
)

macro = (
    fed
    .merge(treasury_monthly, on="date", how="left")
    .merge(cpi, on="date", how="left")
    .merge(housing, on="date", how="left")
)

macro["inflation_yoy"] = (
    macro["cpi_index"]
    .pct_change(periods=12,fill_method=None) * 100
)

macro["yield_spread"] = (
    macro["treasury_10y_rate"] - macro["fed_funds_rate"]
)

macro["real_policy_rate"] = (
    macro["fed_funds_rate"] - macro["inflation_yoy"]
)

features = macro[[
    "date",
    "fed_funds_rate",
    "treasury_10y_rate",
    "yield_spread",
    "inflation_yoy",
    "real_policy_rate",
    "housing_starts"
]]

features.to_csv(FEATURES / "macro_features.csv", index=False)
