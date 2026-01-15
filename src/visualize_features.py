from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES = PROJECT_ROOT / "data" / "features"

df = pd.read_csv(FEATURES / "macro_features.csv", parse_dates=["date"])

plt.figure()
plt.plot(df["date"], df["fed_funds_rate"], label="Fed Funds Rate")
plt.plot(df["date"], df["inflation_yoy"], label="Inflation (YoY)")
plt.legend()
plt.title("Policy Rate vs Inflation")
plt.show()
plt.figure()
plt.plot(df["date"], df["yield_spread"])
plt.axhline(0)
plt.title("10Y – Fed Funds Yield Spread")
plt.show()
plt.figure()
plt.plot(df["date"], df["real_policy_rate"])
plt.axhline(0)
plt.title("Real Policy Rate")
plt.show()
plt.figure()
plt.plot(df["date"], df["housing_starts"], label="Housing Starts")
plt.plot(df["date"], df["fed_funds_rate"], label="Fed Funds Rate")
plt.legend()
plt.title("Housing Activity vs Interest Rates")
plt.show()
