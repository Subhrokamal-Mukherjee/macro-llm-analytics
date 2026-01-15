from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES = PROJECT_ROOT / "data" / "features"

# Load features
df = pd.read_csv(
    FEATURES / "macro_features.csv",
    parse_dates=["date"]
)

# Select signals for regime detection
signal_cols = [
    "real_policy_rate",
    "inflation_yoy",
    "yield_spread"
]

signals = df[signal_cols].dropna()

# Scale signals (distance-based model)
scaler = StandardScaler()
X = scaler.fit_transform(signals)

# Fit KMeans (3 macro regimes)
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

signals["regime"] = kmeans.fit_predict(X)

# Preserve date for inspection / export
signals["date"] = df.loc[signals.index, "date"]

# Save regime-only table
signals.to_csv(
    FEATURES / "regime_signals.csv",
    index=False
)

# Merge regimes back into main dataset
df_with_regime = df.merge(
    signals[["regime"]],
    left_index=True,
    right_index=True,
    how="left"
)

df_with_regime.to_csv(
    FEATURES / "macro_features_with_regime.csv",
    index=False
)

centroids = scaler.inverse_transform(kmeans.cluster_centers_)
centroid_df = pd.DataFrame(
    centroids,
    columns=signal_cols
)

print("\nRegime centroids:")
print(centroid_df.round(2))

print("\nRegime counts:")
print(signals["regime"].value_counts().sort_index())
