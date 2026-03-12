"""
F1 Tire Degradation Predictor — Data Processing Engine
=======================================================
Loads race sessions via FastF1, extracts high-frequency telemetry,
cleans it, and builds sliding-window sequences for the Bi-LSTM.
"""

import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

import fastf1
from fastf1 import plotting
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── FastF1 Cache ──────────────────────────────────────────────────────────────
CACHE_DIR = Path("./f1_cache")
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

# ── Constants ─────────────────────────────────────────────────────────────────
TELEMETRY_FEATURES = ["Speed", "Throttle", "Brake", "RPM", "nGear", "Distance"]
LAP_FEATURES       = ["LapTime_s", "Sector1Time_s", "Sector2Time_s", "Sector3Time_s",
                      "TyreLife", "CompoundEncoded"]
COMPOUND_MAP       = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTERMEDIATE": 3, "WET": 4}
SEQUENCE_LENGTH    = 10   # look-back window (laps)
FORECAST_HORIZON   = 5    # predict next N laps


# ─────────────────────────────────────────────────────────────────────────────
# 1. SESSION LOADER
# ─────────────────────────────────────────────────────────────────────────────
class SessionLoader:
    """Wraps FastF1 session loading with robust error handling."""

    def __init__(self, year: int, grand_prix: str, session_type: str = "R"):
        self.year         = year
        self.grand_prix   = grand_prix
        self.session_type = session_type
        self.session       = None

    def load(self) -> fastf1.core.Session:
        print(f"[SessionLoader] Loading {self.year} {self.grand_prix} — {self.session_type}")
        self.session = fastf1.get_session(self.year, self.grand_prix, self.session_type)
        self.session.load(telemetry=True, laps=True, weather=True)
        print(f"[SessionLoader] ✓ Loaded. Drivers: {list(self.session.drivers)}")
        return self.session


# ─────────────────────────────────────────────────────────────────────────────
# 2. TELEMETRY EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────
class TelemetryExtractor:
    """
    Extracts per-lap aggregated telemetry + lap metadata for one driver.
    Aggregation: mean of high-frequency signals per lap (avoids sequence mismatch).
    """

    def __init__(self, session: fastf1.core.Session, driver: str):
        self.session = session
        self.driver  = driver

    def _timedelta_to_seconds(self, value) -> float:
        """Convert a single FastF1 lap time value to float seconds robustly."""
        try:
            if pd.isnull(value):
                return np.nan
        except (TypeError, ValueError):
            pass
        if hasattr(value, "total_seconds"):
            return value.total_seconds()
        if isinstance(value, np.timedelta64):
            return float(value / np.timedelta64(1, 's'))
        if isinstance(value, np.datetime64):
            return float(value.astype('datetime64[ns]').astype(np.int64)) / 1e9
        try:
            return pd.Timedelta(value).total_seconds()
        except Exception:
            return np.nan

    def extract(self) -> pd.DataFrame:
        laps = self.session.laps.pick_driver(self.driver).copy()
        laps = laps[laps["IsPersonalBest"].notna() | laps["LapTime"].notna()]

        records = []
        for _, lap in laps.iterrows():
            try:
                tel = lap.get_telemetry()
            except Exception:
                continue

            # ── Telemetry aggregates ──────────────────────────────────────────
            row = {feat: tel[feat].mean() for feat in TELEMETRY_FEATURES if feat in tel.columns}

            # ── Lap metadata ─────────────────────────────────────────────────
            row["LapNumber"]     = lap["LapNumber"]
            row["LapTime_s"]     = self._timedelta_to_seconds(lap["LapTime"])
            row["Sector1Time_s"] = self._timedelta_to_seconds(lap["Sector1Time"])
            row["Sector2Time_s"] = self._timedelta_to_seconds(lap["Sector2Time"])
            row["Sector3Time_s"] = self._timedelta_to_seconds(lap["Sector3Time"])
            row["TyreLife"]    = lap.get("TyreLife", np.nan)
            row["Compound"]    = lap.get("Compound", "UNKNOWN")
            row["CompoundEncoded"] = COMPOUND_MAP.get(str(lap.get("Compound", "")), -1)
            row["Driver"]      = self.driver

            records.append(row)

        df = pd.DataFrame(records).sort_values("LapNumber").reset_index(drop=True)
        print(f"[TelemetryExtractor] Driver {self.driver}: {len(df)} laps extracted")
        return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. DATA CLEANER
# ─────────────────────────────────────────────────────────────────────────────
class DataCleaner:
    """
    Handles all data quality issues common in FastF1 output:
      • Safety Car / VSC laps (anomalously slow)
      • Pit-in / Pit-out laps
      • Missing telemetry (NaN imputation)
      • Outlier lap times (IQR fence)
    """

    PIT_LAP_SPEED_THRESHOLD = 60.0   # km/h avg speed — almost certainly pit lap
    LAP_TIME_IQR_MULTIPLIER = 2.5

    def __init__(self, df: pd.DataFrame, mark_only: bool = False):
        self.df        = df.copy()
        self.mark_only = mark_only   # if True, add flag columns instead of dropping

    def _flag_pit_laps(self) -> None:
        self.df["IsPitLap"] = (
            self.df.get("Speed", pd.Series(np.inf, index=self.df.index)) < self.PIT_LAP_SPEED_THRESHOLD
        )

    def _flag_outlier_lap_times(self) -> None:
        q1 = self.df["LapTime_s"].quantile(0.25)
        q3 = self.df["LapTime_s"].quantile(0.75)
        iqr = q3 - q1
        upper = q3 + self.LAP_TIME_IQR_MULTIPLIER * iqr
        lower = q1 - self.LAP_TIME_IQR_MULTIPLIER * iqr
        self.df["IsOutlier"] = ~self.df["LapTime_s"].between(lower, upper)

    def _impute_missing(self) -> None:
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        # Forward-fill first (carry last known value), then back-fill edges
        self.df[numeric_cols] = (
            self.df[numeric_cols]
            .fillna(method="ffill")
            .fillna(method="bfill")
        )

    def _drop_invalid_compounds(self) -> None:
        self.df = self.df[self.df["CompoundEncoded"] >= 0]

    def clean(self) -> pd.DataFrame:
        initial_len = len(self.df)
        self._flag_pit_laps()
        self._flag_outlier_lap_times()
        self._impute_missing()
        self._drop_invalid_compounds()

        if not self.mark_only:
            self.df = self.df[~self.df["IsPitLap"] & ~self.df["IsOutlier"]]
            self.df = self.df.drop(columns=["IsPitLap", "IsOutlier"], errors="ignore")

        self.df = self.df.reset_index(drop=True)
        print(f"[DataCleaner] {initial_len} → {len(self.df)} laps after cleaning")
        return self.df


# ─────────────────────────────────────────────────────────────────────────────
# 4. SLIDING-WINDOW SEQUENCE BUILDER
# ─────────────────────────────────────────────────────────────────────────────
class SequenceBuilder:
    """
    Converts a cleaned lap-level DataFrame into (X, y) arrays for the Bi-LSTM.

    X shape: (n_samples, SEQUENCE_LENGTH, n_features)
    y shape: (n_samples, FORECAST_HORIZON)  — lap-time deltas (seconds vs lap 1)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[list] = None,
        seq_len: int = SEQUENCE_LENGTH,
        horizon: int = FORECAST_HORIZON,
    ):
        self.df           = df
        self.feature_cols = feature_cols or (TELEMETRY_FEATURES + LAP_FEATURES)
        self.seq_len      = seq_len
        self.horizon      = horizon
        self.scaler       = StandardScaler()

    def _compute_lap_delta(self) -> pd.Series:
        """Lap time delta relative to the driver's first clean lap."""
        baseline = self.df["LapTime_s"].iloc[0]
        return self.df["LapTime_s"] - baseline

    def build(self) -> tuple[np.ndarray, np.ndarray]:
        available_cols = [c for c in self.feature_cols if c in self.df.columns]
        if not available_cols:
            raise ValueError("No feature columns found in DataFrame.")

        feature_matrix = self.scaler.fit_transform(self.df[available_cols].values)
        deltas = self._compute_lap_delta().values

        X_list, y_list = [], []
        total = len(self.df)

        for i in range(self.seq_len, total - self.horizon + 1):
            X_list.append(feature_matrix[i - self.seq_len : i])
            y_list.append(deltas[i : i + self.horizon])

        if not X_list:
            raise ValueError(
                f"Not enough laps ({total}) for seq_len={self.seq_len} + horizon={self.horizon}."
            )

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
        print(f"[SequenceBuilder] X: {X.shape}  y: {y.shape}")
        return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 5. PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────
def build_dataset(
    year: int = 2024,
    grand_prix: str = "British Grand Prix",
    drivers: Optional[list] = None,
    session_type: str = "R",
) -> dict:
    """
    End-to-end pipeline: load → extract → clean → sequence.
    Returns a dict keyed by driver abbreviation.
    """
    loader  = SessionLoader(year, grand_prix, session_type)
    session = loader.load()

    drivers = drivers or list(session.drivers)[:3]   # default: top 3 drivers
    dataset = {}

    for drv in drivers:
        print(f"\n── Processing driver: {drv} ──")
        try:
            raw_df   = TelemetryExtractor(session, drv).extract()
            clean_df = DataCleaner(raw_df).clean()
            X, y     = SequenceBuilder(clean_df).build()
            dataset[drv] = {"X": X, "y": y, "laps": clean_df}
        except Exception as e:
            print(f"[WARNING] Driver {drv} skipped: {e}")

    return dataset


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data = build_dataset(year=2024, grand_prix="British Grand Prix", drivers=["VER", "HAM", "NOR"])
    for drv, d in data.items():
        print(f"{drv}: X={d['X'].shape}, y={d['y'].shape}")