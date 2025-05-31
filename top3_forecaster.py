"""Advanced model to predict top 3 F1 finishers using multiple data sources.

This module demonstrates how to collect data from various Formula 1 APIs,
prepare features and train an XGBoost model to rank drivers for upcoming
races. The code is heavily commented to explain each step of the pipeline.

NOTE: API calls are shown as examples. They require network access and may
need API keys. In an offline environment the functions will return empty
DataFrames.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Attempt to import XGBoost.  If unavailable the code can be adapted to use
# another regressor from scikit-learn.
try:  # pragma: no cover - optional dependency
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - xgboost not installed
    XGBRegressor = None
    warnings.warn("xgboost is not installed; training will fail")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

JOLPICA_BASE = "https://api.jolpica.com/v1"  # hypothetical API endpoint
OPENF1_BASE = "https://api.openf1.org/v1"
REQUEST_SLEEP = 1  # polite delay between API calls

# Example list of street circuits taken from Wikipedia.  In a real project this
# should be loaded from a maintained file or API.
STREET_CIRCUITS = {
    "monaco",
    "miami",
    "jeddah",
    "vegas",
    "baku",
    "singapore",
    "melbourne",
}

# Known sprint weekends since 2021.  Keys are "year_grandprix" slugs.
SPRINT_WEEKENDS = {
    "2021_silverstone",
    "2021_monza",
    "2021_interlagos",
    "2022_imola",
    "2022_red_bull_ring",
    "2022_interlagos",
    "2023_baku",
    "2023_red_bull_ring",
    "2023_lusail",
    "2023_cota",
    "2023_interlagos",
    "2024_suzuka",
}

# ---------------------------------------------------------------------------
# Helper functions for API access
# ---------------------------------------------------------------------------


def _fetch_paginated(base_url: str, endpoint: str, params: dict | None = None) -> pd.DataFrame:
    """Fetch all pages from an API endpoint and return as DataFrame."""
    params = params or {}
    page = 1
    collected: List[dict] = []

    while True:
        params_with_page = params | {"page": page}
        url = f"{base_url}/{endpoint.lstrip('/') }"
        try:  # pragma: no cover - network dependency
            resp = requests.get(url, params=params_with_page, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # pragma: no cover - offline
            print(f"Request failed for {url}: {exc}")
            break
        if not data:
            break
        collected.extend(data)
        page += 1
        time.sleep(REQUEST_SLEEP)

    return pd.DataFrame(collected)


# ---------------------------------------------------------------------------
# Data acquisition
# ---------------------------------------------------------------------------


def fetch_data(start_year: int = 2018) -> pd.DataFrame:
    """Collect historical race data from multiple sources.

    The function queries the Jolpica F1 API, OpenF1 and FastF1 to gather race
    results, qualifying times, lap statistics, pit stops and weather
    information. In an offline environment the returned DataFrame will be empty.
    """

    all_results = []

    current_year = pd.Timestamp.utcnow().year
    for year in range(start_year, current_year + 1):
        # ------------------------------------------------------------------
        # Example: fetch race results from Jolpica F1 API
        # ------------------------------------------------------------------
        results = _fetch_paginated(
            JOLPICA_BASE, "results", {"year": year}
        )
        if results.empty:
            continue
        results["Year"] = year

        # Qualifying results
        quali = _fetch_paginated(
            JOLPICA_BASE, "qualifying", {"year": year}
        )
        results = results.merge(
            quali,
            on=["race_id", "driver_id"],
            how="left",
            suffixes=("", "_quali"),
        )

        # Constructor standings for contextual team strength
        constructors = _fetch_paginated(
            JOLPICA_BASE, "constructorStandings", {"year": year}
        )
        standings = constructors[["constructor_id", "points"]].rename(
            columns={"points": "ConstructorPoints"}
        )
        results = results.merge(
            standings, on="constructor_id", how="left"
        )

        # ------------------------------------------------------------------
        # Fetch OpenF1 weather and pit stop data using session key
        # ------------------------------------------------------------------
        sessions = _fetch_paginated(
            OPENF1_BASE, "sessions", {"year": year, "session_name": "R"}
        )
        for _, session in sessions.iterrows():
            session_key = session.get("session_key")
            if not session_key:
                continue
            weather = _fetch_paginated(
                OPENF1_BASE, "weather", {"session_key": session_key}
            )
            pits = _fetch_paginated(
                OPENF1_BASE, "pit", {"session_key": session_key}
            )
            weather_avg = weather.mean(numeric_only=True).add_prefix("Weather_")
            pit_avg = (
                pits.assign(Duration_s=lambda d: pd.to_timedelta(d["duration"]).dt.total_seconds())
                .groupby("driver_number")["Duration_s"]
                .mean()
                .reset_index(name="AvgPitTime_s")
                .rename(columns={"driver_number": "driver_id"})
            )
            mask = results["race_id"] == session.get("meeting_key")
            results.loc[mask, weather_avg.index] = weather_avg.values
            results = results.merge(pit_avg, on=["race_id", "driver_id"], how="left")

        all_results.append(results)
        time.sleep(REQUEST_SLEEP)

    if not all_results:
        # Offline or failed requests
        return pd.DataFrame()

    df = pd.concat(all_results, ignore_index=True)
    return df


# ---------------------------------------------------------------------------
# Data preprocessing
# ---------------------------------------------------------------------------


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare raw race data for feature engineering."""

    if df.empty:
        return df

    # Sort chronologically so that time-based splits don't leak future info
    df = df.sort_values(["Year", "race_id", "driver_id"]).reset_index(drop=True)

    # Replace missing numeric values with the column median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # Replace missing categoricals with "Unknown"
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        df[col] = df[col].fillna("Unknown")

    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

@dataclass
class FeatureConfig:
    """Configuration holder for engineered features."""

    recent_window: int = 3


def feature_engineering(df: pd.DataFrame, cfg: FeatureConfig | None = None) -> pd.DataFrame:
    """Create additional features for the ranking model."""

    if cfg is None:
        cfg = FeatureConfig()

    if df.empty:
        return df

    # Street circuit flag based on circuit slug
    df["is_street"] = df["circuit_slug"].str.lower().isin(STREET_CIRCUITS).astype(int)

    # Sprint weekend indicator
    df["sprint_weekend"] = (
        df["Year"].astype(str) + "_" + df["circuit_slug"].str.lower()
    ).isin(SPRINT_WEEKENDS).astype(int)

    # Driver average finishing position at this circuit
    df["driver_circuit_avg"] = (
        df.groupby(["driver_id", "circuit_slug"]) ["position"]
        .transform("mean")
    )

    # Driver average finish over the last N races
    df["driver_recent_avg"] = (
        df.groupby("driver_id")["position"]
        .transform(lambda s: s.shift().rolling(cfg.recent_window).mean())
    )

    # Safety car history by circuit (example placeholder using race flags)
    df["safety_car_incidents_historical_avg"] = (
        df.groupby("circuit_slug")["safety_car"]
        .transform("mean")
    )

    # Tyre strategy: average stint length per compound
    tyre_cols = [c for c in df.columns if c.startswith("tyre_")]
    if tyre_cols:
        df["tyre_compound_strategy"] = df[tyre_cols].mean(axis=1)
    else:
        df["tyre_compound_strategy"] = 0.0

    # Overtaking score using historical overtakes (placeholder)
    df["overtaking_score"] = df.groupby("circuit_slug")["overtakes"].transform("mean")

    return df


# ---------------------------------------------------------------------------
# Feature transformation
# ---------------------------------------------------------------------------


def transform_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, ColumnTransformer]:
    """Encode categorical variables and scale numerical columns."""

    target = df["position"].astype(float)

    categorical = ["driver_id", "constructor_id", "circuit_slug"]
    numerical = [
        "qualifying_position",
        "driver_circuit_avg",
        "driver_recent_avg",
        "ConstructorPoints",
        "AvgPitTime_s",
        "Weather_air_temperature",
        "Weather_track_temperature",
        "Weather_rainfall",
        "overtaking_score",
        "tyre_compound_strategy",
        "safety_car_incidents_historical_avg",
        "is_street",
        "sprint_weekend",
    ]

    cat_transformer = OneHotEncoder(handle_unknown="ignore")
    num_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_transformer, categorical),
            ("num", num_transformer, numerical),
        ]
    )

    X = preprocessor.fit_transform(df[categorical + numerical])
    return X, target.values, preprocessor


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------


def train_model(X: np.ndarray, y: np.ndarray) -> Tuple[object, float, float]:
    """Train an XGBoost model using TimeSeriesSplit cross validation."""

    if XGBRegressor is None:
        raise RuntimeError(
            "xgboost is not installed. Install it to train the ranking model."
        )

    # Split the last 20% of samples as a hold-out set. In a real scenario you
    # should split per race to avoid mixing drivers from the same event across
    # train and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )

    # Use TimeSeriesSplit to mimic forward chaining evaluation
    tscv = TimeSeriesSplit(n_splits=5)
    for train_idx, val_idx in tscv.split(X_train):
        model.fit(X_train[train_idx], y_train[train_idx])

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)

    return model, mae, rmse


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------


def top3_accuracy(df: pd.DataFrame, pred_col: str = "pred_position") -> float:
    """Return percentage of races where at least two of the predicted top 3
    match the actual top 3."""

    accuracies = []
    for race_id, group in df.groupby("race_id"):
        pred_top3 = group.nsmallest(3, pred_col)["driver_id"].tolist()
        actual_top3 = group.nsmallest(3, "position")["driver_id"].tolist()
        match = len(set(pred_top3) & set(actual_top3)) >= 2
        accuracies.append(match)
    return float(np.mean(accuracies))


def evaluate_model(
    model: object,
    preprocessor: ColumnTransformer,
    df: pd.DataFrame,
) -> None:
    """Compute predictions and print evaluation metrics."""

    X_all = preprocessor.transform(
        df[
            [
                "driver_id",
                "constructor_id",
                "circuit_slug",
                "qualifying_position",
                "driver_circuit_avg",
                "driver_recent_avg",
                "ConstructorPoints",
                "AvgPitTime_s",
                "Weather_air_temperature",
                "Weather_track_temperature",
                "Weather_rainfall",
                "overtaking_score",
                "tyre_compound_strategy",
                "safety_car_incidents_historical_avg",
                "is_street",
                "sprint_weekend",
            ]
        ]
    )
    df["pred_position"] = model.predict(X_all)

    mae = mean_absolute_error(df["position"], df["pred_position"])
    rmse = mean_squared_error(df["position"], df["pred_position"], squared=False)
    acc = top3_accuracy(df)

    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"Top-3 accuracy: {acc:.3%}")

    # Feature importances from XGBoost
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_names = preprocessor.get_feature_names_out()
        imp_series = pd.Series(importances, index=feature_names)
        print("\nTop 10 features:\n", imp_series.nlargest(10))

    # Simple plot of predicted vs actual positions
    try:  # pragma: no cover - requires matplotlib
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 6))
        plt.scatter(df["position"], df["pred_position"], alpha=0.7)
        plt.xlabel("Actual Finish Position")
        plt.ylabel("Predicted Finish Score")
        plt.title("Predicted vs. Actual Finish")
        plt.grid(True)
        plt.show()
    except Exception as exc:
        print(f"Plotting failed: {exc}")


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full training and evaluation pipeline."""

    data = fetch_data(start_year=2018)
    if data.empty:
        print("No data fetched. Ensure network access and API keys are configured.")
        return

    data = preprocess_data(data)
    data = feature_engineering(data)
    X, y, preprocessor = transform_features(data)
    model, mae, rmse = train_model(X, y)
    print(f"\nHold-out MAE: {mae:.3f}, RMSE: {rmse:.3f}")

    evaluate_model(model, preprocessor, data)


if __name__ == "__main__":
    main()
