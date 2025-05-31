import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import make_scorer
from xgboost import XGBRegressor


def fetch_jolpica_results(years: List[int]) -> pd.DataFrame:
    """Fetch race results, qualifying, pit stops and sprint results from Jolpica API."""
    base_url = "https://ergast.com/api/f1"  # placeholder for Jolpica endpoints
    all_records = []

    for year in years:
        # Example endpoint for results
        url = f"{base_url}/{year}/results.json"
        resp = requests.get(url)
        if resp.status_code != 200:
            print(f"Failed to fetch {url}: {resp.status_code}")
            continue
        data = resp.json().get("MRData", {}).get("RaceTable", {}).get("Races", [])
        for race in data:
            round_number = race.get("round")
            for result in race.get("Results", []):
                record = {
                    "year": year,
                    "round": round_number,
                    "raceName": race.get("raceName"),
                    "driver": result.get("Driver", {}).get("driverId"),
                    "constructor": result.get("Constructor", {}).get("constructorId"),
                    "position": pd.to_numeric(result.get("position"), errors="coerce"),
                }
                all_records.append(record)

        # Qualifying results per round
        for race in data:
            round_number = race.get("round")
            q_url = f"{base_url}/{year}/{round_number}/qualifying.json"
            q_resp = requests.get(q_url)
            if q_resp.status_code != 200:
                continue
            q_data = q_resp.json().get("MRData", {}).get("RaceTable", {}).get("Races", [])
            if not q_data:
                continue
            for res in q_data[0].get("QualifyingResults", []):
                all_records.append({
                    "year": year,
                    "round": round_number,
                    "driver": res.get("Driver", {}).get("driverId"),
                    "qualifying_position": pd.to_numeric(res.get("position"), errors="coerce"),
                })
        # Pit stop data
        for race in data:
            round_number = race.get("round")
            pit_url = f"{base_url}/{year}/{round_number}/pitstops.json?limit=200"
            pit_resp = requests.get(pit_url)
            if pit_resp.status_code != 200:
                continue
            pit_data = pit_resp.json().get("MRData", {}).get("RaceTable", {}).get("Races", [])
            if not pit_data:
                continue
            for stop in pit_data[0].get("PitStops", []):
                all_records.append({
                    "year": year,
                    "round": round_number,
                    "driver": stop.get("driverId"),
                    "pit_stop_time": pd.to_numeric(stop.get("duration"), errors="coerce"),
                })
        # Sprint results if available
        for race in data:
            round_number = race.get("round")
            sprint_url = f"{base_url}/{year}/{round_number}/sprint.json"
            sprint_resp = requests.get(sprint_url)
            if sprint_resp.status_code != 200:
                continue
            sprint_data = sprint_resp.json().get("MRData", {}).get("RaceTable", {}).get("Races", [])
            if not sprint_data:
                continue
            for res in sprint_data[0].get("SprintResults", []):
                all_records.append({
                    "year": year,
                    "round": round_number,
                    "driver": res.get("Driver", {}).get("driverId"),
                    "sprint_position": pd.to_numeric(res.get("position"), errors="coerce"),
                })

    return pd.DataFrame(all_records)


def load_kaggle_data(weather_path: str, tires_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load supplementary data from Kaggle datasets."""
    weather_df = pd.read_csv(weather_path)
    tires_df = pd.read_csv(tires_path)
    return weather_df, tires_df


def merge_datasets(results: pd.DataFrame, weather: pd.DataFrame, tires: pd.DataFrame, overtaking: pd.DataFrame) -> pd.DataFrame:
    """Merge results with weather, tire, and overtaking data."""
    df = results.merge(weather, on=["year", "round"], how="left")
    df = df.merge(tires, on=["year", "round", "compound"], how="left")
    df = df.merge(overtaking, on="circuitId", how="left")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].apply(lambda col: col.fillna(col.median()))
    df[categorical_cols] = df[categorical_cols].fillna("Unknown")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate additional features avoiding data leakage."""
    df = df.sort_values(["year", "round"]).reset_index(drop=True)

    # Recent average finish for each driver
    df["driver_recent_avg"] = df.groupby("driver")["position"].transform(
        lambda s: s.shift().rolling(window=5, min_periods=1).mean()
    )

    # Average finish at this circuit for each driver
    df["driver_circuit_avg"] = df.groupby(["driver", "circuitId"])["position"].transform(
        lambda s: s.shift().expanding().mean()
    )

    # Average pit stop time per team recent races
    if "pit_stop_time" in df.columns:
        df["team_recent_pit_avg"] = df.groupby("constructor")["pit_stop_time"].transform(
            lambda s: s.shift().rolling(window=3, min_periods=1).mean()
        )
    else:
        df["team_recent_pit_avg"] = np.nan

    # Binary precipitation indicator
    df["precipitation_flag"] = (df["precipitation"].fillna(0) > 0).astype(int)

    return df


def top3_accuracy(y_true: np.ndarray, y_pred: np.ndarray, groups: pd.Series) -> float:
    """Compute top-3 accuracy across races."""
    correct = 0
    unique_races = groups.unique()
    for race_id in unique_races:
        mask = groups == race_id
        actual_top3 = set(y_true[mask].nsmallest(3).index)
        pred_top3 = set(y_pred[mask].argsort()[:3])
        if len(actual_top3.intersection(pred_top3)) >= 2:
            correct += 1
    return correct / len(unique_races)


def train_model(df: pd.DataFrame) -> XGBRegressor:
    features = [
        col for col in df.columns
        if col not in {"position", "raceName", "driver", "constructor"}
    ]
    X = df[features]
    y = df["position"]
    groups = df["year"].astype(str) + "_" + df["round"].astype(str)

    tscv = TimeSeriesSplit(n_splits=5)
    xgb = XGBRegressor(objective="reg:squarederror", n_estimators=200)
    param_grid = {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }
    scorer = make_scorer(lambda yt, yp: -np.mean(np.abs(yt - yp)))
    search = RandomizedSearchCV(
        xgb,
        param_distributions=param_grid,
        n_iter=10,
        scoring=scorer,
        cv=tscv,
        verbose=1,
    )
    search.fit(X, y)

    return search.best_estimator_


def main():
    years = list(range(2018, 2024))
    results_df = fetch_jolpica_results(years)

    # Placeholder paths for supplementary datasets
    weather_path = os.path.join("data", "weather.csv")
    tires_path = os.path.join("data", "tires.csv")
    overtaking_path = os.path.join("data", "overtaking.csv")

    if os.path.exists(weather_path):
        weather_df, tires_df = load_kaggle_data(weather_path, tires_path)
        overtaking_df = pd.read_csv(overtaking_path)
        merged = merge_datasets(results_df, weather_df, tires_df, overtaking_df)
    else:
        merged = results_df

    merged = clean_data(merged)
    features = engineer_features(merged)
    model = train_model(features)

    print("Model trained with best params:", model.get_params())


if __name__ == "__main__":
    main()
