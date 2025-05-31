import os
import fastf1
from fastf1 import ergast
import pandas as pd
import requests
from .openf1_utils import pit_stop_features
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

fastf1.Cache.enable_cache(os.environ.get("FASTF1_CACHE_DIR", "f1_cache"))


def _constructor_points_for_year(year: int) -> dict:
    """Return constructor points for a season via ergast."""
    try:
        standings = ergast.fetch_constructor_standings(season=year)
    except Exception:
        return {}
    df = standings.iloc[0]
    return {row["constructorId"]: float(row["points"]) for row in df.itertuples()}


def _extract_overtakes(laps: pd.DataFrame) -> pd.DataFrame:
    """Count overtakes for each driver based on position changes."""
    laps = laps.sort_values(["Driver", "LapNumber"])
    pos_diff = laps.groupby("Driver")["Position"].diff().fillna(0)
    overtakes = (
        (pos_diff < 0).groupby(laps["Driver"]).sum().reset_index(name="Overtakes")
    )
    return overtakes


def _qualifying_segments(year: int, grand_prix: str) -> pd.DataFrame:
    """Return Q1, Q2 and Q3 times for each driver."""
    try:
        session = fastf1.get_session(year, grand_prix, "Q")
        session.load()
    except Exception:
        return pd.DataFrame(columns=["Driver", "Q1Time_s", "Q2Time_s", "Q3Time_s"])

    cols = ["Q1", "Q2", "Q3"]
    res = session.results[["Abbreviation"] + cols].copy()
    for c in cols:
        res[c] = pd.to_timedelta(res[c]).dt.total_seconds()
    res.rename(columns={"Abbreviation": "Driver", "Q1": "Q1Time_s", "Q2": "Q2Time_s", "Q3": "Q3Time_s"}, inplace=True)
    return res


def _track_overtake_potential(session_r) -> float:
    """Average improvement from grid to finish for drivers in a race."""
    results = session_r.results[["Abbreviation", "GridPosition", "Position"]].copy()
    results[["GridPosition", "Position"]] = results[["GridPosition", "Position"]].apply(
        pd.to_numeric, errors="coerce"
    )

    # When a driver starts from the pit lane ``GridPosition`` is often ``0`` or
    # missing in the timing data.  Likewise ``Position`` can be ``0`` for a DNF.
    # Replace these values with the total number of participants so they still
    # contribute to the average improvement calculation.
    total_drivers = len(results)
    results["GridPosition"] = results["GridPosition"].fillna(0)
    results.loc[results["GridPosition"] <= 0, "GridPosition"] = total_drivers
    results["Position"] = results["Position"].fillna(total_drivers)
    results.loc[results["Position"] <= 0, "Position"] = total_drivers

    diff = results["GridPosition"] - results["Position"]
    diff = diff[diff > 0]
    if len(diff) == 0:
        return 0.0
    return float(diff.mean())


def _weather_features(session_r) -> dict:
    """Return average weather metrics for a session."""
    weather = session_r.weather_data
    air_col = "AirTemp" if "AirTemp" in weather.columns else "AirTemperature"
    track_col = "TrackTemp" if "TrackTemp" in weather.columns else "TrackTemperature"
    return {
        "AirTemp": float(weather.get(air_col, 0).mean()) if not weather.empty else 0.0,
        "TrackTemp": (
            float(weather.get(track_col, 0).mean()) if not weather.empty else 0.0
        ),
        "Rainfall": (
            float(weather.get("Rainfall", 0).fillna(0).mean())
            if not weather.empty
            else 0.0
        ),
    }


def load_training_data(years, grand_prix: str):
    """Load race and qualifying data for multiple seasons."""
    if isinstance(years, int):
        years = [years]

    all_features = []
    all_targets = []
    sectors = []

    for year in years:
        session_r = fastf1.get_session(year, grand_prix, "R")
        session_r.load()
        laps = session_r.laps[
            [
                "Driver",
                "LapNumber",
                "LapTime",
                "Sector1Time",
                "Sector2Time",
                "Sector3Time",
                "Position",
            ]
        ].dropna()
        for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
            laps[f"{col}_s"] = laps[col].dt.total_seconds()

        overtakes = _extract_overtakes(laps)
        race_avg = laps.groupby("Driver").mean().reset_index()

        q_segments = _qualifying_segments(year, grand_prix)

        session_q = fastf1.get_session(year, grand_prix, "Q")
        session_q.load()
        q_laps = session_q.laps.pick_fastest()
        if not hasattr(q_laps, "columns"):
            q_laps = pd.DataFrame([q_laps.to_dict()])
        if "Driver" not in q_laps.columns:
            q_laps = q_laps.reset_index().rename(columns={"index": "Driver"})
        q_times = (
            q_laps.groupby("Driver")["LapTime"]
            .min()
            .dt.total_seconds()
            .reset_index(name="QualifyingTime_s")
        )

        df = race_avg.merge(q_times, on="Driver", how="left")
        df = df.merge(q_segments, on="Driver", how="left")

        # ``session_r.results`` uses ``Abbreviation`` for the driver code and
        # ``TeamName`` for the constructor. Rename them to keep the rest of the
        # pipeline consistent with the laps data which uses ``Driver``.
        team_map = (
            session_r.results[["Abbreviation", "TeamName"]]
            .drop_duplicates()
            .rename(columns={"Abbreviation": "Driver", "TeamName": "Team"})
        )
        df = df.merge(team_map, on="Driver", how="left")

        constructor_pts = _constructor_points_for_year(year)
        df["ConstructorPoints"] = df["Team"].map(constructor_pts).fillna(0)

        df = df.merge(overtakes, on="Driver", how="left")

        # additional data from OpenF1
        pit_df = pit_stop_features(year, grand_prix)
        df = df.merge(pit_df, on="Driver", how="left")

        # track specific features
        df["TrackOvertakePotential"] = _track_overtake_potential(session_r)
        weather = _weather_features(session_r)
        df["AirTemp"] = weather["AirTemp"]
        df["TrackTemp"] = weather["TrackTemp"]
        df["Rainfall"] = weather["Rainfall"]

        all_features.append(
            df[
                [
                    "QualifyingTime_s",
                    "Q1Time_s",
                    "Q2Time_s",
                    "Q3Time_s",
                    "Sector1Time_s",
                    "Sector2Time_s",
                    "Sector3Time_s",
                    "Overtakes",
                    "ConstructorPoints",
                    "AvgPitTime_s",
                    "TrackOvertakePotential",
                    "AirTemp",
                    "TrackTemp",
                    "Rainfall",
                ]
            ]
            .fillna(0)
            .infer_objects(copy=False)
        )
        all_targets.append(df["LapTime_s"])
        sectors.append(
            df[["Driver", "Team", "Sector1Time_s", "Sector2Time_s", "Sector3Time_s"]]
        )

    features = pd.concat(all_features, ignore_index=True)
    target = pd.concat(all_targets, ignore_index=True)
    sector_times = (
        pd.concat(sectors, ignore_index=True)
        .drop_duplicates(subset=["Driver"])
        .reset_index(drop=True)
    )
    return features, target, sector_times


def train_model(features: pd.DataFrame, target: pd.Series):
    """Train a gradient boosting model on the provided data."""
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    model = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.1, random_state=42
    )
    model.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, model.predict(X_test))
    return model, mae


def predict_race(
    model: GradientBoostingRegressor,
    sector_times: pd.DataFrame,
    race_data: pd.DataFrame,
    constructor_points: dict | None = None,
) -> pd.DataFrame:
    """Predict race results using qualifying times and previous sector data."""
    df = race_data.merge(sector_times, on="Driver", how="left")
    df["Overtakes"] = 0
    if constructor_points:
        df["ConstructorPoints"] = df["Team"].map(constructor_points).fillna(0)
    else:
        df["ConstructorPoints"] = 0

    feature_cols = [
        "QualifyingTime_s",
        "Q1Time_s",
        "Q2Time_s",
        "Q3Time_s",
        "Sector1Time_s",
        "Sector2Time_s",
        "Sector3Time_s",
        "Overtakes",
        "ConstructorPoints",
        "AvgPitTime_s",
        "TrackOvertakePotential",
        "AirTemp",
        "TrackTemp",
        "Rainfall",
    ]
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    features = df[feature_cols].fillna(0).infer_objects(copy=False)
    df["PredictedLapTime_s"] = model.predict(features)
    return df.sort_values("PredictedLapTime_s")[["Driver", "PredictedLapTime_s"]]
