import argparse
import os
import yaml
import pandas as pd
from dotenv import load_dotenv

from predictor.model import (
    load_training_data,
    train_model,
    predict_race,
    _constructor_points_for_year,
)
from predictor.utils import parse_time_value, normalize_driver_name
from predictor.openf1_utils import race_name_by_round


def main():
    """Run race prediction for a given race key or round number.

    Usage:
        python3 predict.py chinese_gp
        python3 predict.py --round 5 --year 2025
    """
    load_dotenv()
    parser = argparse.ArgumentParser(description="Predict F1 race results")
    parser.add_argument("race", nargs="?", help="Race key as defined in races.yaml")
    parser.add_argument("--round", type=int, help="Round number of the season")
    parser.add_argument("--year", type=int, default=2025, help="Season year for --round")
    parser.add_argument(
        "--train-years",
        default="2021,2022,2023,2024",
        help="Comma separated list of years to train on (default 2021,2022,2023,2024)",
    )
    args = parser.parse_args()

    race_key = args.race
    if args.round is not None:
        gp_name = race_name_by_round(args.year, args.round)
        if not gp_name:
            raise SystemExit(f"Could not resolve round {args.round} for {args.year}")
        with open("races.yaml", "r") as f:
            races_map = yaml.safe_load(f)
        for k, v in races_map.items():
            if str(v.get("grand_prix", "")).lower() == gp_name.lower():
                race_key = k
                break
        if race_key is None:
            raise SystemExit(f"Grand Prix '{gp_name}' not found in races.yaml")
    elif race_key is None:
        parser.error("either race key or --round must be provided")

    with open("races.yaml", "r") as f:
        races = yaml.safe_load(f)

    if race_key not in races:
        raise SystemExit(f"Race '{race_key}' not found in races.yaml")

    race_cfg = races[race_key]

    years = [int(y) for y in args.train_years.split(',')]

    features, target, sector_times = load_training_data(
        years, race_cfg["grand_prix"]
    )
    model, mae = train_model(features, target)

    drivers = pd.DataFrame(race_cfg["drivers"])
    drivers["Driver"] = drivers["Driver"].apply(normalize_driver_name)
    for col in ["Q1Time_s", "Q2Time_s", "Q3Time_s", "QualifyingTime_s"]:
        if col in drivers:
            drivers[col] = drivers[col].apply(parse_time_value)

    weather = race_cfg.get("weather", {})
    drivers["AirTemp"] = weather.get("air_temp", 0)
    drivers["TrackTemp"] = weather.get("track_temp", 0)
    drivers["Rainfall"] = weather.get("rainfall", 0)
    drivers["TrackOvertakePotential"] = weather.get("overtake_potential", 0)

    constructor_points = _constructor_points_for_year(years[-1])
    result = predict_race(model, sector_times, drivers, constructor_points)

    print(result)
    print(
        f"Model Error on years {','.join(map(str, years))} (MAE): {mae:.2f} seconds"
    )


if __name__ == "__main__":
    main()
