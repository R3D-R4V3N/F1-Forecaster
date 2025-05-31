import argparse
import fastf1
import os
import pandas as pd
fastf1.Cache.enable_cache(os.environ.get("FASTF1_CACHE_DIR", "f1_cache"))


def _track_overtake_potential(session_r) -> float:
    """Average improvement from grid to finish for drivers in a race."""
    results = session_r.results[["Abbreviation", "GridPosition", "Position"]].copy()
    results[["GridPosition", "Position"]] = results[["GridPosition", "Position"]].apply(pd.to_numeric, errors="coerce")
    diff = results["GridPosition"] - results["Position"]
    diff = diff[diff > 0]
    if len(diff) == 0:
        return 0.0
    return float(diff.mean())


def calc_overtake_potential(years, grand_prix):
    """Return list of overtake potentials for the given seasons."""
    potentials = []
    for year in years:
        session_r = fastf1.get_session(year, grand_prix, "R")
        session_r.load()
        potentials.append(_track_overtake_potential(session_r))
    return potentials


def main():
    parser = argparse.ArgumentParser(description="Calculate track overtake potential")
    parser.add_argument("years", nargs="+", help="Season(s) to analyze")
    parser.add_argument("grand_prix", help="Grand Prix name, e.g. 'Bahrain Grand Prix'")
    args = parser.parse_args()

    years = [int(y) for y in args.years]
    potentials = calc_overtake_potential(years, args.grand_prix)
    for year, val in zip(years, potentials):
        print(f"{year}: {val:.2f}")
    if len(potentials) > 1:
        avg = sum(potentials) / len(potentials)
        print(f"Average: {avg:.2f}")


if __name__ == "__main__":
    main()
