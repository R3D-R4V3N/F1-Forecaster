#!/usr/bin/env python3
import os
import pandas as pd
import argparse
import fastf1
from predictor.openf1_utils import slugify


def dump_session_times(year: int, grand_prix: str, outdir: str = "ff1_cache_dump") -> str:
    """Export fastest lap times from FastF1 cache to CSV."""
    fastf1.Cache.enable_cache(os.environ.get("FASTF1_CACHE_DIR", "f1_cache"))
    ff1_map = [
        ("FP1", "FP1Time_s"),
        ("FP2", "FP2Time_s"),
        ("FP3", "FP3Time_s"),
        ("Q", "QualifyingTime_s"),
    ]
    result = pd.DataFrame({"Driver": []})
    for sess_code, col in ff1_map:
        try:
            sess = fastf1.get_session(year, grand_prix, sess_code)
            sess.load()
            laps = sess.laps.pick_fastest()
            if not hasattr(laps, "columns"):
                laps = pd.DataFrame([laps.to_dict()])
            if "Driver" not in laps.columns:
                laps = laps.reset_index().rename(columns={"index": "Driver"})
            fast = (
                laps.groupby("Driver")["LapTime"]
                .min()
                .dt.total_seconds()
                .reset_index(name=col)
            )
            if result.empty:
                result = fast
            else:
                result = result.merge(fast, on="Driver", how="outer")
        except Exception as exc:
            print(f"Failed to load {sess_code}: {exc}")
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, f"{year}_{slugify(grand_prix)}.csv")
    result.to_csv(csv_path, index=False)
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump FastF1 session times")
    parser.add_argument("year", type=int)
    parser.add_argument("grand_prix")
    parser.add_argument("--outdir", default="ff1_cache_dump")
    args = parser.parse_args()
    path = dump_session_times(args.year, args.grand_prix, args.outdir)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
