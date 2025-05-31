import requests
import os
from fastf1 import ergast

# Directory used to cache fastest session times. Can be overridden with the
# ``SESSION_TIMES_DIR`` environment variable.
SESSION_TIMES_DIR = os.environ.get("SESSION_TIMES_DIR", "session_times")

# Import helper utilities. ``slugify`` is re-exported so callers can simply
# import it from this module as well.
from .utils import DRIVER_NAME_TO_CODE, slugify as _slugify

# Re-export ``slugify`` for backwards compatibility.  This allows
# ``from predictor.openf1_utils import slugify`` which older code expected.
slugify = _slugify

__all__ = [
    "fetch",
    "pit_stop_features",
    "list_drivers",
    "race_drivers",
    "race_name_by_round",
    "season_schedule",
    "weather_summary",
    "session_fastest_times",
    "overtake_potential",
    "slugify",
]

# Directory used for caching fastest session times (practice and qualifying).
# Can be overridden via the ``SESSION_TIMES_DIR`` environment variable.
SESSION_TIMES_DIR = os.environ.get("SESSION_TIMES_DIR", "session_times")

BASE_URL = "https://api.openf1.org/v1"

# Directory used for caching fastest session times (practice and qualifying).
# Can be overridden via the ``SESSION_TIMES_DIR`` environment variable.
SESSION_TIMES_DIR = os.environ.get("SESSION_TIMES_DIR", "session_times")


def fetch(endpoint: str, params: dict | None = None) -> list:
    """Fetch data from the OpenF1 API.

    This helper wraps ``requests.get`` and returns the parsed JSON data. In
    offline environments it returns an empty list.
    """
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # pragma: no cover - network issues
        print(f"OpenF1 request failed for {endpoint}: {exc}")
        return []


def pit_stop_features(year: int, grand_prix: str) -> "pd.DataFrame":
    """Return average pit stop duration per driver.

    Parameters match those used for FastF1 sessions. The function attempts to
    query the OpenF1 ``pit`` endpoint and returns a DataFrame with columns
    ``Driver`` and ``AvgPitTime_s``. If data cannot be retrieved an empty
    DataFrame is returned.
    """
    try:
        import pandas as pd
    except Exception:  # pragma: no cover - pandas missing
        return __import__("pandas").DataFrame(columns=["Driver", "AvgPitTime_s"])

    # Fetch session metadata first to get the session key
    sessions = fetch(
        "sessions", {"year": year, "meeting_name": grand_prix, "session_name": "R"}
    )
    if not sessions:
        return pd.DataFrame(columns=["Driver", "AvgPitTime_s"])
    session_key = sessions[0].get("session_key")
    if not session_key:
        return pd.DataFrame(columns=["Driver", "AvgPitTime_s"])

    pits = fetch("pit", {"session_key": session_key})
    if not pits:
        return pd.DataFrame(columns=["Driver", "AvgPitTime_s"])

    df = pd.DataFrame(pits)
    if "duration" not in df.columns or "driver_number" not in df.columns:
        return pd.DataFrame(columns=["Driver", "AvgPitTime_s"])
    df["Duration_s"] = pd.to_timedelta(df["duration"]).dt.total_seconds()
    avg = (
        df.groupby("driver_number")["Duration_s"]
        .mean()
        .reset_index(name="AvgPitTime_s")
    )
    avg.rename(columns={"driver_number": "Driver"}, inplace=True)
    return avg


def list_drivers(year: int) -> "pd.DataFrame":
    """Return a list of drivers for a season.

    The OpenF1 ``drivers`` endpoint is queried. If fetching fails, a
    DataFrame constructed from ``DRIVER_NAME_TO_CODE`` is returned.
    """
    try:  # pragma: no cover - network dependency
        import pandas as pd
    except Exception:  # pragma: no cover - pandas missing
        return __import__("pandas").DataFrame(columns=["Driver", "FullName"])

    drivers = fetch("drivers", {"year": year})
    if not drivers:
        data = [
            {"Driver": code, "FullName": name}
            for name, code in DRIVER_NAME_TO_CODE.items()
        ]
        return pd.DataFrame(data).drop_duplicates("Driver")

    df = pd.DataFrame(drivers)
    name_col = "full_name" if "full_name" in df.columns else "name"
    code_col = "driver_number" if "driver_number" in df.columns else "driver_id"
    return df.rename(columns={name_col: "FullName", code_col: "Driver"})[
        ["Driver", "FullName"]
    ]


def race_drivers(year: int, grand_prix: str) -> "pd.DataFrame":
    """Return the list of drivers who participated in a race."""
    try:  # pragma: no cover - requires pandas
        import pandas as pd
    except Exception:  # pragma: no cover - pandas missing
        return __import__("pandas").DataFrame(columns=["Driver", "FullName"])

    # Try FastF1 results first
    try:  # pragma: no cover - requires fastf1
        import fastf1

        fastf1.Cache.enable_cache(os.environ.get("FASTF1_CACHE_DIR", "f1_cache"))
        session_r = fastf1.get_session(year, grand_prix, "R")
        session_r.load()
        df = session_r.results[["Abbreviation", "FullName"]].copy()
        return df.rename(columns={"Abbreviation": "Driver"})
    except Exception:
        pass

    # Fallback to OpenF1 laps/results
    sessions = fetch(
        "sessions",
        {"year": year, "meeting_name": grand_prix, "session_name": "R"},
    )
    if sessions:
        session_key = sessions[0].get("session_key")
        if session_key:
            laps = fetch("laps", {"session_key": session_key})
            if laps:
                df = pd.DataFrame(laps)
                name_col = "driver_name" if "driver_name" in df.columns else None
                if name_col and "driver_number" in df.columns:
                    df = df[["driver_number", name_col]].drop_duplicates()
                    df.rename(
                        columns={"driver_number": "Driver", name_col: "FullName"},
                        inplace=True,
                    )
                    return df

    # Final fallback to full season list
    return list_drivers(year)


def race_name_by_round(year: int, round_number: int) -> str | None:
    """Return Grand Prix name for a season round.

    Attempts the OpenF1 API first and falls back to FastF1/Ergast data.
    """

    sessions = fetch(
        "sessions",
        {"year": year, "round_number": round_number, "session_name": "R"},
    )
    if sessions and isinstance(sessions, list):
        name = sessions[0].get("meeting_name")
        if name:
            return name

    # Fallback to our season schedule which uses the meetings endpoint
    schedule = season_schedule(year)
    for ev in schedule:
        if ev.get("Round") == round_number:
            return ev.get("GrandPrix")

    try:  # pragma: no cover - requires fastf1
        schedule = ergast.fetch_schedule(season=year)
        row = schedule[schedule["RoundNumber"] == round_number]
        if not row.empty:
            return row.iloc[0]["EventName"]
    except Exception:
        pass
    return None


def season_schedule(year: int) -> list[dict]:
    """Return list of Grand Prix names for the season.

    The function now queries the ``meetings`` endpoint first as the
    ``sessions`` endpoint no longer includes 2025 data.  If no events are
    returned it falls back to ``sessions`` and finally to Ergast via
    ``fastf1``.
    """

    events: list[dict] = []

    # Preferred source starting 2025
    meetings = fetch("meetings", {"year": year})
    if not meetings:
        print(f"No results for {year} from /meetings endpoint")
    else:
        try:
            meetings.sort(key=lambda m: m.get("date_start", ""))
        except Exception:
            pass
        round_no = 1
        for m in meetings:
            gp = m.get("meeting_name") or m.get("meeting_official_name")
            if not gp or "grand prix" not in gp.lower():
                continue
            events.append(
                {
                    "GrandPrix": gp,
                    "Round": round_no,
                    "RaceKey": slugify(gp),
                }
            )
            round_no += 1

    # If meetings didn't work, try the legacy sessions endpoint
    if not events:
        sessions = fetch("sessions", {"year": year, "session_name": "R"})
        if not sessions:
            print(f"No results for {year} from /sessions endpoint")
        else:
            for s in sessions:
                gp = s.get("meeting_name")
                rnd = s.get("round_number")
                if gp and rnd is not None:
                    events.append(
                        {
                            "GrandPrix": gp,
                            "Round": int(rnd),
                            "RaceKey": slugify(gp),
                        }
                    )

    if events:
        events.sort(key=lambda x: x["Round"])
        return events

    # Final fallback using Ergast if FastF1 is available
    try:
        schedule = ergast.fetch_schedule(season=year)
        for _, row in schedule.iterrows():
            events.append(
                {
                    "GrandPrix": row["EventName"],
                    "Round": int(row["RoundNumber"]),
                    "RaceKey": slugify(row["EventName"]),
                }
            )
    except Exception:
        pass

    return events


def weather_summary(year: int, grand_prix: str) -> dict:
    """Return average air and track temps for a race.

    The function first attempts to query the ``weather`` endpoint using the
    meeting key of the event.  This is required for the 2025 season as weather
    data is provided per meeting rather than per session.  If no meeting key is
    found it falls back to the older session-based lookup.
    """
    try:
        import pandas as pd
    except Exception:  # pragma: no cover - pandas missing
        return {"air_temp": 0.0, "track_temp": 0.0, "rainfall": 0.0}

    meeting = fetch("meetings", {"year": year, "meeting_name": grand_prix})
    meeting_key = None
    if meeting:
        meeting_key = meeting[0].get("meeting_key")

    data: list = []
    if meeting_key:
        data = fetch("weather", {"meeting_key": meeting_key})

    # Fall back to session-based query if no weather returned
    if not data:
        sessions = fetch(
            "sessions",
            {"year": year, "meeting_name": grand_prix, "session_name": "R"},
        )
        if sessions:
            session_key = sessions[0].get("session_key")
            if session_key:
                data = fetch("weather", {"session_key": session_key})

    if not data:
        return {"air_temp": 0.0, "track_temp": 0.0, "rainfall": 0.0}

    df = pd.DataFrame(data)
    return {
        "air_temp": float(
            df.get("air_temperature", pd.Series(dtype=float)).mean() or 0
        ),
        "track_temp": float(
            df.get("track_temperature", pd.Series(dtype=float)).mean() or 0
        ),
        "rainfall": float(df.get("rainfall", pd.Series(dtype=float)).mean() or 0),
    }


def session_fastest_times(year: int, grand_prix: str) -> "pd.DataFrame":
    """Return Q1/Q2/Q3 times for each driver."""
    try:  # pragma: no cover - requires pandas
        import pandas as pd
    except Exception:  # pragma: no cover - pandas missing
        return __import__("pandas").DataFrame(
            columns=["Driver", "Q1Time_s", "Q2Time_s", "Q3Time_s", "QualifyingTime_s"]
        )

    os.makedirs(SESSION_TIMES_DIR, exist_ok=True)
    csv_file = os.path.join(SESSION_TIMES_DIR, f"{year}_{slugify(grand_prix)}.csv")
    if os.path.exists(csv_file):
        try:
            return pd.read_csv(csv_file)
        except Exception:
            pass

    result = list_drivers(year)[["Driver"]].copy()
    found = False

    # --- FastF1 primary source -------------------------------------------------
    try:
        import fastf1

        fastf1.Cache.enable_cache(os.environ.get("FASTF1_CACHE_DIR", "f1_cache"))
        sess = fastf1.get_session(year, grand_prix, "Q")
        sess.load()
        qres = sess.results[["Abbreviation", "Q1", "Q2", "Q3"]].copy()
        for c in ["Q1", "Q2", "Q3"]:
            qres[c] = pd.to_timedelta(qres[c]).dt.total_seconds()
        qres.rename(
            columns={"Abbreviation": "Driver", "Q1": "Q1Time_s", "Q2": "Q2Time_s", "Q3": "Q3Time_s"},
            inplace=True,
        )
        result = result.merge(qres, on="Driver", how="left")
        found = True
    except Exception:
        pass

    # --- OpenF1 fallback ------------------------------------------------------
    if not found:
        sessions = fetch(
            "sessions",
            {"year": year, "meeting_name": grand_prix, "session_name": "Q"},
        )
        if sessions:
            session_key = sessions[0].get("session_key")
            if session_key:
                res = fetch("results", {"session_key": session_key})
                if res:
                    df = pd.DataFrame(res)
                    cols = []
                    for src, col in [("q1_time", "Q1Time_s"), ("q2_time", "Q2Time_s"), ("q3_time", "Q3Time_s")]:
                        if src in df.columns:
                            df[src] = pd.to_timedelta(df[src]).dt.total_seconds()
                            cols.append(col)
                    if "driver_number" in df.columns and cols:
                        df.rename(
                            columns={
                                "driver_number": "Driver",
                                "q1_time": "Q1Time_s",
                                "q2_time": "Q2Time_s",
                                "q3_time": "Q3Time_s",
                            },
                            inplace=True,
                        )
                        result = result.merge(df[["Driver", "Q1Time_s", "Q2Time_s", "Q3Time_s"]], on="Driver", how="left")
                        found = True

    if found:
        result["QualifyingTime_s"] = result[["Q1Time_s", "Q2Time_s", "Q3Time_s"]].min(axis=1)

    try:
        result.to_csv(csv_file, index=False)
    except Exception:
        pass
    return result


def overtake_potential(years: int | list[int], grand_prix: str) -> float:
    """Return the average overtake potential for the given season(s)."""
    try:
        import pandas as pd
        import fastf1

        fastf1.Cache.enable_cache(os.environ.get("FASTF1_CACHE_DIR", "f1_cache"))
    except Exception:  # pragma: no cover - missing dependencies
        try:
            import pandas as pd  # type: ignore
        except Exception:
            return 0.0

    if isinstance(years, int):
        years = [years]

    vals = []
    for yr in years:
        try:
            session_r = fastf1.get_session(yr, grand_prix, "R")
            session_r.load()
        except Exception:
            continue
        results = session_r.results[["Abbreviation", "GridPosition", "Position"]].copy()
        results[["GridPosition", "Position"]] = results[
            ["GridPosition", "Position"]
        ].apply(pd.to_numeric, errors="coerce")

        total = len(results)
        results["GridPosition"] = results["GridPosition"].fillna(0)
        results.loc[results["GridPosition"] <= 0, "GridPosition"] = total
        results["Position"] = results["Position"].fillna(total)
        results.loc[results["Position"] <= 0, "Position"] = total

        diff = results["GridPosition"] - results["Position"]
        diff = diff[diff > 0]
        if len(diff) == 0:
            continue
        vals.append(float(diff.mean()))

    if not vals:
        # Try Ergast as a fallback if FastF1 provided nothing
        for yr in years:
            try:
                schedule = ergast.fetch_schedule(season=yr)
                row = schedule[schedule["EventName"].str.contains(grand_prix, case=False, regex=False)]
                if row.empty:
                    continue
                rnd = int(row.iloc[0]["RoundNumber"])
                resp = requests.get(
                    f"https://ergast.com/api/f1/{yr}/{rnd}/results.json",
                    timeout=10,
                )
                resp.raise_for_status()
                data = resp.json()
                races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
                if not races:
                    continue
                results = races[0].get("Results", [])
                df = pd.DataFrame({
                    "GridPosition": [int(r.get("grid", 0)) for r in results],
                    "Position": [int(r.get("position", 0)) for r in results],
                })
                diff = df["GridPosition"] - df["Position"]
                diff = diff[diff > 0]
                if len(diff) == 0:
                    continue
                vals.append(float(diff.mean()))
            except Exception:
                continue

    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))
