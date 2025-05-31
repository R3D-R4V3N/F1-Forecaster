from __future__ import annotations


def parse_time_value(value) -> float:
    """Convert a time representation to seconds as float.

    Accepts numeric values or strings in ``m:ss.sss`` format.
    Returns ``0.0`` for empty or invalid values.
    """
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)

    val = str(value).strip()
    if not val:
        return 0.0

    # Try simple float conversion first
    try:
        return float(val)
    except ValueError:
        pass

    if ":" in val:
        mins, secs = val.split(":", 1)
        try:
            return float(mins) * 60.0 + float(secs)
        except ValueError:
            return 0.0

    # As a fallback attempt to remove any stray characters
    try:
        return float(val.replace(":", ""))
    except ValueError:
        return 0.0


def format_time(seconds: float | None) -> str:
    """Return a ``m:ss.sss`` string for a seconds value.

    Returns an empty string when ``seconds`` is ``None`` or ``NaN``.
    """
    try:
        if seconds is None or seconds != seconds:
            return ""
        seconds = float(seconds)
    except Exception:
        return ""

    mins = int(seconds // 60)
    secs = seconds % 60
    if mins:
        return f"{mins}:{secs:06.3f}"
    return f"{secs:.3f}"


# Mapping from common full driver names to the FastF1 3-letter codes.
# This is not exhaustive but covers the 2025 grid and reserve drivers.
DRIVER_NAME_TO_CODE = {
    "Max Verstappen": "VER",
    "Sergio Perez": "PER",
    "Sergio Pérez": "PER",
    "Lewis Hamilton": "HAM",
    "George Russell": "RUS",
    "Charles Leclerc": "LEC",
    "Carlos Sainz": "SAI",
    "Carlos Sainz Jr.": "SAI",
    "Lando Norris": "NOR",
    "Oscar Piastri": "PIA",
    "Fernando Alonso": "ALO",
    "Lance Stroll": "STR",
    "Esteban Ocon": "OCO",
    "Pierre Gasly": "GAS",
    "Yuki Tsunoda": "TSU",
    "Daniel Ricciardo": "RIC",
    "Alexander Albon": "ALB",
    "Logan Sargeant": "SAR",
    "Valtteri Bottas": "BOT",
    "Zhou Guanyu": "ZHO",
    "Guanyu Zhou": "ZHO",
    "Kevin Magnussen": "MAG",
    "Nico Hülkenberg": "HUL",
    "Nico Hulkenberg": "HUL",
    "Oliver Bearman": "BEA",
    "Andrea Kimi Antonelli": "ANT",
    "Isack Hadjar": "HAD",
    "Jack Doohan": "DOO",
    "Gabriel Bortoleto": "BOR",
    "Liam Lawson": "LAW",
}


def normalize_driver_name(name: str) -> str:
    """Return the FastF1 driver code for a name if known."""
    if not name:
        return ""
    name = str(name).strip()
    return DRIVER_NAME_TO_CODE.get(name, name)


def slugify(text: str) -> str:
    """Return a simple slug for a string."""
    import re

    slug = re.sub(r"[^a-z0-9]+", "_", text.lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug
