import pandas as pd
from datetime import date
import os
from typing import Tuple

def _coerce_dates(s: pd.Series) -> pd.Series:
    """
    Coerce a Pandas Series to datetime.date (no timezone).
    """
    dt = pd.to_datetime(s, errors="coerce", utc=False)
    return dt.dt.date


def load_dataframe(path: str) -> pd.DataFrame:
    """
    Load SNR/frequency dataframe from CSV/Parquet/Feather and normalize columns.

    Expected columns: 'dates', 'snr', 'frequency'
    - 'dates' is converted to datetime.date
    - 'snr' and 'frequency' are downcasted to save memory
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    elif ext in (".feather", ".ft"):
        df = pd.read_feather(path)
    else:
        # default assume CSV
        df = pd.read_csv(path)

    # Normalize required columns
    required = {"dates", "snr", "frequency"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["dates"] = _coerce_dates(df["dates"])  # -> datetime.date

    # Downcast numerics to save memory for large files
    for col in ("snr", "frequency"):
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="float")
        elif pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        else:
            # attempt to coerce
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows that failed coercion
    df = df.dropna(subset=["dates", "snr", "frequency"]).reset_index(drop=True)
    return df

def _filter_key(df: pd.DataFrame, start: str | None, end: str | None) -> Tuple[date, date]:
    """
    Take optional start/end (as strings or None) and return a concrete
    (start_date, end_date) tuple, falling back to full range of df["dates"].
    """
    # Guard against 'None' string coming from Dash
    if not start or start == "None":
        s = df["dates"].min()
    else:
        s = pd.to_datetime(start).date()

    if not end or end == "None":
        e = df["dates"].max()
    else:
        e = pd.to_datetime(end).date()

    return s, e


def filter_df(df: pd.DataFrame, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    """
    Filter df between start_date and end_date (inclusive) using the 'dates' column.
    """
    s, e = _filter_key(df, start_date, end_date)
    mask = (df["dates"] >= s) & (df["dates"] <= e)
    return df.loc[mask]
