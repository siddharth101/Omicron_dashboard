"""
Flask + Dash dashboard for exploring SNR and frequency histograms
with start/end date filtering.

"""
from __future__ import annotations

import os
import argparse
from datetime import date
from functools import lru_cache

import pandas as pd

from flask import Flask
from dash import Dash, dcc, html, Input, Output, State, callback_context
import plotly.express as px

# -------------------------------
# Config & Data Loading
# -------------------------------

def _coerce_dates(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", utc=False)
    return dt.dt.date


def load_dataframe(path: str) -> pd.DataFrame:
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


# CLI args
parser = argparse.ArgumentParser(description="SNR/Frequency Dashboard")

# Choose which interferometer
parser.add_argument(
    "--ifo",
    choices=["H1", "L1"],
    default="L1",
    help="IFO to load data for (H1 or L1).",
)



# Optional custom file path (overrides --ifo)
# parser.add_argument(
#     "--data",
#     dest="data_path",
#     default=None,
#     help="Path to CSV/Parquet/Feather file. If not given, defaults to data/{IFO}_data.csv",
# )

parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", default=8050, type=int)
parser.add_argument("--debug", action="store_true")
args, _ = parser.parse_known_args()

ifo = args.ifo

# Decide which file to load
# if args.data_path:
#     data_file = args.data_path
# else:
#     data_file = f"data/{args.ifo}_data.csv"

data_file = f"data/{args.ifo}_data.csv"

# Load data
try:
    DF = load_dataframe(data_file)
except Exception as e:
    print(f"[WARN] Failed to load {data_file}: {e}\nUsing a small sample dataset instead.")
    DF = pd.DataFrame({
        "dates": pd.to_datetime(["2024-12-01", "2024-12-02", "2024-12-03"]).date,
        "snr": [9.4, 31.3, 7.5],
        "frequency": [20.8, 11.5, 60.2],
    })

print(DF)


# parser = argparse.ArgumentParser(description="SNR/Frequency Dashboard")
# parser.add_argument("--data", dest="data_path", default=os.getenv("DATA_PATH", "data/L1_data.csv"),
#                     help="Path to CSV/Parquet/Feather file with required columns.")
# parser.add_argument("--host", default="127.0.0.1")
# parser.add_argument("--port", default=8050, type=int)
# parser.add_argument("--debug", action="store_true")
# args, _ = parser.parse_known_args()

# # Load data
# try:
#     DF = load_dataframe(args.data_path)
# except Exception as e:
#     # Create a tiny placeholder dataset if loading fails, so the app still runs.
#     print(f"[WARN] Failed to load {args.data_path}: {e}\nUsing a small sample dataset instead.")
#     DF = pd.DataFrame({
#         "dates": pd.to_datetime(["2024-12-01", "2024-12-02", "2024-12-03"]).date,
#         "snr": [9.4, 31.3, 7.5],
#         "frequency": [20.8, 11.5, 60.2],
#     })

MIN_DATE: date = DF["dates"].min()
MAX_DATE: date = DF["dates"].max()

# -------------------------------
# Flask + Dash app
# -------------------------------
server = Flask(__name__)
app = Dash(
    __name__,
    server=server,
    title="SNR & Frequency Dashboard",
    suppress_callback_exceptions=True,
)

app.layout = html.Div(
    className="container mx-auto p-4",
    children=[
        html.H1("SNR & Frequency Distribution", style={"marginBottom": 8}),
        html.Div(
            style={"display": "flex", "gap": "1rem", "flexWrap": "wrap", "alignItems": "center"},
            children=[
                dcc.DatePickerRange(
                    id="date-range",
                    min_date_allowed=MIN_DATE,
                    max_date_allowed=MAX_DATE,
                    start_date=MIN_DATE,
                    end_date=MAX_DATE,
                    display_format="YYYY-MM-DD",
                    with_full_screen_portal=False,
                ),
                dcc.Slider(
                    id="bins",
                    min=10, max=150, step=1, value=40,
                    marks={10: "10", 40: "40", 80: "80", 120: "120", 150: "150"},
                    tooltip={"always_visible": False},
                ),
                dcc.Checklist(
                    id="logy",
                    options=[{"label": "Log Y", "value": "log"}],
                    value=[],
                    style={"marginLeft": "0.5rem"},
                ),
                html.Div(id="record-count", style={"fontStyle": "italic"})
            ],
        ),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gridTemplateRows": "auto auto", "gap": "1rem", "marginTop": "1rem"},
            children=[
                dcc.Graph(id="hist-snr-1", config={"displayModeBar": True}),       # SNR 0–200
                dcc.Graph(id="hist-frequency-1", config={"displayModeBar": True}), # Freq 0–100 Hz
                dcc.Graph(id="hist-snr-2", config={"displayModeBar": True}),       # SNR 200–4000
                dcc.Graph(id="hist-frequency-2", config={"displayModeBar": True}), # Freq 100–200 Hz
            ],
        ),
        html.Hr(),
        html.Details([
            html.Summary("Data & App Info"),
            html.Pre(
                id="meta-info",
                style={"whiteSpace": "pre-wrap", "fontSize": "0.85rem", "opacity": 0.8},
            ),
        ]),
    ],
)

colors = {'L1':{'snr':'dodgerblue', 'frequency':'mediumpurple'},
           'H1':{'snr':'orangered', 'frequency':'lightcoral'}}
@lru_cache(maxsize=256)
def _filter_key(start: str, end: str) -> tuple[date, date]:
    """Cache-friendly key by converting to date objects."""
    s = pd.to_datetime(start).date() if start else MIN_DATE
    e = pd.to_datetime(end).date() if end else MAX_DATE
    return s, e


def filter_df(start_date: str | None, end_date: str | None) -> pd.DataFrame:
    s, e = _filter_key(str(start_date), str(end_date))
    mask = (DF["dates"] >= s) & (DF["dates"] <= e)
    return DF.loc[mask]


from plotly import graph_objects as go

@app.callback(
    [
        Output("hist-snr-1", "figure"),
        Output("hist-snr-2", "figure"),
        Output("hist-frequency-1", "figure"),
        Output("hist-frequency-2", "figure"),
        Output("record-count", "children"),
        Output("meta-info", "children"),
    ],
    [
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
        Input("bins", "value"),   # kept for compatibility, but unused now
        Input("logy", "value"),
    ],
)


def update_plots(start_date, end_date, _bins_unused, logy_values):
    dff = filter_df(start_date, end_date)
    log_y = "log" in (logy_values or [])

    def make_hist(x, title, x_title, x_start, x_end, x_step, color='dodgerblue'):
        fig = go.Figure(
            data=[go.Histogram(x=x, xbins=dict(start=x_start, end=x_end, size=x_step),
                               marker=dict(color=color))]
        )
        fig.update_layout(
            title=title,
            xaxis_title=x_title,
            yaxis_title="Count",
            bargap=0.1,
        )
        fig.update_xaxes(range=[x_start, x_end])
        if log_y:
            fig.update_yaxes(type="log")
        return fig

    # SNR
    fig_snr_1 = make_hist(
        dff["snr"],
        "SNR Distribution (0–200)",
        "SNR",
        0, 200, 20, color=colors[ifo]['snr']
    )
    fig_snr_2 = make_hist(
        dff["snr"],
        "SNR Distribution (200–4000)",
        "SNR",
        200, 4000, 200, color=colors[ifo]['snr']
    )

    # Frequency
    fig_freq_1 = make_hist(
        dff["frequency"],
        "Frequency Distribution (10–100 Hz)",
        "Frequency [Hz]",
        10, 100, 10, color=colors[ifo]['frequency']
    )
    fig_freq_2 = make_hist(
        dff["frequency"],
        "Frequency Distribution (100–2000 Hz)",
        "Frequency [Hz]",
        100, 2000, 100, color=colors[ifo]['frequency']
    )

    s, e = _filter_key(str(start_date), str(end_date))
    info = (
        #f"Data path: {os.path.abspath(args.data_path)}\n"
        f"Data path: {data_file}\n"
        f"Rows total: {len(DF):,}\n"
        f"Rows in range [{s} .. {e}]: {len(dff):,}\n"
        f"Date span in file: {MIN_DATE} .. {MAX_DATE}"
    )
    count_txt = f"Showing {len(dff):,} rows"

    return fig_snr_1, fig_snr_2, fig_freq_1, fig_freq_2, count_txt, info


if __name__ == "__main__":
    # For Gunicorn/WSGI: expose `server`
    print(f"Loaded data rows: {len(DF):,} | Dates: {MIN_DATE} .. {MAX_DATE}")
    app.run(host=args.host, port=args.port, debug=args.debug)
