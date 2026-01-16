"""
streamlit_recruiter_app.py

Recruiter-friendly Streamlit app for a Cross-Sectional Equity (CSE) Alpha Model.
- Uses *precomputed* outputs from your pipeline:
  - outputs/ic_scores_lgb.csv  (daily rank IC time series)
  - outputs/val_df_lgb.csv     (validation panel with y_val, y_val_pred, y_val_pred_neut, rank)

Core interactions:
1) User selects a ticker and a date.
2) App shows:
   - The IC score for that date (cross-sectional IC; same for all tickers on that date)
   - The model's rank for that ticker on that date (based on Neutralized Alpha Score by default)
   - A price chart for that ticker (around the selected date), capped to <= Jan 1, 2022

Run:
  streamlit run streamlit_recruiter_app_v7.py

Notes:
- This app does NOT retrain the model (model is pre-run).
- It pulls price data live via yfinance for the selected ticker.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt


st.set_page_config(page_title="CSE Alpha Model Showcase", layout="wide")


# ----------------------------
# Paths (expected project structure)
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
IC_PATH = BASE_DIR / "outputs" / "ic_scores_lgb.csv"
VAL_PATH = BASE_DIR / "outputs" / "val_df_lgb.csv"

# Charts show data no later than Jan 1, 2022
CUTOFF_DATE = pd.Timestamp("2022-01-01")

# Display names for recruiter-facing UI
SCORE_LABELS = {
    "y_val_pred_neut": "Neutralized Alpha Score",
    "y_val_pred": "Raw Alpha Score",
}
LABEL_TO_COL = {v: k for k, v in SCORE_LABELS.items()}


# ----------------------------
# Data loading helpers
# ----------------------------
@st.cache_data(show_spinner=False)
def load_ic_scores(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Handle either explicit "date" col or index-saved first column
    if "date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"])

    if "ic_scores" not in df.columns:
        if len(df.columns) < 2:
            raise ValueError("IC CSV must contain at least two columns: date and ic_scores.")
        df = df.rename(columns={df.columns[1]: "ic_scores"})

    df["ic_scores"] = pd.to_numeric(df["ic_scores"], errors="coerce")
    df = df.dropna(subset=["date", "ic_scores"]).sort_values("date").reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_val_panel(path: str) -> pd.DataFrame:
    """
    Loads val_df_lgb.csv. Assumes it was saved with index=True, so (date, ticker) are likely in index cols.
    Supports:
      - MultiIndex columns saved as first two columns
      - Or explicit date/ticker columns
    """
    df = pd.read_csv(path)

    cols_lower = [c.lower() for c in df.columns]
    if "date" in cols_lower and "ticker" in cols_lower:
        df["date"] = pd.to_datetime(df["date"])
        df["ticker"] = df["ticker"].astype(str)
        out = df
    else:
        if len(df.columns) < 3:
            raise ValueError("val_df_lgb.csv must contain (date,ticker) plus prediction columns.")
        df = df.rename(columns={df.columns[0]: "date", df.columns[1]: "ticker"})
        df["date"] = pd.to_datetime(df["date"])
        df["ticker"] = df["ticker"].astype(str)
        out = df

    for c in ["y_val", "y_val_pred"]:
        if c not in out.columns:
            raise ValueError(f"Missing column '{c}' in val_df_lgb.csv")

    if "y_val_pred_neut" not in out.columns:
        out["y_val_pred_neut"] = np.nan
    if "rank" not in out.columns:
        out["rank"] = np.nan

    for c in ["y_val", "y_val_pred", "y_val_pred_neut", "rank"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["date", "ticker"]).sort_values(["date", "ticker"]).reset_index(drop=True)
    return out


@st.cache_data(show_spinner=False)
def fetch_price_series(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """
    Fetch adjusted close via yfinance. Returns Series indexed by date.
    """
    data = yf.download(
        tickers=ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    if data is None or data.empty:
        return pd.Series(dtype=float)
    if "Adj Close" in data.columns:
        s = data["Adj Close"].dropna()
    else:
        s = data.iloc[:, 0].dropna()
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def ic_summary(ic: pd.Series) -> dict:
    ic = ic.dropna()
    n = int(ic.shape[0])
    if n == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "ic_ir": np.nan, "hit": np.nan}
    mean = float(ic.mean())
    std = float(ic.std(ddof=1))
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ic_ir": (mean / std) if std > 0 else np.nan,
        "hit": float((ic > 0).mean()),
    }


def plot_price(prices: pd.Series, title: str) -> plt.Figure:
    fig, ax = plt.subplots()
    ax.plot(prices.index, prices.values)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Adj Close")
    fig.autofmt_xdate()
    return fig


def plot_ic_timeseries(ic_df: pd.DataFrame, roll: int = 60) -> plt.Figure:
    fig, ax = plt.subplots()
    ax.plot(ic_df["date"], ic_df["ic_scores"], label="Daily Rank IC")
    ax.plot(ic_df["date"], ic_df["ic_scores"].rolling(roll).mean(), label=f"{roll}D rolling mean")
    ax.axhline(0, linewidth=1)
    ax.set_title("Daily Rank IC Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("IC")
    ax.legend()
    fig.autofmt_xdate()
    return fig


# ----------------------------
# UI: Title / bio
# ----------------------------
st.title("Cross-Sectional Equity Alpha Model")
st.caption("By Jacob Emerson")
st.caption(" ")
st.caption(
    "This app demonstrates a cross-sectional equity ranking model trained on historical data from all S&P 500 stocks. "
    "Instead of predicting exact prices, the model scores and ranks stocks from most to least attractive each day. "
    "Performance is evaluated by whether higher-ranked stocks tend to outperform lower-ranked ones over time, "
    "with the goal of identifying consistent signals that can support long–short investment strategies."
)

# ----------------------------
# Controls (sidebar)
# ----------------------------
with st.sidebar:
    st.header("Controls")
    pred_choice_label = st.selectbox(
        "Ranking score",
        options=["Neutralized Alpha Score", "Raw Alpha Score"],
        index=0,
        help="Cross-sectional rank is computed within each date using this score.",
    )
    price_window_days = st.slider("Price chart window (days)", 30, 365, 180, 30)
    ic_roll = st.selectbox("IC rolling window (days)", [20, 60, 120, 252], index=1)

pred_choice = LABEL_TO_COL[pred_choice_label]  # internal column name


# ----------------------------
# Load data (fail fast)
# ----------------------------
try:
    ic_df = load_ic_scores(str(IC_PATH))
except Exception as e:
    st.error(f"Failed to load IC CSV at {IC_PATH}: {e}")
    st.stop()

try:
    val_df = load_val_panel(str(VAL_PATH))
except Exception as e:
    st.error(f"Failed to load validation panel CSV at {VAL_PATH}: {e}")
    st.stop()

# Always compute rank_view based on chosen score column
tmp = val_df.copy()
tmp["rank_view"] = tmp.groupby("date")[pred_choice].rank(method="first", ascending=False)

available_dates = sorted(tmp["date"].unique())
available_tickers = sorted(tmp["ticker"].unique())

if not available_dates or not available_tickers:
    st.error("No dates or tickers found in val_df_lgb.csv.")
    st.stop()

# ----------------------------
# 1) Query a ticker + date
# ----------------------------
st.caption(" ")
st.markdown("## 1) Select Ticker and Date")

q1, q2 = st.columns(2)
with q1:
    default_idx = available_tickers.index("AAPL") if "AAPL" in available_tickers else 0
    ticker = st.selectbox("Select ticker", options=available_tickers, index=default_idx)

with q2:
    # Show clean YYYY-MM-DD (no time) in dropdown
    date_strs = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in available_dates]
    date_str = st.selectbox("Select date", options=date_strs, index=len(date_strs) - 1)

date = pd.Timestamp(date_str)

# ----------------------------
# 2) Stock trading performance (price chart)
# ----------------------------
st.markdown("## 2) Stock Trading Performance")

# Cap chart to show no data later than Jan 1, 2022
anchor_date = min(date, CUTOFF_DATE)

start = anchor_date - pd.Timedelta(days=int(price_window_days))
end = CUTOFF_DATE + pd.Timedelta(days=1)  # yfinance end is exclusive

with st.spinner("Fetching price data (yfinance)..."):
    prices = fetch_price_series(ticker, start, end)

if prices.empty:
    st.warning("Could not fetch price data for this ticker/date window.")
else:
    st.pyplot(
        plot_price(prices, title=f"{ticker} Adj Close "),
        clear_figure=True,
    )

# ----------------------------
# 3) Results for selection
# ----------------------------
st.caption(" ")
st.markdown("## 3) Results")

ic_row = ic_df.loc[ic_df["date"] == date, "ic_scores"]
ic_value = float(ic_row.iloc[0]) if len(ic_row) else np.nan

row = tmp.loc[(tmp["date"] == date) & (tmp["ticker"] == ticker)]
if row.empty:
    st.warning("No prediction available for that (date, ticker) in your validation panel.")
else:
    row0 = row.iloc[0]
    rank_val = row0["rank_view"]
    score_val = row0[pred_choice]
    y_true = row0["y_val"]

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Daily IC (that date)", f"{ic_value:.4f}" if np.isfinite(ic_value) else "—")
    r2.metric(pred_choice_label, f"{score_val:.4f}" if np.isfinite(score_val) else "—")
    r3.metric("Cross-sectional rank", f"{int(rank_val)}" if np.isfinite(rank_val) else "—")
    r4.metric("Realized target (y_val)", f"{y_true:.4f}" if np.isfinite(y_true) else "—")

st.caption(
    "Daily IC shows whether the model’s rankings worked across all S&P 500 stocks on that day. "
    "The Neutralized Alpha Score reflects how attractive the selected stock looked after removing broad market effects. "
    "Cross-Sectional Rank shows the stock’s position among all stocks for that day. "
    "Realized Target shows what actually happened after the ranking was made."
)


# ----------------------------
# 4) Model performance snapshot
# ----------------------------
st.caption(" ")
st.markdown("## 4) Model Performance")

summ = ic_summary(ic_df["ic_scores"])
k1, k2, k3, k4 = st.columns(4)
k1.metric("IC Mean", f"{summ['mean']:.4f}" if np.isfinite(summ["mean"]) else "—")
k2.metric("IC IR (Mean/Std)", f"{summ['ic_ir']:.3f}" if np.isfinite(summ["ic_ir"]) else "—")
k3.metric("Hit Rate (IC>0)", f"{summ['hit']:.1%}" if np.isfinite(summ["hit"]) else "—")
k4.metric("Days", f"{summ['n']}")

st.caption(
    "IC Mean shows the model’s average ranking accuracy across all days. "
    "IC IR (Mean/Std) measures how consistent that ranking performance is over time. "
    "Hit Rate (IC>0) shows how often the model ranked stocks in the correct direction. "
    "Days indicates the number of trading days used to evaluate the model."
)

# ----------------------------
# 5) IC time series + narrative
# ----------------------------
st.caption(" ")
st.markdown("## 5) IC Analysis")
left, right = st.columns([2, 1])

with left:
    st.pyplot(plot_ic_timeseries(ic_df, roll=int(ic_roll)), clear_figure=True)

with right:
    st.subheader("Breakdown")
    st.write(
        "- **What this chart shows:** how consistently the model ranks stocks in the right order.\n"
        "- **Positive IC is good:** it means higher-ranked stocks tended to perform better than lower-ranked stocks.\n"
        "- **Daily results are noisy:** short-term ups/downs are normal in markets.\n"
        "- **The rolling line is the key:** it shows whether performance is *consistently* positive over time.\n"
        f"- **This demo uses {pred_choice_label}:** the score used to rank stocks each day.\n"
        "- **Why this matters:** consistent ranking skill is the foundation of long/short equity strategies.\n"
        "- **Takeaway for recruiters:** this project demonstrates an end-to-end workflow: data → features → model → evaluation."
    )


# ----------------------------
# 6) Cross-Sectional Rankings for Selected Date
# ----------------------------

st.markdown("## 6) Cross-Sectional Rankings for Selected Date")

topn = st.slider("Show Top / Bottom N", 5, 50, 15, 5)

day_panel = tmp.loc[
    tmp["date"] == date,
    ["ticker", pred_choice, "rank_view", "y_val"]
].dropna(subset=[pred_choice]).copy()

day_panel = day_panel.sort_values(pred_choice, ascending=False)

# Display-friendly column names in table
day_panel = day_panel.rename(columns={
    "ticker": "Ticker",
    pred_choice: pred_choice_label,
    "rank_view": "Rank",
    "y_val": "Realized Target",
})


c_top, c_bot = st.columns(2)

with c_top:
    st.caption(f"Top {topn} by {pred_choice_label}")
    st.dataframe(day_panel.head(topn), use_container_width=True)

with c_bot:
    st.caption(f"Bottom {topn} by {pred_choice_label}")
    st.dataframe(
        day_panel.tail(topn).sort_values(pred_choice_label, ascending=True),
        use_container_width=True
    )

st.caption(
    "Tip: A common baseline is **long top decile** and **short bottom decile** by score."
)

