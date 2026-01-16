"""
cse_alpha_model.py

Pipeline:
- Load S&P 500 constituents
- Download OHLCV via yfinance
- Build features
- Winsorize + cross-sectional z-score
- Train LightGBM
- Neutralize predictions per date vs exposures
- Save daily rank ICs
- Save val_df_lgb.csv including per-date ranks
- Utility: get_stock_rank(date, ticker)
- Utility: plot_stock_price_train_window(ticker)
"""

from __future__ import annotations

import os
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import yfinance as yf

import matplotlib.pyplot as plt
import matplotlib.dates as mdates


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "constituents.csv")

OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

IC_PATH = os.path.join(OUT_DIR, "ic_scores_lgb.csv")
VAL_DF_PATH = os.path.join(OUT_DIR, "val_df_lgb.csv")

START_DATE = "2018-01-01"
END_DATE = "2024-01-01"

EPS = 1e-9


def to_yahoo_ticker(t: str) -> str:
    return t.replace(".", "-")


def download_prices_chunked(
    tickers_yf: list[str],
    start: str,
    end: str,
    chunk: int = 50,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for i in range(0, len(tickers_yf), chunk):
        part = tickers_yf[i : i + chunk]
        raw = yf.download(
            tickers=part,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            threads=True,
        )
        price_cols = ["Adj Close", "Close", "High", "Low", "Open", "Volume"]
        raw = raw.loc[:, price_cols]

        df_part = (
            raw.stack(level=1, future_stack=True)
            .rename_axis(index=["date", "ticker"])
            .sort_index()
            .rename(
                columns={
                    "Adj Close": "adj_close",
                    "Close": "close",
                    "High": "high",
                    "Low": "low",
                    "Open": "open",
                    "Volume": "volume",
                }
            )
        )
        frames.append(df_part)

    df = pd.concat(frames).apply(pd.to_numeric, errors="coerce")
    return df.sort_index(level=["date", "ticker"])


def drop_tickers_without_price_data(df: pd.DataFrame, price_col: str = "adj_close") -> pd.DataFrame:
    has_data = df[price_col].groupby(level="ticker").apply(lambda s: s.notna().any())
    good = has_data[has_data].index
    return df[df.index.get_level_values("ticker").isin(good)]


def fetch_info_map_safe(tickers: list[str], key: str) -> pd.Series:
    out: dict[str, object] = {}
    for t in tickers:
        try:
            out[t] = yf.Ticker(t).info.get(key)
        except Exception:
            out[t] = np.nan
    return pd.Series(out, name=key)


def sector_map_from_constituents(stocks: pd.DataFrame, tickers: list[str]) -> pd.Series:
    for col in ("Sector", "sector", "GICS Sector", "GICS_Sector"):
        if col in stocks.columns:
            s = stocks.set_index("Symbol")[col]
            s.name = "sector"
            return s
    return pd.Series(index=tickers, data=np.nan, name="sector")


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby(level="ticker", group_keys=False)

    df["ret_1"] = np.log(df["adj_close"]).groupby(level="ticker").diff()
    df["dollar_volume"] = df["adj_close"] * df["volume"]

    for w in (5, 21, 63):
        df[f"mom_{w}"] = g["ret_1"].rolling(w).sum().reset_index(level=0, drop=True)

    df["mom_252"] = g["ret_1"].rolling(252).sum().reset_index(level=0, drop=True)
    df["mom_252_21"] = df["mom_252"] - g["ret_1"].rolling(21).sum().reset_index(level=0, drop=True)

    roll60_mean = g["ret_1"].rolling(60).mean().reset_index(level=0, drop=True)
    roll60_std = g["ret_1"].rolling(60).std(ddof=0).reset_index(level=0, drop=True)
    df["sharpe_60"] = roll60_mean / (roll60_std + EPS)

    df["rev_1"] = -df["ret_1"]
    df["rev_5"] = -g["ret_1"].rolling(5).sum().reset_index(level=0, drop=True)

    df["vol_20"] = g["ret_1"].rolling(20).std(ddof=0).reset_index(level=0, drop=True)
    df["vol_60"] = g["ret_1"].rolling(60).std(ddof=0).reset_index(level=0, drop=True)

    prev_close = g["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["true_range"] = tr
    df["atr_14"] = g["true_range"].rolling(14).mean().reset_index(level=0, drop=True)
    df["atr_14_pct"] = df["atr_14"] / df["adj_close"]

    dv20 = g["dollar_volume"].rolling(20).mean().reset_index(level=0, drop=True)
    df["log_dollar_vol_20"] = np.log(dv20)

    amihud = df["ret_1"].abs() / df["dollar_volume"]
    df["amihud_20"] = (
        amihud.groupby(level="ticker").rolling(20).mean().reset_index(level=0, drop=True)
    )

    ma20 = g["adj_close"].rolling(20).mean().reset_index(level=0, drop=True)
    df["ma_gap_20"] = (df["adj_close"] - ma20) / ma20

    def _rsi_14_per_ticker(s: pd.Series) -> pd.Series:
        d = s.diff()
        up = d.clip(lower=0.0)
        down = (-d.clip(upper=0.0))
        roll_up = up.ewm(alpha=1 / 14, adjust=False).mean()
        roll_down = down.ewm(alpha=1 / 14, adjust=False).mean()
        rs = roll_up / (roll_down + EPS)
        return 100 - (100 / (1 + rs))

    df["rsi_14"] = (
        df.groupby(level="ticker", group_keys=False)["adj_close"].apply(_rsi_14_per_ticker)
    )

    hi20 = g["high"].rolling(20).max().reset_index(level=0, drop=True)
    lo20 = g["low"].rolling(20).min().reset_index(level=0, drop=True)
    df["high_low_20"] = (hi20 - lo20) / df["adj_close"]

    return df


def add_size_and_beta(df: pd.DataFrame, shares_out: pd.Series) -> pd.DataFrame:
    df = df.copy()

    tmp = df.reset_index()
    tmp["log_mkt_cap"] = np.log(tmp["adj_close"] * tmp["ticker"].map(shares_out))
    df = tmp.set_index(["date", "ticker"]).sort_index()

    spy = yf.download(
        tickers="SPY",
        start=df.index.get_level_values("date").min(),
        end=df.index.get_level_values("date").max(),
        auto_adjust=False,
        progress=False,
    )["Adj Close"].dropna()

    spy_ret = np.log(spy / spy.shift(1)).sort_index()
    dates = df.index.get_level_values("date")
    df["spy_ret_1"] = spy_ret.reindex(dates).to_numpy()

    def _beta_60_one_ticker(g: pd.DataFrame) -> pd.Series:
        r = g["ret_1"]
        m = g["spy_ret_1"]
        cov = r.rolling(60).cov(m)
        var = m.rolling(60).var(ddof=0)
        return cov / var

    df["beta_60"] = (
        df.groupby(level="ticker", group_keys=False)
        .apply(_beta_60_one_ticker)
        .astype(float)
    )

    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby(level="ticker", group_keys=False)

    df["fwd_ret_1"] = g["ret_1"].shift(-1)
    df["fwd_ret_1_vol_scaled"] = df["fwd_ret_1"] / (df["vol_20"] + 1e-8)

    df["y_vol_scaled"] = df.groupby(level="date")["fwd_ret_1_vol_scaled"].transform(
        lambda x: (x - x.mean()) / x.std(ddof=0)
    )
    return df


def winsorize_by_date(
    df: pd.DataFrame,
    cols: list[str],
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> pd.DataFrame:
    out = df.copy()
    dates = out.index.get_level_values("date")

    for c in cols:
        s = pd.to_numeric(out[c], errors="coerce")
        lo = s.groupby(level="date").quantile(lower_q)
        hi = s.groupby(level="date").quantile(upper_q)
        out[c] = s.clip(
            lower=lo.reindex(dates).to_numpy(),
            upper=hi.reindex(dates).to_numpy(),
        )
    return out


def zscore_by_date(df: pd.DataFrame, cols: list[str], eps: float = 1e-12) -> pd.DataFrame:
    out = df[cols].copy()
    dates = out.index.get_level_values("date")

    for c in cols:
        g = out[c].groupby(level="date")
        mu = g.mean().reindex(dates).to_numpy()
        sd = g.std(ddof=0).reindex(dates).to_numpy()
        out[c] = (out[c].to_numpy() - mu) / np.where(sd > eps, sd, np.nan)
    return out


def drop_na_per_day(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(level="date", group_keys=False).apply(lambda x: x.dropna(how="any"))


def test_train_split(
    df: pd.DataFrame,
    label_col: str,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    train_years: int = 4,
    val_years: int = 1,
    test_years: int = 1,
):
    data = df.copy()

    if not isinstance(data.index, pd.MultiIndex) or data.index.names[:2] != ["date", "ticker"]:
        raise ValueError("df.index must be MultiIndex with names ['date','ticker']")

    data.index = data.index.set_levels(
        [pd.to_datetime(data.index.levels[0]), data.index.levels[1]],
        level=["date", "ticker"],
    )

    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    d = data.index.get_level_values("date")
    data = data[(d >= start_date) & (d <= end_date)]
    if data.empty:
        raise ValueError("No data in [start_date, end_date].")

    max_date = data.index.get_level_values("date").max()
    data = data[data.index.get_level_values("date") < max_date]
    if data.empty:
        raise ValueError("Empty after dropping last date for label safety.")

    max_date = data.index.get_level_values("date").max()
    last_full_year = max_date.year if max_date.month == 12 else (max_date.year - 1)

    test_start_year = last_full_year - (test_years - 1)
    val_start_year = test_start_year - val_years
    train_start_year = val_start_year - train_years

    train_start = pd.Timestamp(f"{train_start_year}-01-01")
    train_end = pd.Timestamp(f"{val_start_year - 1}-12-31")

    val_start = pd.Timestamp(f"{val_start_year}-01-01")
    val_end = pd.Timestamp(f"{test_start_year - 1}-12-31")

    test_start = pd.Timestamp(f"{test_start_year}-01-01")
    test_end = pd.Timestamp(f"{last_full_year}-12-31")

    d = data.index.get_level_values("date")
    train = data[(d >= train_start) & (d <= train_end)]
    val = data[(d >= val_start) & (d <= val_end)]
    test = data[(d >= test_start) & (d <= test_end)]

    for name, part, s, e in [
        ("train", train, train_start, train_end),
        ("val", val, val_start, val_end),
        ("test", test, test_start, test_end),
    ]:
        if part.empty:
            raise ValueError(f"{name} split is empty. Requested: {s.date()}..{e.date()}")

    if label_col not in train.columns:
        raise KeyError(f"Label '{label_col}' not found in columns.")

    X_train, y_train = train.drop(columns=[label_col]), train[label_col]
    X_val, y_val = val.drop(columns=[label_col]), val[label_col]
    X_test, y_test = test.drop(columns=[label_col]), test[label_col]
    return X_train, y_train, X_val, y_val, X_test, y_test


def neutralize_by_date(
    df: pd.DataFrame,
    pred_col: str,
    exposure: pd.Series | pd.DataFrame | None = None,
    exposure_col: str = "beta_60",
    out_col: str = "pred_neut",
    add_intercept: bool = True,
    ridge: float = 1e-6,
    winsorize_pct: float | None = 0.01,
    min_names: int = 30,
) -> pd.DataFrame:
    if not isinstance(df.index, pd.MultiIndex) or df.index.names[:2] != ["date", "ticker"]:
        raise ValueError("df.index must be MultiIndex with names ['date','ticker']")
    if pred_col not in df.columns:
        raise KeyError(f"'{pred_col}' not found in df.columns")

    if exposure is None:
        if exposure_col not in df.columns:
            raise KeyError(f"'{exposure_col}' not found and exposure=None")
        expo = df[[exposure_col]].copy()
    else:
        if isinstance(exposure, pd.Series):
            expo = exposure.reindex(df.index).to_frame(exposure.name or exposure_col)
        else:
            expo = exposure.reindex(df.index).copy()

    work = pd.concat([df[[pred_col]], expo], axis=1)

    def _winsorize(g: pd.DataFrame) -> pd.DataFrame:
        if not winsorize_pct:
            return g
        lo = g.quantile(winsorize_pct)
        hi = g.quantile(1 - winsorize_pct)
        return g.clip(lo, hi, axis=1)

    def _one_date(g: pd.DataFrame) -> pd.Series:
        if len(g) < min_names:
            return pd.Series(index=g.index, data=np.nan, dtype=float)

        g2 = g.dropna()
        if len(g2) < min_names:
            return pd.Series(index=g.index, data=np.nan, dtype=float)

        g2 = _winsorize(g2)

        y = g2[pred_col].to_numpy(dtype=float).reshape(-1, 1)
        X = g2.drop(columns=[pred_col]).to_numpy(dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n = X.shape[0]
        if add_intercept:
            X = np.column_stack([np.ones(n), X])

        XtX = X.T @ X
        reg = ridge * np.eye(XtX.shape[0])
        if add_intercept:
            reg[0, 0] = 0.0

        beta = np.linalg.solve(XtX + reg, X.T @ y)
        resid = (y - X @ beta).ravel()

        out = pd.Series(index=g.index, data=np.nan, dtype=float)
        out.loc[g2.index] = resid
        return out

    neut = work.groupby(level="date", group_keys=False).apply(_one_date)
    out = df.copy()
    out[out_col] = neut
    return out


def daily_rank_ic(df: pd.DataFrame, pred_col: str, target_col: str) -> pd.Series:
    tmp = df.reset_index()
    return (
        tmp.groupby("date")
        .apply(lambda x: x[pred_col].corr(x[target_col], method="spearman"))
        .dropna()
    )


def get_stock_rank(
    df: pd.DataFrame,
    date: str | pd.Timestamp,
    ticker: str,
    pred_col: str = "y_val_pred_neut",
    rank_col: str = "rank",
) -> int | None:
    """
    Return cross-sectional rank of a stock on a given date based on pred_col.
    Works if date/ticker are columns OR in a MultiIndex.

    Rank convention: 1 = highest prediction on that date.
    """
    date = pd.Timestamp(date)

    # Case 1: columns
    if "date" in df.columns and "ticker" in df.columns:
        work = df
        if rank_col not in work.columns:
            work = work.copy()
            work[rank_col] = work.groupby("date")[pred_col].rank(method="first", ascending=False)

        row = work.loc[(work["date"] == date) & (work["ticker"] == ticker), rank_col]
        return None if row.empty else int(row.iloc[0])

    # Case 2: MultiIndex
    if isinstance(df.index, pd.MultiIndex) and set(["date", "ticker"]).issubset(df.index.names):
        work = df
        if rank_col not in work.columns:
            work = work.copy()
            work[rank_col] = work.groupby(level="date")[pred_col].rank(method="first", ascending=False)

        try:
            return int(work.loc[(date, ticker), rank_col])
        except KeyError:
            return None

    raise KeyError("Need 'date'/'ticker' as columns or MultiIndex levels.")


def plot_stock_price_train_window(
    price_df: pd.DataFrame,
    ticker: str,
    start_date: str | pd.Timestamp = START_DATE,
    end_date: str | pd.Timestamp = END_DATE,
    price_col: str = "adj_close",
    val_years: int = 1,
    test_years: int = 1,
):
    """
    Plot the stock price for *training window only* (excludes val+test years at the end).
    """
    if not isinstance(price_df.index, pd.MultiIndex) or price_df.index.names[:2] != ["date", "ticker"]:
        raise ValueError("price_df.index must be MultiIndex with names ['date','ticker']")

    price_df = price_df.copy()
    price_df.index = price_df.index.set_levels(pd.to_datetime(price_df.index.levels[0]), level="date")

    start_date, end_date = pd.Timestamp(start_date), pd.Timestamp(end_date)
    d = price_df.index.get_level_values("date")
    sub = price_df[(d >= start_date) & (d <= end_date)]
    if sub.empty:
        raise ValueError("No data in the provided date window.")

    # Define training end as (end of dataset) - (val_years + test_years)
    max_date = sub.index.get_level_values("date").max()
    train_end = max_date - pd.DateOffset(years=val_years + test_years)

    g = sub.xs(ticker, level="ticker").sort_index()
    g = g[(g.index >= start_date) & (g.index <= train_end)]
    if g.empty:
        raise ValueError(f"No data for {ticker} in the training window.")

    fig, ax = plt.subplots()
    ax.plot(g.index, g[price_col])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%Y"))
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(f"{ticker} Stock Price (Train Window): {start_date.date()} → {train_end.date()}")
    plt.tight_layout()
    plt.show()


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)

    stocks = pd.read_csv(DATA_PATH)
    tickers = stocks["Symbol"].astype(str).tolist()
    tickers_yf = [to_yahoo_ticker(t) for t in tickers]

    # ---- Download prices ----
    df = download_prices_chunked(tickers_yf, START_DATE, END_DATE, chunk=50)
    df = drop_tickers_without_price_data(df, "adj_close")

    # ---- Sector mapping ----
    sector_map = sector_map_from_constituents(stocks, tickers)
    df["sector"] = df.index.get_level_values("ticker").map(sector_map)

    # ---- Features ----
    df = add_basic_features(df)

    shares_out = fetch_info_map_safe(
        df.index.get_level_values("ticker").unique().tolist(),
        "sharesOutstanding",
    )
    df = add_size_and_beta(df, shares_out)

    # ---- Target ----
    df = add_target(df)

    winsor_cols = [
        "mom_5", "mom_21", "mom_63", "mom_252", "mom_252_21", "sharpe_60",
        "rev_1", "rev_5", "vol_20", "vol_60", "atr_14_pct", "log_dollar_vol_20",
        "amihud_20", "ma_gap_20", "rsi_14", "high_low_20", "beta_60",
    ]

    z_cols = [
        "mom_5", "mom_21", "mom_63", "mom_252_21", "sharpe_60",
        "rev_1", "rev_5", "vol_20", "vol_60", "atr_14_pct",
        "log_dollar_vol_20", "amihud_20", "ma_gap_20", "rsi_14",
        "high_low_20", "log_mkt_cap", "beta_60",
    ]

    # ---- Winsorize + Z-score ----
    df_w = winsorize_by_date(df, winsor_cols, 0.01, 0.99)
    df_z = zscore_by_date(df_w, z_cols)
    df_z = drop_na_per_day(df_z)

    df_z["y_vol_scaled"] = df["y_vol_scaled"]
    df_final = df_z.drop(columns=["log_mkt_cap"])

    # ---- Split ----
    X_train, y_train, X_val, y_val, _, _ = test_train_split(
        df_final,
        label_col="y_vol_scaled",
        start_date=START_DATE,
        end_date=END_DATE,
    )

    # ---- Train LightGBM ----
    lgb_params = {
        "objective": "regression",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 5,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 1,
        "lambda_l1": 1.0,
        "lambda_l2": 5.0,
        "verbosity": -1,
    }

    train_set = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    val_set = lgb.Dataset(X_val, label=y_val, free_raw_data=False)

    model = lgb.train(
        params=lgb_params,
        train_set=train_set,
        num_boost_round=5000,
        valid_sets=[val_set],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)

    # ---- Build val_df (MultiIndex: date,ticker) ----
    val_df_lgb = pd.DataFrame(
        {"y_val": y_val.values, "y_val_pred": y_val_pred},
        index=y_val.index,  # MultiIndex (date,ticker)
    ).sort_index()

    # ---- Build exposures (same index) ----
    idx = val_df_lgb.index
    sector_dummies = pd.get_dummies(df.loc[idx, "sector"], prefix="sec", dtype=float)

    exposures = pd.concat(
        [
            df_z.loc[idx, ["beta_60", "log_dollar_vol_20", "vol_20"]],
            sector_dummies.loc[idx],
        ],
        axis=1,
    )

    # ---- Neutralize predictions ----
    val_df_lgb = neutralize_by_date(
        val_df_lgb,
        pred_col="y_val_pred",
        exposure=exposures,
        out_col="y_val_pred_neut",
        ridge=1e-6,
        winsorize_pct=0.01,
    )

    # ---- Rank by date (based on neutralized predictions) ----
    val_df_lgb["rank"] = (
        val_df_lgb.groupby(level="date")["y_val_pred_neut"]
        .rank(method="first", ascending=False)
    )

    # ---- Save outputs ----
    ic_scores = daily_rank_ic(val_df_lgb, "y_val_pred_neut", "y_val").to_frame("ic_scores")
    ic_scores.to_csv(IC_PATH, index=True)

    val_df_lgb.to_csv(VAL_DF_PATH, index=True)

    print(f"Saved: {IC_PATH}")
    print(f"Saved: {VAL_DF_PATH}")

    # Example usage:
    r = get_stock_rank(val_df_lgb, "2022-01-03", "AAPL", pred_col="y_val_pred_neut", rank_col="rank")
    print("Example rank:", r)

    # Optional plot example (train window):
    # plot_stock_price_train_window(df, ticker="AAPL")


if __name__ == "__main__":
    main()
