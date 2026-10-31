import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBRegressor

st.set_page_config(page_title="PJMW Demand Forecast", page_icon="⚡", layout="wide")

TARGET = "PJMW_MW"
FEATURE_INPUTS = ["lag_1", "lag_24", "lag_168", "rolling_24", "rolling_168"]
TIME_COLS = ["hour", "day_of_week", "month", "year", "is_weekend",
             "hour_sin", "hour_cos", "month_sin", "month_cos"]
HISTORY_NEEDED = 168  # previous hours required for lag_168 / rolling_168


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model(folder: str):
    folder = Path(folder)
    model = joblib.load(folder / "pjmw_final_model.pkl")
    columns = joblib.load(folder / "pjmw_feature_columns.pkl")
    return model, columns


@st.cache_data
def load_table(path: str):
    p = Path(path)
    return pd.read_csv(p) if p.suffix == ".csv" else pd.read_excel(p)


def time_features(ts_index) -> pd.DataFrame:
    """Build the time-based features from a DatetimeIndex."""
    ts = pd.DatetimeIndex(ts_index)
    return pd.DataFrame({
        "hour": ts.hour, "day_of_week": ts.dayofweek, "month": ts.month, "year": ts.year,
        "is_weekend": (ts.dayofweek >= 5).astype(int),
        "hour_sin": np.sin(2 * np.pi * ts.hour / 24), "hour_cos": np.cos(2 * np.pi * ts.hour / 24),
        "month_sin": np.sin(2 * np.pi * ts.month / 12), "month_cos": np.cos(2 * np.pi * ts.month / 12),
    })


def build_features(timestamp, lag_1, lag_24, lag_168, rolling_24, rolling_168, columns):
    row = time_features(pd.DatetimeIndex([pd.Timestamp(timestamp)])).iloc[0].to_dict()
    row.update({"lag_1": lag_1, "lag_24": lag_24, "lag_168": lag_168,
                "rolling_24": rolling_24, "rolling_168": rolling_168})
    return pd.DataFrame([row])[columns]


def build_features_df(df, columns):
    """Batch version: df must contain 'timestamp' + the 5 numeric feature columns."""
    out = time_features(pd.to_datetime(df["timestamp"]))
    for c in FEATURE_INPUTS:
        out[c] = df[c].astype(float).values
    return out[columns]


def recursive_forecast(model, columns, history, future_time_feats: pd.DataFrame):
    """
    Future forecast (recursive): predict one hour, append that prediction to the
    history, then recompute lag_1 / lag_24 / lag_168 / rolling_* for the next hour.
    history = last known demand values (oldest -> newest, at least 168 values).
    """
    h = [float(v) for v in history][-HISTORY_NEEDED:]
    if len(h) < HISTORY_NEEDED:
        raise ValueError(f"At least {HISTORY_NEEDED} hours of history are required (got {len(h)})")
    preds = []
    for k in range(len(future_time_feats)):
        arr = np.asarray(h)
        row = future_time_feats.iloc[k].to_dict()
        row.update({"lag_1": arr[-1], "lag_24": arr[-24], "lag_168": arr[-168],
                    "rolling_24": arr[-24:].mean(), "rolling_168": arr[-168:].mean()})
        p = float(model.predict(pd.DataFrame([row])[columns])[0])
        preds.append(p)
        h = (h + [p])[-HISTORY_NEEDED:]
    return np.asarray(preds)


def read_history_file(up):
    """Convert an uploaded history file (timestamp + demand) into a Series."""
    df = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
    tcol = next((c for c in df.columns if str(c).lower() in ("timestamp", "datetime", "date", "time")), None)
    if tcol is None:
        raise ValueError("Timestamp column not found (expected name: timestamp / Datetime / date).")
    vcol = TARGET if TARGET in df.columns else next(
        (c for c in df.columns if c != tcol and pd.api.types.is_numeric_dtype(df[c])), None)
    if vcol is None:
        raise ValueError("Demand (MW) column not found.")
    s = pd.Series(df[vcol].astype(float).values, index=pd.to_datetime(df[tcol]), name=TARGET)
    return s[~s.index.duplicated()].sort_index()


@st.cache_resource(show_spinner=False)
def backtest_model(folder: str, data_path: str):
    """Honest backtest: train a temporary model on the first 80% of the data."""
    df = load_table(data_path)
    X, y = df.drop(columns=[TARGET]), df[TARGET].values
    split = int(len(df) * 0.8)
    params = json.load(open(Path(folder) / "best_params.json"))
    m = XGBRegressor(**params, n_jobs=-1, random_state=42).fit(X.iloc[:split], y[:split])
    return m, list(X.columns), X, y, split


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.title("⚡ PJMW Forecast")
folder = st.sidebar.text_input("Results folder", value="results", key="folder",
                               help="Output folder created by running: pjmw_development.py train")

try:
    model, columns = load_model(folder)
    st.sidebar.success("Model loaded ✅")
except Exception as e:
    st.title("⚡ PJMW Electricity Demand Forecast")
    st.error(f"Could not load the model: {e}")
    st.info("Run this first:\n\n"
            "`python pjmw_development.py train --data PJMW_features.xlsx --out results`")
    st.stop()

st.title("⚡ PJMW Electricity Demand Forecast")
st.caption("Hourly demand (MW) prediction and future forecast using a tuned XGBoost model")

tab1, tab2, tab3, tab4 = st.tabs(["🔮 Single prediction", "📅 Future forecast",
                                  "📂 Batch prediction", "📊 Model results"])

# ----------------------------------------------------------------------------
# Tab 1: single prediction
# ----------------------------------------------------------------------------
with tab1:
    st.subheader("Predict demand for a single hour")
    c1, c2 = st.columns(2)
    with c1:
        d = st.date_input("Date", value=pd.Timestamp("2018-08-02"))
        h = st.slider("Hour of day", 0, 23, 14)
    with c2:
        lag_1 = st.number_input("lag_1 (previous hour demand, MW)", value=6500.0, step=10.0)
        lag_24 = st.number_input("lag_24 (same hour yesterday, MW)", value=6400.0, step=10.0)
        lag_168 = st.number_input("lag_168 (same hour last week, MW)", value=6300.0, step=10.0)
    c3, c4 = st.columns(2)
    rolling_24 = c3.number_input("rolling_24 (avg of previous 24 hrs, MW)", value=6000.0, step=10.0)
    rolling_168 = c4.number_input("rolling_168 (avg of previous 168 hrs, MW)", value=5800.0, step=10.0)

    if st.button("Predict", type="primary", key="single_btn"):
        ts = pd.Timestamp(d) + pd.Timedelta(hours=h)
        x_new = build_features(ts, lag_1, lag_24, lag_168, rolling_24, rolling_168, columns)
        pred = float(model.predict(x_new)[0])
        m1, m2 = st.columns(2)
        m1.metric(f"Predicted demand @ {ts:%Y-%m-%d %H:%M}", f"{pred:,.0f} MW")
        m2.metric("vs yesterday same hour", f"{pred - lag_24:+,.0f} MW")
        with st.expander("Features passed to the model"):
            st.dataframe(x_new.T.rename(columns={0: "value"}))

# ----------------------------------------------------------------------------
# Tab 2: FUTURE forecast
# ----------------------------------------------------------------------------
with tab2:
    st.subheader("Future demand forecast (next hours / days)")
    st.write("Starting from the last known demand history, the model forecasts the next N hours, "
             "**feeding each predicted hour back in as input for the next hour**.")

    horizon = st.slider("How many hours to forecast?", 1, 168, 24, key="horizon",
                        help="24 = 1 day, 168 = 1 week")
    SRC_TRAIN = "From the training data file (PJMW_features.xlsx)"
    SRC_UPLOAD = "Upload a history file (timestamp + PJMW_MW)"
    source = st.radio("History source", [SRC_TRAIN, SRC_UPLOAD], key="source")

    history, last_ts, ready = None, None, False

    if source == SRC_TRAIN:
        data_path = st.text_input("Data file path", value="PJMW_features.xlsx", key="fc_data_path")
        try:
            dfd = load_table(data_path)
            hist_vals = dfd[TARGET].values[-HISTORY_NEEDED:]
            last_row = dfd.iloc[-1]
            st.caption(f"Last row in data: year={int(last_row['year'])}, month={int(last_row['month'])}, "
                       f"hour={int(last_row['hour'])}, day_of_week={int(last_row['day_of_week'])} "
                       f"(0=Mon ... 6=Sun), demand={last_row[TARGET]:,.0f} MW")
            # The data has no exact date -> list candidate dates matching year/month/day-of-week/hour
            days = pd.date_range(f"{int(last_row['year'])}-{int(last_row['month']):02d}-01", periods=31, freq="D")
            days = days[(days.month == int(last_row["month"])) & (days.dayofweek == int(last_row["day_of_week"]))]
            cands = [f"{d:%Y-%m-%d} {int(last_row['hour']):02d}:00" for d in days]
            st.info("The data file has no exact date/time. Enter the **date & time** of the last row.\n\n"
                    "Dates matching its year / month / day-of-week / hour: **" + ", ".join(cands) + "**")
            ts_text = st.text_input("Last known hour (YYYY-MM-DD HH:MM)",
                                    value=cands[0] if cands else "2018-08-03 00:00", key="last_ts_text")
            try:
                last_ts = pd.Timestamp(ts_text)
                ok = (last_ts.year == int(last_row["year"]) and last_ts.month == int(last_row["month"])
                      and last_ts.hour == int(last_row["hour"]) and last_ts.dayofweek == int(last_row["day_of_week"]))
                if ok:
                    st.success("Date/time matches the data ✅")
                else:
                    st.warning("This date does not match the last row of the data - please check it.")
                history, ready = hist_vals, True
            except Exception:
                st.error("Invalid date format (use YYYY-MM-DD HH:MM).")
        except Exception as e:
            st.error(f"Could not load the data file: {e}")
    else:
        up = st.file_uploader(f"History file (at least {HISTORY_NEEDED} hourly rows)", type=["csv", "xlsx"], key="hist_up")
        if up is not None:
            try:
                s = read_history_file(up)
                if len(s) < HISTORY_NEEDED:
                    st.error(f"At least {HISTORY_NEEDED} rows are required (got {len(s)}).")
                else:
                    gaps = s.index[-HISTORY_NEEDED:].to_series().diff().dropna().ne(pd.Timedelta(hours=1)).sum()
                    if gaps:
                        st.warning(f"Found {gaps} gaps in the last {HISTORY_NEEDED} hours (not consecutive hourly data) - "
                                   "forecast accuracy may drop.")
                    history, last_ts, ready = s.values[-HISTORY_NEEDED:], s.index[-1], True
                    st.success(f"History loaded. Last known hour: {last_ts:%Y-%m-%d %H:%M}")
            except Exception as e:
                st.error(str(e))

    if st.button("Forecast future", type="primary", disabled=not ready, key="fc_btn"):
        future_idx = pd.date_range(last_ts + pd.Timedelta(hours=1), periods=horizon, freq="h")
        preds = recursive_forecast(model, columns, history, time_features(future_idx))
        hist_idx = pd.date_range(end=last_ts, periods=HISTORY_NEEDED, freq="h")
        st.session_state["fc"] = {
            "forecast": pd.Series(preds, index=future_idx, name="Forecast (MW)"),
            "history": pd.Series(np.asarray(history, dtype=float), index=hist_idx, name="History (MW)"),
        }

    if "fc" in st.session_state:
        fc, hs = st.session_state["fc"]["forecast"], st.session_state["fc"]["history"]
        st.markdown(f"**Forecast: {fc.index[0]:%Y-%m-%d %H:%M}  →  {fc.index[-1]:%Y-%m-%d %H:%M}**")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Peak demand", f"{fc.max():,.0f} MW", f"{fc.idxmax():%a %H:%M}", delta_color="off")
        k2.metric("Lowest demand", f"{fc.min():,.0f} MW", f"{fc.idxmin():%a %H:%M}", delta_color="off")
        k3.metric("Average", f"{fc.mean():,.0f} MW")
        k4.metric("Next hour", f"{fc.iloc[0]:,.0f} MW")

        chart = pd.concat([hs.rename("History"), fc.rename("Forecast")], axis=1)
        st.line_chart(chart)
        out = fc.rename("predicted_MW").rename_axis("timestamp").reset_index()
        st.dataframe(out, width="stretch", height=250)
        st.download_button("Forecast download (CSV)", out.to_csv(index=False), "future_forecast.csv", "text/csv")
        st.caption("⚠️ Recursive forecast: errors accumulate as the horizon grows. "
                   "Use the 'Accuracy check' below to see the error by forecast horizon.")

    with st.expander("📏 Accuracy check (backtest) - how does error grow with the horizon?"):
        st.write("A temporary model is trained on the **first 80%** of the data, then forecasts are made from many points in the "
                 "**last 20%** and compared with the actual values (an honest test, because the final model is trained on the full data).")
        bt_path = st.text_input("Data file path", value="PJMW_features.xlsx", key="bt_path")
        b1, b2 = st.columns(2)
        bt_h = b1.slider("Backtest horizon (hours)", 6, 168, 24, key="bt_h")
        bt_n = b2.slider("Start points count", 5, 100, 30, key="bt_n")
        if st.button("Run backtest", key="bt_btn"):
            try:
                with st.spinner("Training a temporary model..."):
                    m, cols, Xb, yb, split = backtest_model(folder, bt_path)
                starts = np.linspace(split, len(yb) - bt_h - 1, bt_n).astype(int)
                errs, pct = [], []
                prog = st.progress(0.0)
                for j, i in enumerate(starts):
                    p = recursive_forecast(m, cols, yb[:i], Xb[TIME_COLS].iloc[i:i + bt_h])
                    a = yb[i:i + bt_h]
                    errs.append(np.abs(a - p)); pct.append(np.abs(a - p) / a * 100)
                    prog.progress((j + 1) / len(starts))
                errs, pct = np.vstack(errs), np.vstack(pct)
                r1, r2, r3 = st.columns(3)
                r1.metric("Overall MAE", f"{errs.mean():,.0f} MW")
                r2.metric("Overall MAPE", f"{pct.mean():.2f}%")
                r3.metric("1st hour MAE → last hour MAE", f"{errs[:, 0].mean():,.0f} → {errs[:, -1].mean():,.0f} MW")
                st.line_chart(pd.DataFrame({"MAE (MW)": errs.mean(axis=0)},
                                           index=pd.RangeIndex(1, bt_h + 1, name="Hours ahead")))
            except Exception as e:
                st.error(f"Backtest failed: {e}")

# ----------------------------------------------------------------------------
# Tab 3: batch prediction
# ----------------------------------------------------------------------------
with tab3:
    st.subheader("Upload a CSV / Excel file to predict many hours at once")
    st.write("The file must contain these columns: `timestamp, lag_1, lag_24, lag_168, rolling_24, rolling_168`")

    sample = pd.DataFrame({
        "timestamp": ["2018-08-02 14:00", "2018-08-02 15:00"],
        "lag_1": [6500, 6600], "lag_24": [6400, 6450], "lag_168": [6300, 6350],
        "rolling_24": [6000, 6010], "rolling_168": [5800, 5805],
    })
    st.download_button("Sample CSV download", sample.to_csv(index=False), "sample_input.csv", "text/csv")

    up = st.file_uploader("Upload file", type=["csv", "xlsx"], key="batch_up")
    if up is not None:
        df_in = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
        missing = [c for c in ["timestamp"] + FEATURE_INPUTS if c not in df_in.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            preds = model.predict(build_features_df(df_in, columns))
            res = df_in.copy()
            res["predicted_MW"] = preds
            st.dataframe(res, width="stretch")
            res_plot = res.assign(timestamp=pd.to_datetime(res["timestamp"])).set_index("timestamp")
            st.line_chart(res_plot["predicted_MW"])
            st.download_button("Predictions download (CSV)", res.to_csv(index=False),
                               "predictions.csv", "text/csv")

# ----------------------------------------------------------------------------
# Tab 4: model results (training outputs)
# ----------------------------------------------------------------------------
with tab4:
    st.subheader("Training results")
    f = Path(folder)
    comp = f / "model_comparison.csv"
    if comp.exists():
        table = pd.read_csv(comp)
        st.write(f"🏆 Best model (lowest RMSE): **{table.loc[0, 'model']}**")
        st.dataframe(table.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}",
                                         "MAPE": "{:.2f}", "R2": "{:.4f}"}),
                     width="stretch")
        st.bar_chart(table.set_index("model")["RMSE"])
    else:
        st.info("model_comparison.csv not found - run the training script first.")

    images = [("02_actual_vs_predicted.png", "Actual vs Predicted"),
              ("03_arima_vs_sarima.png", "ARIMA vs SARIMA"),
              ("04_residuals.png", "Residual analysis"),
              ("05_error_by_time.png", "Error by hour / month / day"),
              ("06_feature_importance.png", "Feature importance")]
    for fname, title in images:
        if (f / fname).exists():
            st.markdown(f"**{title}**")
            st.image(str(f / fname))