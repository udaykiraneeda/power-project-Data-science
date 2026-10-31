"""
PJMW hourly electricity demand forecasting - full development pipeline.

Run everything (EDA checks -> split -> all models -> comparison -> plots -> save model):
    python pjmw_development.py train --data PJMW_features.xlsx --out results

Quick test run (less tuning):
    python pjmw_development.py train --data PJMW_features.xlsx --out results --fast

Predict demand for one new hour using the saved model:
    python pjmw_development.py predict --out results --timestamp "2018-08-02 14:00" \
        --lag_1 6500 --lag_24 6400 --lag_168 6300 --rolling_24 6000 --rolling_168 5800

Models: Naive, Linear Regression, Random Forest, XGBoost, LightGBM,
        XGBoost (tuned), Ensemble (XGB+LGB), ARIMA, SARIMA

Requirements: pip install pandas numpy matplotlib scikit-learn xgboost lightgbm statsmodels openpyxl joblib
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # plots files ga save avutayi (window open avvadu)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, cross_val_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

TARGET = "PJMW_MW"
RANDOM_STATE = 42


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def banner(text):
    print("\n" + "=" * 70 + f"\n{text}\n" + "=" * 70)


def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((np.asarray(y_true) - np.asarray(y_pred)) / np.asarray(y_true))) * 100
    print(f"{name:20s} MAE={mae:8.2f}  RMSE={rmse:8.2f}  MAPE={mape:5.2f}%  R2={r2:.4f}")
    return {"model": name, "MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


def one_step_ahead(fitted_res, y_new, chunk=1000):
    """ARIMA/SARIMA: params fix chesi test data meeda one-step-ahead predictions."""
    res, out = fitted_res, []
    for i in range(0, len(y_new), chunk):
        res = res.extend(y_new[i:i + chunk])
        out.append(np.asarray(res.fittedvalues))
    return np.concatenate(out)


# ----------------------------------------------------------------------------
# Steps
# ----------------------------------------------------------------------------
def load_and_check(path):
    banner("STEP 1-2: Load data & basic checks")
    df = pd.read_excel(path)
    print("Shape     :", df.shape)
    print("Nulls     :", int(df.isnull().sum().sum()))
    print("Duplicates:", int(df.duplicated().sum()))
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' data lo ledu. Columns: {list(df.columns)}")
    return df


def leakage_check(df):
    banner("STEP 3: Leakage check (rolling_24)")
    y_all = df[TARGET]
    a = df["rolling_24"].iloc[168:]
    ok_shift = np.allclose(a, y_all.shift(1).rolling(24).mean().iloc[168:], rtol=1e-3)
    ok_noshift = np.allclose(a, y_all.rolling(24).mean().iloc[168:], rtol=1e-3)
    print("Matches shift(1) version :", ok_shift)
    print("Matches no-shift version :", ok_noshift)
    if ok_noshift and not ok_shift:
        print("WARNING: rolling_24 lo current value kuda undi -> DATA LEAKAGE!")


def split_data(df):
    banner("STEP 4: Time-based split (train / validation / test)")
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    vs = int(len(X_train) * 0.9)  # train lo last 10% = validation (early stopping kosam)
    X_tr, X_val = X_train.iloc[:vs], X_train.iloc[vs:]
    y_tr, y_val = y_train.iloc[:vs], y_train.iloc[vs:]
    print("Train years:", X_train["year"].min(), "-", X_train["year"].max())
    print("Test years :", X_test["year"].min(), "-", X_test["year"].max())
    print(f"Sizes -> train(sub): {len(X_tr)}  val: {len(X_val)}  test: {len(X_test)}")
    return X, y, X_train, X_test, y_train, y_test, X_tr, X_val, y_tr, y_val


def train_models(X, y, X_train, X_test, y_train, y_test, X_tr, X_val, y_tr, y_val, args):
    results, preds = [], {}

    def log(name, p):
        p = np.asarray(p, dtype=float)
        preds[name] = p
        results.append(evaluate(name, y_test, p))

    banner("STEP 5-6: Baseline & evaluation")
    log("Naive (lag_24)", X_test["lag_24"])

    banner("STEP 7: Machine-learning models")
    lr = LinearRegression().fit(X_train, y_train)
    log("Linear Regression", lr.predict(X_test))

    rf = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5,
                               n_jobs=-1, random_state=RANDOM_STATE).fit(X_train, y_train)
    log("Random Forest", rf.predict(X_test))

    xgb = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, subsample=0.8,
                       colsample_bytree=0.8, early_stopping_rounds=50, n_jobs=-1,
                       random_state=RANDOM_STATE)
    xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    print("XGBoost best iteration:", xgb.best_iteration)
    log("XGBoost", xgb.predict(X_test))

    lgbm = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.03, num_leaves=63, subsample=0.8,
                             colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    lgbm.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
             callbacks=[lgb.early_stopping(50, verbose=False)])
    log("LightGBM", lgbm.predict(X_test))

    banner("STEP 8: TimeSeriesSplit cross-validation (train data only)")
    cv_model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8,
                            colsample_bytree=0.8, n_jobs=-1, random_state=RANDOM_STATE)
    scores = -cross_val_score(cv_model, X_train, y_train, cv=TimeSeriesSplit(n_splits=5),
                              scoring="neg_root_mean_squared_error")
    print("RMSE per fold:", np.round(scores, 2))
    print(f"Mean RMSE: {scores.mean():.2f}   Std: {scores.std():.2f}")

    banner("STEP 9: Hyperparameter tuning (XGBoost)")
    param_dist = {
        "n_estimators": [200, 400, 600, 800],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "max_depth": [4, 6, 8, 10],
        "min_child_weight": [1, 3, 5, 10],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_lambda": [0.5, 1, 5, 10],
    }
    n_iter = 5 if args.fast else 20
    search = RandomizedSearchCV(XGBRegressor(n_jobs=-1, random_state=RANDOM_STATE),
                                param_distributions=param_dist, n_iter=n_iter,
                                cv=TimeSeriesSplit(n_splits=3),
                                scoring="neg_root_mean_squared_error",
                                random_state=RANDOM_STATE, n_jobs=1)
    search.fit(X_train, y_train)
    print("Best params:", search.best_params_)
    best_xgb = search.best_estimator_
    log("XGBoost (tuned)", best_xgb.predict(X_test))

    banner("STEP 10: Ensemble (XGB tuned + LightGBM)")
    log("Ensemble (XGB+LGB)", (preds["XGBoost (tuned)"] + preds["LightGBM"]) / 2)

    banner("STEP 11: ARIMA & SARIMA (univariate, one-step-ahead)")
    window = 24 * args.arima_days
    y_tail = y_train.values[-window:].astype(float)
    y_te = y_test.values.astype(float)

    t0 = time.time()
    arima_fit = ARIMA(y_tail, order=(2, 1, 2)).fit()
    log("ARIMA(2,1,2)", one_step_ahead(arima_fit, y_te))
    print(f"  ARIMA time: {time.time() - t0:.1f}s")

    t0 = time.time()
    sarima_fit = SARIMAX(y_tail, order=(1, 0, 1), seasonal_order=(1, 1, 1, 24),
                         enforce_stationarity=False, enforce_invertibility=False
                         ).fit(disp=False, maxiter=50)
    log("SARIMA(24)", one_step_ahead(sarima_fit, y_te))
    print(f"  SARIMA time: {time.time() - t0:.1f}s")

    return results, preds, search, best_xgb


def make_plots(out, results, preds, y_test, X_test, X, best_xgb):
    banner("STEP 12-13: Comparison table, plots & error analysis")
    table = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    table.to_csv(out / "model_comparison.csv", index=False)
    print(table.to_string(index=False))

    best_name = table.loc[0, "model"]
    best_pred = preds[best_name]
    print("\nBest model (lowest RMSE):", best_name)

    # model comparison
    plt.figure(figsize=(10, 4))
    plt.barh(table["model"][::-1], table["RMSE"][::-1])
    plt.xlabel("RMSE (MW) - takkuva unte better")
    plt.title("Model comparison")
    plt.tight_layout(); plt.savefig(out / "01_model_comparison.png", dpi=120); plt.close()

    # actual vs predicted
    n = 24 * 14
    plt.figure(figsize=(14, 5))
    plt.plot(y_test.values[:n], label="Actual")
    plt.plot(best_pred[:n], label=f"Predicted ({best_name})")
    plt.title("Actual vs Predicted (first 14 days of test)")
    plt.xlabel("Hours"); plt.ylabel("MW"); plt.legend()
    plt.tight_layout(); plt.savefig(out / "02_actual_vs_predicted.png", dpi=120); plt.close()

    # ARIMA vs SARIMA
    n = 24 * 7
    plt.figure(figsize=(14, 5))
    plt.plot(y_test.values[:n], label="Actual", linewidth=2)
    plt.plot(preds["ARIMA(2,1,2)"][:n], label="ARIMA(2,1,2)", alpha=0.8)
    plt.plot(preds["SARIMA(24)"][:n], label="SARIMA(24)", alpha=0.8)
    plt.title("ARIMA vs SARIMA (first 7 days of test, one-step-ahead)")
    plt.xlabel("Hours"); plt.ylabel("MW"); plt.legend()
    plt.tight_layout(); plt.savefig(out / "03_arima_vs_sarima.png", dpi=120); plt.close()

    # residuals
    res = y_test.values - best_pred
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1); plt.hist(res, bins=50); plt.title("Error distribution")
    plt.subplot(1, 2, 2); plt.scatter(best_pred, res, s=2, alpha=0.3)
    plt.axhline(0, color="red"); plt.title("Predicted vs Error")
    plt.tight_layout(); plt.savefig(out / "04_residuals.png", dpi=120); plt.close()

    # error by hour / month / day of week
    err = pd.DataFrame({"hour": X_test["hour"].values, "month": X_test["month"].values,
                        "dow": X_test["day_of_week"].values, "abs_err": np.abs(res)})
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    err.groupby("hour")["abs_err"].mean().plot(ax=ax[0], title="MAE by hour")
    err.groupby("month")["abs_err"].mean().plot(kind="bar", ax=ax[1], title="MAE by month")
    err.groupby("dow")["abs_err"].mean().plot(kind="bar", ax=ax[2], title="MAE by day of week")
    plt.tight_layout(); plt.savefig(out / "05_error_by_time.png", dpi=120); plt.close()

    # feature importance
    imp = pd.Series(best_xgb.feature_importances_, index=X.columns).sort_values()
    imp.plot(kind="barh", figsize=(8, 6), title="Feature Importance (XGBoost tuned)")
    plt.tight_layout(); plt.savefig(out / "06_feature_importance.png", dpi=120); plt.close()

    pd.DataFrame(preds).assign(actual=y_test.values).to_csv(out / "test_predictions.csv", index=False)
    return best_name


def save_final_model(out, X, y, search):
    banner("STEP 14: Final model (tuned XGBoost, full data) -> save")
    final_model = XGBRegressor(**search.best_params_, n_jobs=-1, random_state=RANDOM_STATE)
    final_model.fit(X, y)
    joblib.dump(final_model, out / "pjmw_final_model.pkl")
    joblib.dump(list(X.columns), out / "pjmw_feature_columns.pkl")
    with open(out / "best_params.json", "w") as f:
        json.dump({k: (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()}, f, indent=2)
    print("Saved model, feature columns and best_params.json in:", out)


# ----------------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------------
def build_features(timestamp, lag_1, lag_24, lag_168, rolling_24, rolling_168, columns):
    ts = pd.Timestamp(timestamp)
    row = {
        "hour": ts.hour, "day_of_week": ts.dayofweek, "month": ts.month, "year": ts.year,
        "is_weekend": int(ts.dayofweek >= 5),
        "hour_sin": np.sin(2 * np.pi * ts.hour / 24), "hour_cos": np.cos(2 * np.pi * ts.hour / 24),
        "month_sin": np.sin(2 * np.pi * ts.month / 12), "month_cos": np.cos(2 * np.pi * ts.month / 12),
        "lag_1": lag_1, "lag_24": lag_24, "lag_168": lag_168,
        "rolling_24": rolling_24, "rolling_168": rolling_168,
    }
    return pd.DataFrame([row])[columns]


def cmd_train(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df = load_and_check(args.data)
    leakage_check(df)
    X, y, X_train, X_test, y_train, y_test, X_tr, X_val, y_tr, y_val = split_data(df)
    results, preds, search, best_xgb = train_models(
        X, y, X_train, X_test, y_train, y_test, X_tr, X_val, y_tr, y_val, args)
    make_plots(out, results, preds, y_test, X_test, X, best_xgb)
    save_final_model(out, X, y, search)
    banner("DONE - results folder: " + str(out.resolve()))


def cmd_predict(args):
    out = Path(args.out)
    model = joblib.load(out / "pjmw_final_model.pkl")
    cols = joblib.load(out / "pjmw_feature_columns.pkl")
    x_new = build_features(args.timestamp, args.lag_1, args.lag_24, args.lag_168,
                           args.rolling_24, args.rolling_168, cols)
    print(f"Predicted demand (MW) for {args.timestamp}: {model.predict(x_new)[0]:.2f}")


def main():
    p = argparse.ArgumentParser(description="PJMW demand forecasting pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="Run full pipeline")
    t.add_argument("--data", default="PJMW_features.xlsx")
    t.add_argument("--out", default="results")
    t.add_argument("--fast", action="store_true", help="Less tuning iterations (quick test)")
    t.add_argument("--arima_days", type=int, default=90, help="Days of train data for ARIMA/SARIMA fit")
    t.set_defaults(func=cmd_train)

    q = sub.add_parser("predict", help="Predict one hour using saved model")
    q.add_argument("--out", default="results")
    q.add_argument("--timestamp", required=True)
    for k in ["lag_1", "lag_24", "lag_168", "rolling_24", "rolling_168"]:
        q.add_argument(f"--{k}", type=float, required=True)
    q.set_defaults(func=cmd_predict)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
