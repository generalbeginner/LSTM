"""Apply the repository's multivariate LSTM workflow to TSLA, AMZN, and NVDA.

The workflow follows notebooks/1-3, with a rolling five-year window and
chronological 70/15/15 train/validation/test splits so it remains current.
"""
from pathlib import Path
import json
import os
import random

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "yfinance"
MODEL_DIR = ROOT / "models" / "yfinance_lstm"
REPORT_DIR = ROOT / "reports" / "yfinance_lstm"
for directory in (DATA_DIR, MODEL_DIR, REPORT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

TICKERS = ["TSLA", "AMZN", "NVDA"]
LOOKBACK = 60
FEATURES = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
TARGETS = ["Open", "Close", "High", "Low"]
TARGET = os.environ.get("LSTM_TARGET", "Open")
PIVOT_ORDER = 2
PIVOT_MIN_PROMINENCE = 0.0
SEED = 42

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def download_data(ticker: str) -> pd.DataFrame:
    end = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize() + pd.Timedelta(days=1)
    start = end - pd.DateOffset(years=5)
    frame = yf.download(ticker, start=start.date().isoformat(), end=end.date().isoformat(),
                        interval="1d", auto_adjust=False, progress=False)
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame = frame.reset_index()
    frame["Date"] = pd.to_datetime(frame["Date"]).dt.tz_localize(None)
    frame = frame[["Date"] + FEATURES].dropna().sort_values("Date").reset_index(drop=True)
    if frame.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    return frame


def construct_lstm_data(data: np.ndarray, sequence_size: int, target_idx: int):
    X, y = [], []
    for i in range(sequence_size, len(data)):
        X.append(data[i-sequence_size:i, :])
        y.append(data[i, target_idx])
    return np.asarray(X), np.asarray(y)


def validate_target(target: str) -> str:
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}; received {target!r}")
    return target


def build_model(input_shape):
    model = Sequential([Input(shape=input_shape)])
    for return_sequences in (True, True, True, False):
        model.add(LSTM(units=100, return_sequences=return_sequences))
        model.add(Dropout(rate=0.2))
    model.add(Dense(units=1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def inverse_target(scaler, values, target_idx: int, n_features: int):
    values = np.asarray(values).reshape(-1, 1)
    container = np.ones((len(values), n_features))
    container[:, target_idx] = values[:, 0]
    return scaler.inverse_transform(container)[:, target_idx]


def detect_pivots(dates, prices, pivot_order: int = PIVOT_ORDER,
                  min_prominence: float = PIVOT_MIN_PROMINENCE) -> pd.DataFrame:
    """Return local highs/lows where the price changes direction.

    ``pivot_order`` is the number of observations on each side required to be
    lower/higher than the candidate. ``min_prominence`` is a fractional price
    threshold relative to the candidate, e.g. 0.01 means 1 percent.
    """
    if pivot_order < 1:
        raise ValueError("pivot_order must be at least 1")
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    prices = pd.Series(prices, dtype=float).reset_index(drop=True)
    rows = []
    for i in range(pivot_order, len(prices) - pivot_order):
        window = prices.iloc[i - pivot_order:i + pivot_order + 1]
        value = prices.iloc[i]
        left_right = pd.concat([window.iloc[:pivot_order], window.iloc[pivot_order + 1:]])
        if value == window.max() and value > left_right.max() * (1 + min_prominence):
            rows.append({"Date": dates.iloc[i], "price": value, "pivot_type": "high"})
        elif value == window.min() and value < left_right.min() * (1 - min_prominence):
            rows.append({"Date": dates.iloc[i], "price": value, "pivot_type": "low"})
    return pd.DataFrame(rows, columns=["Date", "price", "pivot_type"])


def compare_pivots(predicted: pd.DataFrame, true: pd.DataFrame,
                   max_date_gap: int = 5) -> pd.DataFrame:
    """Match predicted pivots to nearest unused true pivot of the same type."""
    columns = ["pivot_type", "predicted_date", "true_date", "date_error_days",
               "predicted_price", "true_price", "matched"]
    if predicted.empty and true.empty:
        return pd.DataFrame(columns=columns)
    predicted = predicted.copy()
    true = true.copy()
    predicted["Date"] = pd.to_datetime(predicted["Date"])
    true["Date"] = pd.to_datetime(true["Date"])
    used_true = set()
    rows = []
    for pred_idx, pred in predicted.sort_values("Date").iterrows():
        candidates = true[(true["pivot_type"] == pred["pivot_type"]) & ~true.index.isin(used_true)].copy()
        if candidates.empty:
            match = None
        else:
            candidates["gap"] = (candidates["Date"] - pred["Date"]).abs().dt.days
            match = candidates.sort_values(["gap", "Date"]).iloc[0]
            if match["gap"] > max_date_gap:
                match = None
        if match is None:
            rows.append({"pivot_type": pred["pivot_type"], "predicted_date": pred["Date"],
                         "true_date": pd.NaT, "date_error_days": pd.NA,
                         "predicted_price": pred["price"], "true_price": np.nan, "matched": False})
        else:
            used_true.add(match.name)
            error = abs(int((pred["Date"] - match["Date"]).days))
            rows.append({"pivot_type": pred["pivot_type"], "predicted_date": pred["Date"],
                         "true_date": match["Date"], "date_error_days": error,
                         "predicted_price": pred["price"], "true_price": match["price"], "matched": True})
    for true_idx, actual in true[~true.index.isin(used_true)].iterrows():
        rows.append({"pivot_type": actual["pivot_type"], "predicted_date": pd.NaT,
                     "true_date": actual["Date"], "date_error_days": pd.NA,
                     "predicted_price": np.nan, "true_price": actual["price"], "matched": False})
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["predicted_date", "true_date"], na_position="last").reset_index(drop=True)


def save_forecast_pivots(forecast: pd.DataFrame, ticker: str, target: str,
                         pivot_order: int = PIVOT_ORDER,
                         min_prominence: float = PIVOT_MIN_PROMINENCE) -> Path:
    """Detect and save forecast-only pivot dates with the forecast date window."""
    pivots = detect_pivots(forecast["Date"], forecast[target], pivot_order, min_prominence)
    start = pd.to_datetime(forecast["Date"]).min().date().isoformat()
    end = pd.to_datetime(forecast["Date"]).max().date().isoformat()
    output = REPORT_DIR / f"{ticker.lower()}_{target.lower()}_forecast_pivots_{start}_to_{end}.csv"
    pivots.to_csv(output, index=False)
    return output


def forecast_ticker(ticker: str, target: str = TARGET, horizon: int = 20,
                    pivot_order: int = PIVOT_ORDER,
                    min_prominence: float = PIVOT_MIN_PROMINENCE) -> tuple[pd.DataFrame, Path]:
    """Load a trained ticker/target model, forecast, and save forecast pivot dates."""
    validate_target(target)
    model = tf.keras.models.load_model(MODEL_DIR / f"{ticker.lower()}_{target.lower()}_lstm.keras")
    scaler = joblib.load(MODEL_DIR / f"{ticker.lower()}_{target.lower()}_scaler.gz")
    recent_data = pd.read_csv(DATA_DIR / f"{ticker.lower()}_raw.csv", parse_dates=["Date"])
    forecast = forecast_future(model, recent_data, scaler, target, horizon)
    output = save_forecast_pivots(forecast, ticker, target, pivot_order, min_prominence)
    return forecast, output


def forecast_future(model, recent_data: pd.DataFrame, scaler: MinMaxScaler,
                    target: str, horizon: int, lookback: int = LOOKBACK) -> pd.DataFrame:
    """Recursively forecast future target prices from the latest known OHLCV rows.

    Since future OHLCV features are unknown, the model receives its own previous
    target forecasts in the target column and carries the latest known values
    forward for the other feature columns. This is suitable for a simple
    scenario forecast, not a complete market simulator.
    """
    validate_target(target)
    if len(recent_data) < lookback:
        raise ValueError(f"recent_data must contain at least {lookback} rows")
    target_idx = FEATURES.index(target)
    values = recent_data[FEATURES].tail(lookback).to_numpy(dtype=float)
    scaled = scaler.transform(values)
    forecasts = []
    for _ in range(horizon):
        next_scaled = float(model.predict(scaled[np.newaxis, :, :], verbose=0)[0, 0])
        next_row = scaled[-1].copy()
        next_row[target_idx] = next_scaled
        forecasts.append(inverse_target(scaler, [next_scaled], target_idx, len(FEATURES))[0])
        scaled = np.vstack([scaled[1:], next_row])
    last_date = pd.to_datetime(recent_data["Date"]).iloc[-1]
    dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=horizon)
    return pd.DataFrame({"Date": dates, target: forecasts})


def run_ticker(ticker: str, target: str = TARGET):
    validate_target(target)
    target_idx = FEATURES.index(target)
    data = download_data(ticker)
    data.to_csv(DATA_DIR / f"{ticker.lower()}_raw.csv", index=False)
    n = len(data)
    train_end = int(n * 0.70)
    validate_end = int(n * 0.85)
    train = data.iloc[:train_end].copy()
    validate = data.iloc[train_end:validate_end].copy()
    test = data.iloc[validate_end:].copy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train[FEATURES])
    validate_scaled = scaler.transform(validate[FEATURES])
    test_scaled = scaler.transform(test[FEATURES])
    joblib.dump(scaler, MODEL_DIR / f"{ticker.lower()}_{target.lower()}_scaler.gz")

    all_scaled = np.concatenate([train_scaled, validate_scaled, test_scaled])
    X_train, y_train = construct_lstm_data(train_scaled, LOOKBACK, target_idx)
    X_validate, y_validate = construct_lstm_data(
        all_scaled[train_end - LOOKBACK:validate_end], LOOKBACK, target_idx)
    X_test, y_test = construct_lstm_data(
        all_scaled[-(len(test_scaled) + LOOKBACK):], LOOKBACK, target_idx)

    model = build_model((LOOKBACK, len(FEATURES)))
    model_path = MODEL_DIR / f"{ticker.lower()}_{target.lower()}_lstm.keras"
    callbacks = [
        ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True, mode="min"),
        EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True),
    ]
    history = model.fit(X_train, y_train, validation_data=(X_validate, y_validate),
                        # The reference notebook specifies 200 epochs.  A capped run keeps
                        # this reproducible CPU execution practical; validation checkpointing
                        # and early stopping still select the best epoch.
                        epochs=50, batch_size=64, callbacks=callbacks, verbose=0)
    best_model = tf.keras.models.load_model(model_path)
    predictions = {
        "train": (y_train, best_model.predict(X_train, verbose=0).ravel(), train["Date"].iloc[LOOKBACK:]),
        "validate": (y_validate, best_model.predict(X_validate, verbose=0).ravel(), validate["Date"]),
        "test": (y_test, best_model.predict(X_test, verbose=0).ravel(), test["Date"]),
    }
    metrics = {"ticker": ticker, "rows": n, "start": data["Date"].min().date().isoformat(),
               "end": data["Date"].max().date().isoformat(), "train_rows": len(train),
               "validate_rows": len(validate), "test_rows": len(test),
               "epochs_run": len(history.history["loss"])}
    for split, (actual, predicted, dates) in predictions.items():
        actual_inv = inverse_target(scaler, actual, target_idx, len(FEATURES))
        predicted_inv = inverse_target(scaler, predicted, target_idx, len(FEATURES))
        metrics[f"{split}_rmse"] = float(np.sqrt(mean_squared_error(actual_inv, predicted_inv)))
        metrics[f"{split}_mae"] = float(mean_absolute_error(actual_inv, predicted_inv))

    pivot_rows = []
    for split, (actual, predicted, dates) in predictions.items():
        actual_inv = inverse_target(scaler, actual, target_idx, len(FEATURES))
        predicted_inv = inverse_target(scaler, predicted, target_idx, len(FEATURES))
        predicted_pivots = detect_pivots(dates, predicted_inv)
        true_pivots = detect_pivots(dates, actual_inv)
        compared = compare_pivots(predicted_pivots, true_pivots)
        compared.insert(0, "split", split)
        pivot_rows.append(compared)
    pivot_dates = pd.concat(pivot_rows, ignore_index=True)
    run_start = data["Date"].min().date().isoformat()
    run_end = data["Date"].max().date().isoformat()
    pivot_dates.to_csv(
        REPORT_DIR / f"{ticker.lower()}_{target.lower()}_pivots_{run_start}_to_{run_end}.csv",
        index=False,
    )

    fig, axes = plt.subplots(2, 1, figsize=(15, 10), constrained_layout=True)
    axes[0].plot(data["Date"], data["Open"], label="Actual Open", color="black", linewidth=1)
    axes[0].axvline(validate["Date"].iloc[0], color="orange", linestyle="--", label="Validation start")
    axes[0].axvline(test["Date"].iloc[0], color="green", linestyle="--", label="Test start")
    axes[0].set_title(f"{ticker}: five-year daily Open price and chronological splits")
    axes[0].set_ylabel("Open price (USD)"); axes[0].legend(); axes[0].grid(alpha=.25)
    for split, (actual, predicted, dates) in predictions.items():
        actual_inv = inverse_target(scaler, actual, target_idx, len(FEATURES))
        predicted_inv = inverse_target(scaler, predicted, target_idx, len(FEATURES))
        axes[1].plot(dates, actual_inv, label=f"{split.title()} actual")
        axes[1].plot(dates, predicted_inv, linestyle="--", label=f"{split.title()} predicted")
    axes[1].set_title(f"{ticker}: LSTM next-day Open predictions")
    axes[1].set_ylabel("Open price (USD)"); axes[1].set_xlabel("Date"); axes[1].legend(ncol=3); axes[1].grid(alpha=.25)
    fig.savefig(REPORT_DIR / f"{ticker.lower()}_results.png", dpi=150)
    plt.close(fig)
    return metrics


if __name__ == "__main__":
    results = [run_ticker(ticker, TARGET) for ticker in TICKERS]
    pd.DataFrame(results).to_csv(REPORT_DIR / "metrics.csv", index=False)
    (REPORT_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
    print(pd.DataFrame(results).to_string(index=False))
