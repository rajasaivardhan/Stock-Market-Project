import os
import argparse
import json
from typing import List, Tuple
import numpy as np
import pandas as pd
import yfinance as yf
from pyts.image import GramianAngularField

# -----------------------------
# Utility functions
# -----------------------------

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df['Close']

    # Simple Moving Averages
    df['SMA_10'] = close.rolling(window=10, min_periods=10).mean()
    df['SMA_20'] = close.rolling(window=20, min_periods=20).mean()
    df['SMA_50'] = close.rolling(window=50, min_periods=50).mean()

    # Exponential Moving Averages
    df['EMA_10'] = close.ewm(span=10, adjust=False).mean()
    df['EMA_20'] = close.ewm(span=20, adjust=False).mean()

    # Bollinger Bands (20)
    rolling_mean = close.rolling(window=20, min_periods=20).mean()
    rolling_std = close.rolling(window=20, min_periods=20).std(ddof=0)
    df['BB_Middle'] = rolling_mean
    df['BB_Upper'] = rolling_mean + (rolling_std * 2)
    df['BB_Lower'] = rolling_mean - (rolling_std * 2)

    # RSI (14)
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.ewm(span=14, adjust=False).mean()
    roll_down = down.ewm(span=14, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-8)
    df['RSI_14'] = 100.0 - (100.0 / (1.0 + rs))

    # MACD (12, 26) and signal (9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df['MACD'] = macd
    df['MACD_Signal'] = signal
    df['MACD_Hist'] = macd - signal

    # Daily returns and realized volatility (rolling)
    df['Return'] = close.pct_change()
    df['RV_5'] = df['Return'].rolling(window=5).std(ddof=0)
    df['RV_10'] = df['Return'].rolling(window=10).std(ddof=0)

    return df.dropna().copy()


def minmax_to_unit_interval(x: np.ndarray) -> np.ndarray:
    x_min = np.min(x)
    x_max = np.max(x)
    if x_max - x_min < 1e-12:
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min)


def scale_to_neg1_pos1(x: np.ndarray) -> np.ndarray:
    return (minmax_to_unit_interval(x) * 2.0) - 1.0


def make_windows(
    df: pd.DataFrame,
    window_size: int,
    horizon: int,
    step_size: int,
) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    closes = df['Close'].values
    indices = df.index
    X_idx: List[pd.Timestamp] = []
    y_list: List[int] = []

    # label: future close after horizon vs last close in the window
    last_idx = len(df) - window_size - horizon
    for start in range(0, last_idx + 1, step_size):
        end = start + window_size
        future_idx = end + horizon - 1
        last_close = closes[end - 1]
        future_close = closes[future_idx]
        label = 1 if future_close > last_close else 0
        y_list.append(label)
        X_idx.append(indices[end - 1])

    return np.array(X_idx), np.array(y_list, dtype=np.int64), list(indices)


def compute_window_gasf_channels(
    window_df: pd.DataFrame,
    channel_features: List[str],
    image_size: int,
    gaf: GramianAngularField,
) -> np.ndarray:
    """Return an image of shape (image_size, image_size, C)."""
    channels = []
    for feat in channel_features:
        series = window_df[feat].values.astype(np.float32)
        series_scaled = scale_to_neg1_pos1(series)
        series_scaled = np.nan_to_num(series_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        img = gaf.fit_transform(series_scaled.reshape(1, -1))  # (1, H, W)
        channels.append(img[0])
    stacked = np.stack(channels, axis=-1)  # (H, W, C)
    return stacked


def compute_numeric_features(window_df: pd.DataFrame) -> np.ndarray:
    """Small numeric summary features per window to complement images."""
    # Last known indicators
    last = window_df.iloc[-1]

    # Simple slopes using last 5 points
    def slope_of(series: pd.Series) -> float:
        y = series.values[-5:]
        x = np.arange(len(y))
        x = x - x.mean()
        denom = np.sum(x * x) + 1e-8
        return float(np.sum((y - y.mean()) * x) / denom)

    # Volatility and returns stats over the window
    returns = window_df['Return'].dropna()
    vol = float(returns.std(ddof=0)) if len(returns) > 0 else 0.0
    mean_ret = float(returns.mean()) if len(returns) > 0 else 0.0
    skew = float(((returns - returns.mean())**3).mean() / ((returns.std(ddof=0) + 1e-8)**3)) if len(returns) > 0 else 0.0

    features = np.array([
        float(last['RSI_14']),
        float(last['MACD']),
        float(last['MACD_Signal']),
        float(last['MACD_Hist']),
        float(last['SMA_10']),
        float(last['SMA_20']),
        float(last['SMA_50']),
        float(last['EMA_10']),
        float(last['EMA_20']),
        float(last['BB_Upper'] - last['BB_Lower']),  # band width
        vol,
        mean_ret,
        skew,
        slope_of(window_df['Close']),
        slope_of(window_df['SMA_10']),
        slope_of(window_df['EMA_10']),
    ], dtype=np.float32)

    # Robust scale: z-score with tanh squashing
    mu = np.mean(features)
    sigma = np.std(features) + 1e-6
    z = (features - mu) / sigma
    return np.tanh(z)


def build_dataset(
    df: pd.DataFrame,
    window_size: int,
    horizon: int,
    step_size: int,
    image_size: int,
    channel_features: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gaf = GramianAngularField(image_size=image_size, method='summation')

    # Prepare sliding indices and labels
    _, y, _ = make_windows(df, window_size=window_size, horizon=horizon, step_size=step_size)

    X_images: List[np.ndarray] = []
    X_feats: List[np.ndarray] = []

    # Iterate windows
    last_idx = len(df) - window_size - horizon
    for start in range(0, last_idx + 1, step_size):
        end = start + window_size
        window_df = df.iloc[start:end]
        img = compute_window_gasf_channels(window_df, channel_features, image_size, gaf)
        X_images.append(img.astype(np.float32))
        feats = compute_numeric_features(window_df)
        X_feats.append(feats)

    X_images_arr = np.stack(X_images, axis=0)
    X_feats_arr = np.stack(X_feats, axis=0)
    return X_images_arr, X_feats_arr, y.astype(np.int64)


def save_dataset(
    X_images: np.ndarray,
    X_feats: np.ndarray,
    y: np.ndarray,
    out_dir: str,
    prefix: str = "",
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    def add_prefix(name: str) -> str:
        return f"{prefix}{name}" if prefix else name

    np.save(os.path.join(out_dir, add_prefix('X_images.npy')), X_images)
    np.save(os.path.join(out_dir, add_prefix('X_features.npy')), X_feats)
    np.save(os.path.join(out_dir, add_prefix('y_labels.npy')), y)

    meta = {
        'num_samples': int(X_images.shape[0]),
        'image_shape': list(X_images.shape[1:]),
        'num_features': int(X_feats.shape[1]),
        'class_balance': {
            'num_zeros': int((y == 0).sum()),
            'num_ones': int((y == 1).sum()),
        }
    }
    with open(os.path.join(out_dir, add_prefix('meta.json')), 'w') as f:
        json.dump(meta, f, indent=2)


def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if data.empty:
        raise RuntimeError(f"No data downloaded for {ticker} between {start} and {end}")
    data.index = pd.to_datetime(data.index)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GASF images and labels from yfinance data")
    parser.add_argument('--ticker', type=str, default='AAPL', help='Ticker symbol, e.g., AAPL, SPY')
    parser.add_argument('--start', type=str, default='2018-01-01', help='Start date YYYY-MM-DD')
    parser.add_argument('--end', type=str, default='2023-12-31', help='End date YYYY-MM-DD')
    parser.add_argument('--window', type=int, default=60, help='Sliding window size (timesteps)')
    parser.add_argument('--horizon', type=int, default=1, help='Prediction horizon (days ahead)')
    parser.add_argument('--step', type=int, default=1, help='Step size between windows')
    parser.add_argument('--img_size', type=int, default=64, help='GASF image size (pixels)')
    parser.add_argument('--channels', type=str, default='Close,RSI_14,MACD', help='Comma-separated features for GASF channels')
    parser.add_argument('--out_dir', type=str, default='IMD_Project/GASF_images', help='Output directory for npy files (relative to repo root)')
    args = parser.parse_args()

    # Prepare directories
    out_dir = args.out_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(os.getcwd(), out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Downloading {args.ticker} from {args.start} to {args.end} ...")
    raw = download_data(args.ticker, args.start, args.end)
    print(f"Downloaded {len(raw)} rows")

    print("Computing indicators ...")
    df = compute_technical_indicators(raw)

    channel_features = [c.strip() for c in args.channels.split(',') if c.strip()]
    missing = [c for c in channel_features if c not in df.columns]
    if missing:
        raise ValueError(f"Channel features missing in dataframe: {missing}")

    print("Building dataset (this may take a minute) ...")
    X_images, X_feats, y = build_dataset(
        df=df,
        window_size=args.window,
        horizon=args.horizon,
        step_size=args.step,
        image_size=args.img_size,
        channel_features=channel_features,
    )

    print(f"Dataset: X_images={X_images.shape}, X_features={X_feats.shape}, y={y.shape}, pos_rate={y.mean():.3f}")

    print(f"Saving to {out_dir} ...")
    save_dataset(X_images, X_feats, y, out_dir=out_dir)
    print("Done.")
