"""
LSTM for multi-step energy load forecasting.

Architecture: 2-layer LSTM → dropout → linear head
Input:  sliding window of SEQ_LEN hours of features
Output: HORIZON hours of meter_reading (one building at a time)

Run:
    python -m src.models.train_lstm
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from torch.utils.data import ConcatDataset, DataLoader, Dataset

import mlflow
import mlflow.pytorch

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROCESSED = Path(os.getenv("DATA_PROCESSED_DIR", "data/processed"))

SEQ_LEN = 168       # one week of hourly history as input
HORIZON = 24        # predict next 24 hours
STRIDE = 24         # sample one window per day — consecutive hourly windows are ~96% correlated
BATCH_SIZE = 256
EPOCHS = 30
LR = 1e-3
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2
CLIP_GRAD = 1.0

FEATURE_COLS = [
    "hour", "day_of_week", "month", "is_weekend", "is_business_hours",
    "temp_c", "dew_temp_c", "wind_speed_ms", "hdd", "cdd",
    "load_lag_1h", "load_lag_24h", "load_lag_168h",
    "load_rolling_mean_24h", "load_rolling_std_24h", "load_rolling_mean_168h",
    "use_type_encoded", "log_sqm", "building_age",
]
TARGET = "meter_reading"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class EnergyWindowDataset(Dataset):
    """
    Sliding window dataset for a SINGLE building's time series.

    Each sample:
        X: (SEQ_LEN, n_features)  — normalised feature window
        y: (HORIZON,)             — raw target values for the next HORIZON steps

    Windows are computed lazily in __getitem__ — only the raw arrays are stored,
    not all (~n_timesteps) pre-materialised windows.
    """

    def __init__(self, df: pd.DataFrame, scaler: StandardScaler) -> None:
        df = df.sort_values("timestamp").reset_index(drop=True)
        df = df[FEATURE_COLS + [TARGET]].dropna()

        self.features = scaler.transform(df[FEATURE_COLS].values.astype(np.float32))
        self.targets  = df[TARGET].values.astype(np.float32)
        self.n        = max(0, (len(df) - SEQ_LEN - HORIZON) // STRIDE + 1)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        start = idx * STRIDE
        x = torch.tensor(self.features[start : start + SEQ_LEN])
        y = torch.tensor(self.targets[start + SEQ_LEN : start + SEQ_LEN + HORIZON])
        return x, y


def build_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
    """Fit scaler on all train features, build per-building datasets, return DataLoaders.

    Per-building split prevents sliding windows from crossing building boundaries,
    which would feed the LSTM sequences mixing two unrelated time series.
    """
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_COLS].dropna().values.astype(np.float32))

    def make_dataset(df: pd.DataFrame) -> ConcatDataset:
        parts = [
            EnergyWindowDataset(bdf, scaler)
            for _, bdf in df.groupby("building_id", observed=True)
            if len(bdf) > SEQ_LEN + HORIZON
        ]
        return ConcatDataset(parts)

    train_ds = make_dataset(train_df)
    val_ds   = make_dataset(val_df)
    test_ds  = make_dataset(test_df)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, scaler


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class EnergyLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
        horizon: int = HORIZON,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])   # take last timestep
        return self.head(out)               # (batch, horizon)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def run_epoch(
    model: EnergyLSTM,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    """One pass through the loader. Pass optimizer=None for eval mode."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0

    with torch.set_grad_enabled(training):
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            preds = model(X_batch)
            loss = criterion(preds, y_batch)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
                optimizer.step()

            total_loss += loss.item() * len(X_batch)

    return total_loss / len(loader.dataset)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_pred = np.clip(y_pred, 0, None)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    threshold = np.percentile(y_true, 1)
    mask = y_true > max(threshold, 1.0)
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    return {"rmse": rmse, "mae": mae, "mape": mape}


def predict_all(model: EnergyLSTM, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds_list, targets_list = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            preds_list.append(model(X_batch.to(DEVICE)).cpu().numpy())
            targets_list.append(y_batch.numpy())
    return np.concatenate(preds_list), np.concatenate(targets_list)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("building-energy-baseline")

    print("Loading features...")
    df = pd.read_parquet(PROCESSED / "features.parquet")
    print(f"  {len(df):,} rows, {df['building_id'].nunique()} buildings")

    # Time-based split — same boundaries as train_baseline.py
    test_start = df["timestamp"].max() - pd.DateOffset(months=3)
    val_start  = test_start - pd.DateOffset(months=1)

    train_df = df[df["timestamp"] <  val_start].copy()
    val_df   = df[(df["timestamp"] >= val_start) & (df["timestamp"] < test_start)].copy()
    test_df  = df[df["timestamp"] >= test_start].copy()

    print(f"  train: {len(train_df):,} | val: {len(val_df):,} | test: {len(test_df):,}")

    print("Building data loaders...")
    train_loader, val_loader, test_loader, scaler = build_loaders(
        train_df, val_df, test_df
    )
    print(f"  train batches: {len(train_loader)}, val: {len(val_loader)}, test: {len(test_loader)}")

    model = EnergyLSTM(input_size=len(FEATURE_COLS)).to(DEVICE)
    criterion = nn.HuberLoss()                                      # robust to outlier spikes
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5, verbose=True
    )

    print(f"\nTraining LSTM on {DEVICE}...")
    with mlflow.start_run(run_name="lstm"):
        mlflow.log_params({
            "model": "lstm",
            "seq_len": SEQ_LEN,
            "horizon": HORIZON,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "epochs": EPOCHS,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
        })

        best_val_loss = float("inf")
        best_model_path = PROCESSED.parent / "lstm_best.pt"

        for epoch in range(1, EPOCHS + 1):
            train_loss = run_epoch(model, train_loader, criterion, optimizer)
            val_loss   = run_epoch(model, val_loader,   criterion, None)
            scheduler.step(val_loss)

            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss}, step=epoch
            )
            print(f"  epoch {epoch:02d}/{EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)

        # Reload best checkpoint for test evaluation
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

        y_pred, y_true = predict_all(model, test_loader)
        # flatten horizon dimension for aggregate metrics
        metrics = compute_metrics(y_true.ravel(), y_pred.ravel())
        for k, v in metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        mlflow.log_artifact(str(best_model_path), artifact_path="lstm")
        mlflow.pytorch.log_model(model, artifact_path="lstm_model")

        print("\n" + "=" * 45)
        print(f"  RMSE: {metrics['rmse']:.2f}")
        print(f"  MAE:  {metrics['mae']:.2f}")
        print(f"  MAPE: {metrics['mape']:.2f}%")
        print("=" * 45)
        print(f"\nBest model saved → {best_model_path}")
