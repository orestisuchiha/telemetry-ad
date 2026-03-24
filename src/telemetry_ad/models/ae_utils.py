from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from telemetry_ad.models.cnn_ae import CNNAutoencoder
from telemetry_ad.models.lstm_ae import LSTMAutoencoder
from telemetry_ad.preprocessing.windowing import sliding_windows


def make_windows(x: np.ndarray, window_size: int, stride: int = 1) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return sliding_windows(arr, window_size=window_size, stride=stride)


def _fit_autoencoder(
    model: nn.Module,
    windows: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    model_type: str,
) -> nn.Module:
    model = model.to(device)
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x = torch.tensor(windows, dtype=torch.float32)
    loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            if model_type == "cnn":
                inp = batch.permute(0, 2, 1)
                recon = model(inp)
                loss = criterion(recon, inp)
            else:
                recon = model(batch)
                loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
    return model


def train_lstm_autoencoder(
    windows: np.ndarray,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
) -> LSTMAutoencoder:
    input_dim = windows.shape[-1]
    model = LSTMAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim)
    return _fit_autoencoder(model, windows, epochs, batch_size, lr, device, model_type="lstm")


def train_cnn_autoencoder(
    windows: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
) -> CNNAutoencoder:
    channels = windows.shape[-1]
    model = CNNAutoencoder(channels=channels)
    return _fit_autoencoder(model, windows, epochs, batch_size, lr, device, model_type="cnn")


@torch.no_grad()
def reconstruction_scores(
    model: nn.Module,
    windows: np.ndarray,
    device: str,
    model_type: str,
) -> np.ndarray:
    model = model.to(device)
    model.eval()
    x = torch.tensor(windows, dtype=torch.float32, device=device)
    if model_type == "cnn":
        inp = x.permute(0, 2, 1)
        recon = model(inp)
        mse = ((recon - inp) ** 2).mean(dim=(1, 2))
    else:
        recon = model(x)
        mse = ((recon - x) ** 2).mean(dim=(1, 2))
    return mse.detach().cpu().numpy()


@torch.no_grad()
def reconstruction_error_breakdown(
    model: nn.Module,
    windows: np.ndarray,
    device: str,
    model_type: str,
) -> dict:
    model = model.to(device)
    model.eval()
    x = torch.tensor(windows, dtype=torch.float32, device=device)
    if model_type == "cnn":
        inp = x.permute(0, 2, 1)
        recon = model(inp)
        err = (recon - inp) ** 2
        score = err.mean(dim=(1, 2))
        channel_error = err.mean(dim=2)
        timestep_error = err.mean(dim=1)
    else:
        recon = model(x)
        err = (recon - x) ** 2
        score = err.mean(dim=(1, 2))
        channel_error = err.mean(dim=1)
        timestep_error = err.mean(dim=2)

    return {
        "scores": score.detach().cpu().numpy(),
        "channel_error": channel_error.detach().cpu().numpy(),
        "timestep_error": timestep_error.detach().cpu().numpy(),
    }
