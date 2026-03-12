"""
F1 Tire Degradation Predictor — Bi-LSTM Model
==============================================
Bidirectional LSTM that maps a window of lap telemetry sequences
to predicted lap-time deltas for the next FORECAST_HORIZON laps.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# 1. PYTORCH DATASET
# ─────────────────────────────────────────────────────────────────────────────
class TireDataset(Dataset):
    """Wraps (X, y) numpy arrays as a PyTorch Dataset."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def make_loaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.10,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Split dataset and return train / val / test DataLoaders."""
    dataset = TireDataset(X, y)
    n       = len(dataset)
    n_test  = max(1, int(n * test_split))
    n_val   = max(1, int(n * val_split))
    n_train = n - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. BI-LSTM MODEL
# ─────────────────────────────────────────────────────────────────────────────
class BiLSTMTirePredictor(nn.Module):
    """
    Architecture
    ─────────────
    Input   (batch, seq_len, n_features)
        │
    ┌───┴────────────────────────────────────┐
    │  Bidirectional LSTM ×2  (stacked)      │
    │  hidden_size × 2 (fwd + bwd)           │
    └───┬────────────────────────────────────┘
        │  last time-step  (batch, hidden×2)
    Dropout(0.3)
        │
    FC: hidden×2 → 128 → ReLU
        │
    FC: 128 → forecast_horizon
        │
    Output  (batch, forecast_horizon)  — lap-time deltas in seconds
    """

    def __init__(
        self,
        n_features:       int,
        hidden_size:      int = 128,
        num_layers:       int = 2,
        dropout:          float = 0.3,
        forecast_horizon: int = 5,
    ):
        super().__init__()
        self.hidden_size      = hidden_size
        self.num_layers       = num_layers
        self.forecast_horizon = forecast_horizon

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, forecast_horizon),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        lstm_out, _ = self.lstm(x)          # (batch, seq_len, hidden×2)
        last_step   = lstm_out[:, -1, :]    # take final time-step
        return self.head(last_step)         # (batch, forecast_horizon)


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAINER
# ─────────────────────────────────────────────────────────────────────────────
class Trainer:
    """
    Minimal but complete training loop with:
      • Huber loss (robust to lap-time outliers)
      • CosineAnnealingLR scheduler
      • Early stopping
      • Checkpoint saving
    """

    def __init__(
        self,
        model:          BiLSTMTirePredictor,
        train_loader:   DataLoader,
        val_loader:     DataLoader,
        lr:             float = 1e-3,
        weight_decay:   float = 1e-4,
        patience:       int   = 10,
        checkpoint_dir: str   = "./checkpoints",
    ):
        self.model          = model
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.patience       = patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = nn.HuberLoss(delta=1.0)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-5
        )

        print(f"[Trainer] Device: {self.device}")
        print(f"[Trainer] Parameters: {sum(p.numel() for p in model.parameters()):,}")

    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        self.model.train(train)
        total_loss = 0.0
        with torch.set_grad_enabled(train):
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                preds   = self.model(X_batch)
                loss    = self.criterion(preds, y_batch)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                total_loss += loss.item() * len(X_batch)
        return total_loss / len(loader.dataset)

    def fit(self, epochs: int = 100) -> dict:
        best_val_loss = float("inf")
        patience_ctr  = 0
        history       = {"train_loss": [], "val_loss": []}

        for epoch in range(1, epochs + 1):
            train_loss = self._run_epoch(self.train_loader, train=True)
            val_loss   = self._run_epoch(self.val_loader,   train=False)
            self.scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_ctr  = 0
                torch.save(self.model.state_dict(), self.checkpoint_dir / "best_model.pt")
            else:
                patience_ctr += 1

            if epoch % 10 == 0 or patience_ctr == 0:
                print(
                    f"Epoch {epoch:03d}/{epochs} | "
                    f"Train: {train_loss:.4f}s | Val: {val_loss:.4f}s | "
                    f"LR: {self.scheduler.get_last_lr()[0]:.2e}"
                )

            if patience_ctr >= self.patience:
                print(f"[Trainer] Early stopping at epoch {epoch}.")
                break

        # Restore best weights
        self.model.load_state_dict(torch.load(self.checkpoint_dir / "best_model.pt"))
        print(f"[Trainer] Best val loss: {best_val_loss:.4f}s")
        return history

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        preds_list, truth_list = [], []
        for X_batch, y_batch in loader:
            preds_list.append(self.model(X_batch.to(self.device)).cpu().numpy())
            truth_list.append(y_batch.numpy())
        return np.concatenate(preds_list), np.concatenate(truth_list)


# ─────────────────────────────────────────────────────────────────────────────
# 4. QUICK SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    BATCH, SEQ_LEN, N_FEAT, HORIZON = 16, 10, 18, 5

    dummy_X = np.random.randn(200, SEQ_LEN, N_FEAT).astype(np.float32)
    dummy_y = np.random.randn(200, HORIZON).astype(np.float32)

    train_dl, val_dl, test_dl = make_loaders(dummy_X, dummy_y, batch_size=BATCH)

    model   = BiLSTMTirePredictor(n_features=N_FEAT, forecast_horizon=HORIZON)
    trainer = Trainer(model, train_dl, val_dl, patience=5)
    history = trainer.fit(epochs=30)

    preds, truth = trainer.predict(test_dl)
    print(f"Test predictions shape: {preds.shape}")
    print("Sanity check passed ✓")