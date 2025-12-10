pip install yfinance
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import yfinance as yf
import pandas as pd

# Download S&P 500 historical data
data = yf.download('^GSPC', start='2010-01-01', end='2025-01-01', auto_adjust=True)

prices = data['Close']
prices = prices.fillna(method='ffill')  # fill missing days
prices.head()
prices.tail()


class SP500Dataset(Dataset):
    def __init__(self, prices, seq_len=30):
        """
        prices: pd.Series of adjusted close
        seq_len: input sequence length (e.g., 29 inputs, predict 30th)
        """
        self.seq_len = seq_len
        self.prices = prices.values
        # normalize to mean=0, std=1
        self.mean = self.prices.mean()
        self.std = self.prices.std()
        self.prices = (self.prices - self.mean) / self.std

    def __len__(self):
        return len(self.prices) - self.seq_len

    def __getitem__(self, idx):
        x = torch.tensor(self.prices[idx:idx + self.seq_len - 1], dtype=torch.float32)
        y = torch.tensor(self.prices[idx + self.seq_len - 1], dtype=torch.float32)
        return x, y



seq_len = 30
dataset = SP500Dataset(prices, seq_len=seq_len)

# 80/20 train/test split
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])

from torch.utils.data import DataLoader

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)


device = "cuda" if torch.cuda.is_available() else "cpu"

model = TransformerRegressorFull(d_model=64, nhead=4, num_layers=3, dim_feedforward=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()


for epoch in range(50):
    model.train()
    total_loss = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1} | Train MSE: {avg_loss:.6f}")


def evaluate(model, loader, device, unnormalize_fn=None, verbose=False):
    """
    Evaluate the model on a DataLoader and return metrics: MSE, RMSE, MAE, R2.
    Also returns all predictions and targets for plotting.
    
    Args:
        model: PyTorch model
        loader: DataLoader
        device: torch device
        unnormalize_fn: optional function to unnormalize predictions/targets
        verbose: if True, print predictions and targets

    Returns:
        metrics_dict, all_targets, all_preds
    """
    model.eval()
    total_se = 0.0
    total_ae = 0.0
    all_targets = []
    all_preds = []
    n_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y = y.squeeze(-1)  # [B]

            pred = model(x)    # [B]

            if unnormalize_fn is not None:
                y = unnormalize_fn(y)
                pred = unnormalize_fn(pred)

            # accumulate for metrics
            se = F.mse_loss(pred, y, reduction='sum')
            ae = F.l1_loss(pred, y, reduction='sum')

            total_se += se.item()
            total_ae += ae.item()
            n_samples += y.size(0)

            all_targets.append(y.detach().cpu())
            all_preds.append(pred.detach().cpu())

    # concatenate for R2 and plotting
    all_targets = torch.cat(all_targets, dim=0)
    all_preds = torch.cat(all_preds, dim=0)

    if verbose:
        print("Targets:", all_targets)
        print("Predictions:", all_preds)

    mse = total_se / n_samples
    rmse = mse ** 0.5
    mae = total_ae / n_samples

    ss_res = torch.sum((all_targets - all_preds) ** 2).item()
    ss_tot = torch.sum((all_targets - all_targets.mean()) ** 2).item()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }

    return metrics, all_targets, all_preds

runs = 100
mse_sum = 0.0
rmse_sum = 0.0
mae_sum = 0.0
r2_sum = 0.0

for i in range(runs):
    if i % 10 == 0:
        print(i)
    metrics, targets_0, preds_0 = evaluate(model, test_loader, device, unnormalize_fn=None)
    mse_sum += metrics["mse"]
    rmse_sum += metrics["rmse"]
    mae_sum += metrics["mae"]
    r2_sum += metrics["r2"]

print(f"\nAverage over {runs} runs (normalized):")
print(f"  MSE : {mse_sum / runs:.6f}")
print(f"  RMSE: {rmse_sum / runs:.6f}")
print(f"  MAE : {mae_sum / runs:.6f}")
print(f"  R2  : {r2_sum / runs:.6f}")
