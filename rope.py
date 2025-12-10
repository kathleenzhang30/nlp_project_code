import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class RotaryPositionalEmbedding(nn.Module):
    """
    Generates RoPE embeddings for a given sequence length and embedding dimension.
    """
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i , j -> i j", t, inv_freq)  
        emb = torch.cat((freqs, freqs), dim=-1)            
        self.register_buffer("cos_emb", emb.cos(), persistent=False)
        self.register_buffer("sin_emb", emb.sin(), persistent=False)

    def forward(self, x):
        """
        x: [B, T, D]
        returns x, cos, sin
        """
        T = x.size(1)
        return x, self.cos_emb[:T, :], self.sin_emb[:T, :]

def apply_rope(x, cos, sin):
    """
    Apply rotary positional embeddings.
    x: [B, T, D]
    cos, sin: [T, D]
    """
    cos_even = cos[..., ::2]  
    cos_odd  = cos[..., 1::2]
    sin_even = sin[..., ::2]
    sin_odd  = sin[..., 1::2]

    x1, x2 = x[..., ::2], x[..., 1::2]  

    x1_rot = x1 * cos_even - x2 * sin_even
    x2_rot = x1 * sin_odd  + x2 * cos_odd

    # interleave back
    x_rotated = torch.stack([x1_rot, x2_rot], dim=-1).flatten(-2)
    return x_rotated

class TransformerRegressorRoPE(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=3, dim_feedforward=128, seq_len=30):
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(1, d_model)
        self.pos_emb = RotaryPositionalEmbedding(d_model, max_seq_len=seq_len)
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, 1)

    def forward(self, src):
        """
        src: [B, T] or [B, T, 1]
        returns [B] prediction
        """
        if src.dim() == 2:
            src = src.unsqueeze(-1)  

        x = self.input_proj(src)  

        # apply RoPE
        x, cos, sin = self.pos_emb(x)
        x = apply_rope(x, cos, sin)

        out = self.transformer(x)  
        y = self.output(out[:, -1, :])  
        return y.squeeze(-1)       


seq_len = 30
dataset = SP500Dataset(prices, seq_len=seq_len)

# 80/20 train/test split
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = TransformerRegressorRoPE(d_model=64, nhead=4, num_layers=3, dim_feedforward=128).to(device)
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

runs = 100
mse_sum = 0.0
rmse_sum = 0.0
mae_sum = 0.0
r2_sum = 0.0

for i in range(runs):
    if i % 10 == 0:
        print(i)
    metrics, target_1, preds_1 = evaluate(model, test_loader, device, unnormalize_fn=None)
    mse_sum += metrics["mse"]
    rmse_sum += metrics["rmse"]
    mae_sum += metrics["mae"]
    r2_sum += metrics["r2"]

print(f"\nAverage over {runs} runs (normalized):")
print(f"  MSE : {mse_sum / runs:.6f}")
print(f"  RMSE: {rmse_sum / runs:.6f}")
print(f"  MAE : {mae_sum / runs:.6f}")
print(f"  R2  : {r2_sum / runs:.6f}")


def evaluate(model, loader, device, unnormalize_fn=None):
    """
    Evaluate the model on a DataLoader and return MSE.

    Args:
        model: trained TransformerRegressorRoPE
        loader: DataLoader (test or validation)
        device: "cuda" or "cpu"
        unnormalize_fn: optional function to unnormalize predictions and targets
    """
    model.eval()
    total_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y = y.squeeze(-1)  # ensure shape [B]

            pred = model(x)

            if unnormalize_fn is not None:
                y = unnormalize_fn(y)
                pred = unnormalize_fn(pred)

            loss = nn.functional.mse_loss(pred, y, reduction='sum')
            total_loss += loss.item()
            n_samples += y.size(0)

    return total_loss / n_samples




