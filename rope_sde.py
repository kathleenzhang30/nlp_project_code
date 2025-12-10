import math
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import holidays
import datetime

class RotaryPositionalEmbedding(nn.Module):
    """
    Generates RoPE embeddings for a given sequence length and embedding dimension.
    """
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i , j -> i j", t, inv_freq)  # [seq_len, dim//2]
        emb = torch.cat((freqs, freqs), dim=-1)            # [seq_len, dim]
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
    # slice cos/sin to match interleaved even/odd dimensions
    cos_even = cos[..., ::2]  # [T, D/2]
    cos_odd  = cos[..., 1::2]
    sin_even = sin[..., ::2]
    sin_odd  = sin[..., 1::2]

    x1, x2 = x[..., ::2], x[..., 1::2]  # [B, T, D/2]

    x1_rot = x1 * cos_even - x2 * sin_even
    x2_rot = x1 * sin_odd  + x2 * cos_odd

    # interleave back
    x_rotated = torch.stack([x1_rot, x2_rot], dim=-1).flatten(-2)
    return x_rotated


class TransformerRegressorRoPE(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=3, dim_feedforward=128, seq_len=30, event_emb_dim=8):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        # project scalar price input -> d_model
        self.input_proj = nn.Linear(1, d_model)

        # project event embedding (e.g., 8-d) -> d_model, add to input projection
        self.event_proj = nn.Linear(event_emb_dim, d_model)

        self.pos_emb = RotaryPositionalEmbedding(d_model, max_seq_len=seq_len)

        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, 1)

    def forward(self, src_prices, src_event_emb):
        """
        src_prices: [B, T, 1]
        src_event_emb: [B, T, E] (E == event_emb_dim)
        returns [B] prediction (predicts next-step price or log-price depending on training)
        """
        x_price = self.input_proj(src_prices)           # [B, T, D]
        x_event = self.event_proj(src_event_emb)       # [B, T, D]

        x = x_price + x_event                          # fuse by addition
        x, cos, sin = self.pos_emb(x)
        x = apply_rope(x, cos, sin)

        out = self.transformer(x)                      # [B, T, D]
        y = self.output(out[:, -1, :])                 # use last token
        return y.squeeze(-1)


def prepare_price_dataframe(start='2010-01-01', end='2025-01-01', ticker='^GSPC', fill_holidays=None):
    data = yf.download(ticker, start=start, end=end, auto_adjust=True)
    prices = data['Close'].fillna(method='ffill').reset_index()
    prices.rename(columns={'Date': 'date'}, inplace=True)
    prices['date_str'] = prices['date'].dt.strftime('%Y-%m-%d')
    return prices

def add_special_days(prices_df, special_days):
    """
    special_days: dict mapping 'event_type' -> list of 'YYYY-MM-DD' strings
    returns prices_df with:
      - special_day_type (str)
      - one-hot columns is_<type>
      - positional_embedding column which is numeric vector (np.ndarray)
    """
    def map_special_day(date_str):
        for event_type, days in special_days.items():
            if date_str in days:
                return event_type
        return 'none'
    prices_df['special_day_type'] = prices_df['date_str'].apply(map_special_day)

    event_types = ['none'] + [et for et in special_days.keys() if et != 'none']
    for et in event_types:
        prices_df[f'is_{et}'] = (prices_df['special_day_type'] == et).astype(int)

    embedding_dim = 8
    rng = np.random.RandomState(0)
    event_embeddings = {et: rng.normal(scale=0.1, size=(embedding_dim,)) for et in event_types}
    prices_df['positional_embedding'] = prices_df['special_day_type'].apply(lambda t: event_embeddings[t])

    return prices_df, event_embeddings


class PriceSequenceDataset(Dataset):
    """
    Produces sliding windows of length seq_len from price series.
    Each sample:
      - x_prices: [T, 1] normalized prices (float32)
      - x_event_emb: [T, E] event embeddings aligned to those dates
      - y: target scalar (next-step normalized price)  OR next-day return etc.
    """
    def __init__(self, prices_df, seq_len=30, target='next_log_price', scaler=None):
        """
        prices_df must have columns:
          - 'Close' or the numeric price column
          - 'positional_embedding' vector per row
        target: 'next_log_price' -> predict log(price) at t+1
        """
        self.seq_len = seq_len
        self.df = prices_df.reset_index(drop=True)
        # compute log price and normalized series
        self.df['log_price'] = np.log(self.df['Close'].values + 1e-8)
        if scaler is None:
            self.scaler = StandardScaler()
            self.df['log_price_scaled'] = self.scaler.fit_transform(self.df['log_price'].values.reshape(-1,1)).flatten()
        else:
            self.scaler = scaler
            self.df['log_price_scaled'] = self.scaler.transform(self.df['log_price'].values.reshape(-1,1)).flatten()

        self.log_scaled = self.df['log_price_scaled'].values.astype(np.float32)
        emb_list = np.stack(self.df['positional_embedding'].values).astype(np.float32)  # [N, E]
        self.event_embs = emb_list

        self.idx = []
        N = len(self.df)
        for i in range(N - seq_len):
            self.idx.append(i)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        start = self.idx[i]
        end = start + self.seq_len
        x_prices = self.log_scaled[start:end].reshape(self.seq_len, 1)     
        x_event = self.event_embs[start:end]                             
        y = self.log_scaled[end]                                          
        return torch.from_numpy(x_prices), torch.from_numpy(x_event), torch.tensor(y, dtype=torch.float32)


def train_epoch(model, dataloader, opt, loss_fn, device):
    model.train()
    total_loss = 0.0
    for x_prices, x_event, y in dataloader:
        x_prices = x_prices.to(device)
        x_event  = x_event.to(device)
        y = y.to(device)

        opt.zero_grad()
        pred = model(x_prices, x_event)           # [B]
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        total_loss += float(loss.item()) * x_prices.size(0)
    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    preds = []
    ys = []
    with torch.no_grad():
        for x_prices, x_event, y in dataloader:
            x_prices = x_prices.to(device)
            x_event  = x_event.to(device)
            y = y.to(device)

            pred = model(x_prices, x_event)
            loss = loss_fn(pred, y)
            total_loss += float(loss.item()) * x_prices.size(0)
            preds.append(pred.cpu().numpy())
            ys.append(y.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    ys = np.concatenate(ys, axis=0)
    return total_loss / len(dataloader.dataset), preds, ys


def get_market_holidays(start="2010-01-01", end="2025-01-01"):
    """
    Returns a sorted list of all NYSE market holidays between the given dates.
    """
    us_market = holidays.financial_holidays.USNYSE() 
    holiday_list = [
        d.strftime("%Y-%m-%d")
        for d in us_market
        if start <= d.strftime("%Y-%m-%d") <= end
    ]
    return sorted(holiday_list)

def generate_quarterly_report_days(start_year=2010, end_year=2025):
    quarterly_dates = []
    for year in range(start_year, end_year + 1):
        quarterly_dates.append(f"{year}-03-31")
        quarterly_dates.append(f"{year}-06-30")
        quarterly_dates.append(f"{year}-09-30")
        quarterly_dates.append(f"{year}-12-31")
    return quarterly_dates

def generate_us_election_days(start_year=2010, end_year=2025):
    election_days = []

    for year in range(start_year, end_year + 1):
        if year % 2 != 0:
            continue  # skip odd years (no federal general election)

        nov1 = datetime.date(year, 11, 1)
        first_monday_offset = (0 - nov1.weekday()) % 7  # Monday = 0
        first_monday = nov1 + datetime.timedelta(days=first_monday_offset)

        election_day = first_monday + datetime.timedelta(days=1)
        election_days.append(election_day.strftime("%Y-%m-%d"))

    return election_days

def run_pipeline(
    ticker='^GSPC',
    start='2010-01-01',
    end='2025-01-01',
    seq_len=30,
    d_model=64,
    event_emb_dim=8,
    batch_size=128,
    test_fraction=0.1,
    val_fraction=0.1,
    epochs=30,
    lr=1e-3,
    device=None
):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    prices = prepare_price_dataframe(start=start, end=end, ticker=ticker)

    market_holidays = get_market_holidays(start='2010-01-01', end='2025-01-01')

    quarterly_report_day = generate_quarterly_report_days()

    election_days = generate_us_election_days()
  
    special_days = {
        'holiday': market_holidays,
        'quarterly_report': quarterly_report_days,
        'election': election_days
    }

    prices, event_embeddings_map = add_special_days(prices, special_days)

    dataset = PriceSequenceDataset(prices, seq_len=seq_len, target='next_log_price')

    N = len(dataset)
    test_size = int(N * test_fraction)
    val_size = int(N * val_fraction)
    train_size = N - val_size - test_size
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = TransformerRegressorRoPE(
        d_model=d_model,
        nhead=min(8, d_model//8 if d_model>=8 else 1),
        num_layers=3,
        dim_feedforward=4*d_model,
        seq_len=seq_len,
        event_emb_dim=event_emb_dim
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float('inf')
    for ep in range(1, epochs+1):
        train_loss = train_epoch(model, train_dl, optimizer, loss_fn, device)
        val_loss, _, _ = evaluate(model, val_dl, loss_fn, device)
        print(f"Epoch {ep:03d}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()

    model.load_state_dict(best_state)
    test_loss, preds, ys = evaluate(model, test_dl, loss_fn, device)
    print(f"Test loss (MSE on scaled log-price): {test_loss:.6f}")

    scaler = dataset.scaler
    preds_unscaled = scaler.inverse_transform(preds.reshape(-1,1)).flatten()
    ys_unscaled = scaler.inverse_transform(ys.reshape(-1,1)).flatten()
    preds_price = np.exp(preds_unscaled)
    ys_price = np.exp(ys_unscaled)

    return {
        "model": model,
        "scaler": scaler,
        "test_loss": test_loss,
        "preds": preds,
        "ys": ys,
        "preds_price": preds_price,
        "ys_price": ys_price,
        "prices_df": prices,
        "event_embeddings_map": event_embeddings_map
    }


if __name__ == "__main__":
    out = run_pipeline(
        ticker='^GSPC',
        start='2010-01-01',
        end='2025-01-01',
        seq_len=30,
        d_model=64,
        event_emb_dim=8,
        batch_size=256,
        epochs=15,
        lr=3e-4
    )
