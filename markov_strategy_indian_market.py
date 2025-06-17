"""Markov chain trading strategy for Indian markets.

Dependencies:
    pip install pandas numpy matplotlib seaborn scikit-learn ta yfinance

This script demonstrates how to load OHLCV data, transform price changes
into market states, build an order-n Markov transition matrix, optionally
combine Markov probabilities with a machine learning model, backtest a
simple strategy and plot results. Multiple models can be evaluated and
their performance compared.

Usage example::

    python markov_strategy_indian_market.py --symbol ^NSEI --order 2 \
        --start 2020-01-01 --end 2024-01-01 --up 0.6 --down 0.6 --plot

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import yfinance as yf
except Exception:  # pragma: no cover - yfinance might be unavailable
    yf = None  # type: ignore

# Yahoo Finance tickers for Nifty50 constituents
NIFTY50_TICKERS: List[str] = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "INFY.NS",
    "HINDUNILVR.NS",
    "ITC.NS",
    "SBIN.NS",
    "HDFC.NS",
    "BHARTIARTL.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "AXISBANK.NS",
    "BAJFINANCE.NS",
    "ASIANPAINT.NS",
    "HCLTECH.NS",
    "DMART.NS",
    "MARUTI.NS",
    "TITAN.NS",
    "ULTRACEMCO.NS",
    "SUNPHARMA.NS",
    "ADANIENT.NS",
    "ADANIPORTS.NS",
    "NESTLEIND.NS",
    "ONGC.NS",
    "NTPC.NS",
    "POWERGRID.NS",
    "COALINDIA.NS",
    "M&M.NS",
    "BAJAJ-AUTO.NS",
    "BPCL.NS",
    "INDUSINDBK.NS",
    "TECHM.NS",
    "TATASTEEL.NS",
    "HEROMOTOCO.NS",
    "EICHERMOT.NS",
    "WIPRO.NS",
    "HDFCLIFE.NS",
    "GRASIM.NS",
    "DIVISLAB.NS",
    "BRITANNIA.NS",
    "CIPLA.NS",
    "SBILIFE.NS",
    "TATACONSUM.NS",
    "DRREDDY.NS",
    "BAJAJFINSV.NS",
    "JSWSTEEL.NS",
    "UPL.NS",
    "APOLLOHOSP.NS",
]

try:  # ML dependencies are optional
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
except Exception:  # pragma: no cover - sklearn might be unavailable
    RandomForestClassifier = None  # type: ignore

try:  # technical indicators
    from ta.momentum import RSIIndicator
    from ta.trend import EMAIndicator
except Exception:  # pragma: no cover - ta-lib might be unavailable
    RSIIndicator = None  # type: ignore
    EMAIndicator = None  # type: ignore


@dataclass
class MarkovModel:
    """Container for a Markov transition matrix."""

    order: int
    matrix: np.ndarray
    seq_to_idx: Dict[Tuple[str, ...], int]
    idx_to_seq: Dict[int, Tuple[str, ...]]
    state_to_idx: Dict[str, int]
    idx_to_state: Dict[int, str]

    def next_state_prob(self, sequence: Sequence[str]) -> np.ndarray:
        """Return probability distribution for the next state."""
        key = tuple(sequence[-self.order :])
        idx = self.seq_to_idx.get(key)
        if idx is None:
            raise KeyError(f"Sequence {key} not found in model")
        return self.matrix[idx]

    def most_likely_next(self, sequence: Sequence[str]) -> str:
        """Return state with highest probability for a given sequence."""
        probs = self.next_state_prob(sequence)
        return self.idx_to_state[int(np.argmax(probs))]


def download_yahoo_data(symbol: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance."""

    if yf is None:
        raise RuntimeError("yfinance is required for downloading data")

    data = yf.download(symbol, start=start, end=end, progress=False)
    data.reset_index(inplace=True)
    data.rename(columns={
        "Adj Close": "Adj_Close",
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close",
        "Volume": "Volume",
    }, inplace=True)
    data.rename(columns={"Date": "Date"}, inplace=True)
    return data[["Date", "Open", "High", "Low", "Close", "Volume"]]


def download_nifty50_data(start: Optional[str] = None, end: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """Download data for all Nifty50 constituents."""

    datasets: Dict[str, pd.DataFrame] = {}
    for ticker in NIFTY50_TICKERS:
        try:
            df = download_yahoo_data(ticker, start=start, end=end)
            datasets[ticker] = df
        except Exception:
            continue
    return datasets


def load_data(path: Optional[str] = None, symbol: Optional[str] = None, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """Load data from CSV or download from Yahoo Finance."""

    if path:
        df = pd.read_csv(path, parse_dates=["Date"])
        df.sort_values("Date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    if symbol:
        return download_yahoo_data(symbol, start=start, end=end)
    raise ValueError("Either path or symbol must be provided")


def generate_states(df: pd.DataFrame, price_col: str = "Close", threshold: float = 0.0) -> pd.Series:
    """Convert price changes into discrete market states."""

    returns = df[price_col].pct_change()
    states = pd.Series(index=df.index, dtype=object)
    states[returns > threshold] = "UP"
    states[returns < -threshold] = "DOWN"
    states[(returns >= -threshold) & (returns <= threshold)] = "NO_CHANGE"
    return states


def build_markov_matrix(states: pd.Series, order: int = 1) -> MarkovModel:
    """Construct an order-n Markov transition matrix."""

    states = states.dropna().reset_index(drop=True)
    uniq_states = sorted(states.unique())
    state_to_idx = {s: i for i, s in enumerate(uniq_states)}
    idx_to_state = {i: s for s, i in state_to_idx.items()}

    counts: Dict[Tuple[str, ...], np.ndarray] = {}
    for i in range(order, len(states)):
        seq = tuple(states.iloc[i - order : i])
        nxt = states.iloc[i]
        if seq not in counts:
            counts[seq] = np.zeros(len(uniq_states))
        counts[seq][state_to_idx[nxt]] += 1

    seq_to_idx = {seq: idx for idx, seq in enumerate(sorted(counts))}
    idx_to_seq = {idx: seq for seq, idx in seq_to_idx.items()}
    matrix = np.zeros((len(seq_to_idx), len(uniq_states)))
    for seq, idx in seq_to_idx.items():
        row = counts[seq]
        total = row.sum()
        if total == 0:
            matrix[idx] = np.ones(len(uniq_states)) / len(uniq_states)
        else:
            matrix[idx] = row / total

    return MarkovModel(order, matrix, seq_to_idx, idx_to_seq, state_to_idx, idx_to_state)


def query_transition(model: MarkovModel, sequence: Sequence[str]) -> Dict[str, float]:
    """Return transition probabilities for a given state sequence."""

    probs = model.next_state_prob(sequence)
    return {model.idx_to_state[i]: float(p) for i, p in enumerate(probs)}


def simulate_next_state(model: MarkovModel, sequence: Sequence[str]) -> str:
    """Simulate the most likely next state from ``sequence``."""

    return model.most_likely_next(sequence)


def _prepare_ml_features(df: pd.DataFrame, states: pd.Series, lookback: int, model: MarkovModel) -> Tuple[pd.DataFrame, pd.Series]:
    """Create ML features using previous states and technical indicators."""

    if RSIIndicator is None or EMAIndicator is None:
        raise RuntimeError("ta package required for ML features")

    features = pd.DataFrame(index=df.index)
    for i in range(1, lookback + 1):
        features[f"state_{i}"] = states.shift(i).map(model.state_to_idx)

    rsi = RSIIndicator(df["Close"], window=14).rsi()
    ema = EMAIndicator(df["Close"], window=14).ema_indicator()
    features["rsi"] = rsi
    features["ema"] = ema

    labels = states.shift(-1)

    valid = features.notna().all(axis=1) & labels.notna()
    return features[valid], labels[valid]


def train_ml_model(features: pd.DataFrame, labels: pd.Series) -> Tuple[RandomForestClassifier, float]:
    """Train a RandomForest classifier and return it with accuracy."""

    if RandomForestClassifier is None:
        raise RuntimeError("scikit-learn is required for ML integration")

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(features, labels)
    preds = clf.predict(features)
    acc = accuracy_score(labels, preds)
    return clf, float(acc)


def combine_probabilities(
    markov_probs: np.ndarray,
    ml_probs: Optional[np.ndarray],
    markov_weight: float = 0.5,
) -> np.ndarray:
    """Combine probabilities with a configurable weight."""

    if ml_probs is None:
        return markov_probs
    return markov_weight * markov_probs + (1.0 - markov_weight) * ml_probs


def backtest(
    df: pd.DataFrame,
    states: pd.Series,
    model: MarkovModel,
    up_threshold: float,
    down_threshold: float,
    ml_clf: Optional[RandomForestClassifier] = None,
    ml_features: Optional[pd.DataFrame] = None,
    markov_weight: float = 0.5,
) -> pd.DataFrame:
    """Run a simple backtest using Markov probabilities."""

    positions = []  # 1 long, -1 short, 0 flat
    equity = [1.0]

    for i in range(model.order, len(df)):
        seq = states.iloc[i - model.order : i]
        markov_probs = model.next_state_prob(seq)

        ml_probs = None
        if ml_clf is not None and ml_features is not None and i in ml_features.index:
            ml_prob_df = ml_clf.predict_proba(ml_features.loc[[i]])[0]
            ml_probs = ml_prob_df

        probs = combine_probabilities(markov_probs, ml_probs, markov_weight)

        if probs[model.state_to_idx["UP"]] > up_threshold:
            pos = 1
        elif probs[model.state_to_idx["DOWN"]] > down_threshold:
            pos = -1
        else:
            pos = 0

        positions.append(pos)

        ret = df["Close"].pct_change().iloc[i]
        new_equity = equity[-1] * (1 + pos * ret)
        equity.append(new_equity)

    bt_index = df.index[model.order :]
    return pd.DataFrame({"Position": positions, "Equity": equity[1:]}, index=bt_index)


def compute_metrics(equity_curve: pd.Series) -> Dict[str, float]:
    """Calculate basic performance metrics."""

    returns = equity_curve.pct_change().dropna()
    total_return = equity_curve.iloc[-1] - 1.0
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if not returns.empty else 0.0
    win_rate = (returns > 0).mean()
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = drawdown.min()
    return {
        "total_return": float(total_return),
        "sharpe": float(sharpe),
        "win_rate": float(win_rate),
        "max_drawdown": float(max_drawdown),
    }


def plot_results(df: pd.DataFrame, model: MarkovModel, bt: pd.DataFrame, show: bool = True) -> None:
    """Plot heatmap, equity curve and trade markers."""

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # Heatmap of transition matrix
    sns.heatmap(model.matrix, annot=True, fmt=".2f", cmap="Blues", ax=axes[0],
                xticklabels=model.idx_to_state.values(),
                yticklabels=["->".join(seq) for seq in model.idx_to_seq.values()])
    axes[0].set_title("Transition probabilities")

    # Equity curve
    axes[1].plot(df["Date"].iloc[model.order :], bt["Equity"], label="Equity")
    axes[1].set_title("Equity Curve")
    axes[1].legend()

    # Price with trades
    axes[2].plot(df["Date"], df["Close"], label="Close")
    buy_signals = bt.index[bt["Position"] == 1]
    sell_signals = bt.index[bt["Position"] == -1]
    axes[2].scatter(df.loc[buy_signals, "Date"], df.loc[buy_signals, "Close"], marker="^", color="g", label="Buy")
    axes[2].scatter(df.loc[sell_signals, "Date"], df.loc[sell_signals, "Close"], marker="v", color="r", label="Sell")
    axes[2].set_title("Trades")
    axes[2].legend()

    fig.tight_layout()
    if show:
        plt.show()


def run_single(df: pd.DataFrame, args: argparse.Namespace, label: str = "") -> None:
    """Execute the full pipeline on one dataset."""
    if args.years:
        end_date = df["Date"].max()
        start_date = end_date - pd.DateOffset(years=args.years)
        df = df[df["Date"] >= start_date].reset_index(drop=True)

    states = generate_states(df, threshold=args.threshold)
    model = build_markov_matrix(states, order=args.order)

    if args.save_matrix:
        out = args.save_matrix
        if label:
            out = Path(f"{label}_{Path(out).name}")
        save_matrix(model, out)

    if args.query:
        seq = [s.strip() for s in args.query]
        probs = query_transition(model, seq)
        for k, v in probs.items():
            print(f"{label}{k}: {v:.2%}")
        return

    ml_clf = None
    ml_features = None
    if args.enable_ml:
        ml_features, ml_labels = _prepare_ml_features(df, states, lookback=5, model=model)
        ml_clf, acc = train_ml_model(ml_features, ml_labels)
        print(f"{label}ML training accuracy: {acc:.2%}")

    if args.compare and args.enable_ml:
        results = {}

        bt_mkv = backtest(
            df,
            states,
            model,
            up_threshold=args.up,
            down_threshold=args.down,
            markov_weight=1.0,
        )
        results["Markov"] = compute_metrics(bt_mkv["Equity"])

        bt_ml = backtest(
            df,
            states,
            model,
            up_threshold=args.up,
            down_threshold=args.down,
            ml_clf=ml_clf,
            ml_features=ml_features,
            markov_weight=0.0,
        )
        results["ML"] = compute_metrics(bt_ml["Equity"])

        bt_comb = backtest(
            df,
            states,
            model,
            up_threshold=args.up,
            down_threshold=args.down,
            ml_clf=ml_clf,
            ml_features=ml_features,
            markov_weight=0.5,
        )
        results["Combined"] = compute_metrics(bt_comb["Equity"])

        print(f"{label}Performance comparison:")
        for name, m in results.items():
            print(f"  {name}:")
            for k, v in m.items():
                print(f"    {k}: {v:.2%}")
        pd.DataFrame(results).to_csv(f"metrics_{label.rstrip('_') or 'result'}.csv")
        if args.plot:
            plot_results(df, model, bt_comb)
        return

    bt = backtest(
        df,
        states,
        model,
        up_threshold=args.up,
        down_threshold=args.down,
        ml_clf=ml_clf,
        ml_features=ml_features,
    )

    metrics = compute_metrics(bt["Equity"])
    print(f"{label}Backtest metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.2%}")

    results = bt.join(df[["Date"]], how="left")
    fname = f"backtest_results_{label.rstrip('_') or 'result'}.csv"
    results.to_csv(fname, index=False)

    if args.plot:
        plot_results(df, model, bt)


def save_matrix(model: MarkovModel, path: Path) -> None:
    """Save transition matrix as CSV."""

    df = pd.DataFrame(
        model.matrix,
        index=["|".join(seq) for seq in model.idx_to_seq.values()],
        columns=model.idx_to_state.values(),
    )
    df.to_csv(path)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Markov chain trading model for Indian markets")
    parser.add_argument("--data", help="Path to CSV with Date/Open/High/Low/Close/Volume")
    parser.add_argument("--symbol", help="Yahoo Finance ticker to download, default ^NSEI for Nifty50")
    parser.add_argument("--nifty50", action="store_true", help="Download all Nifty50 constituents")
    parser.add_argument("--start", help="Start date for Yahoo download")
    parser.add_argument("--end", help="End date for Yahoo download")
    parser.add_argument("--years", type=int, default=8, help="Number of years of data to use")
    parser.add_argument("--order", type=int, default=1, help="Order of the Markov model")
    parser.add_argument("--threshold", type=float, default=0.0, help="Return threshold for state generation")
    parser.add_argument("--up", type=float, default=0.6, help="Buy probability threshold")
    parser.add_argument("--down", type=float, default=0.6, help="Sell probability threshold")
    parser.add_argument("--query", nargs="*", help="State sequence to query, comma separated")
    parser.add_argument("--enable_ml", action="store_true", help="Enable optional ML classifier")
    parser.add_argument("--compare", action="store_true", help="Compare Markov, ML and combined models")
    parser.add_argument("--plot", action="store_true", help="Display plots")
    parser.add_argument("--save_matrix", type=Path, help="Save transition matrix to CSV")
    args = parser.parse_args(argv)

    if args.start is None and args.symbol and not args.data:
        end_ts = pd.to_datetime(args.end) if args.end else pd.Timestamp.today()
        args.start = (end_ts - pd.DateOffset(years=args.years)).strftime("%Y-%m-%d")
        if args.end is None:
            args.end = end_ts.strftime("%Y-%m-%d")

    if args.nifty50:
        datasets = download_nifty50_data(start=args.start, end=args.end)
        for sym, df in datasets.items():
            print(f"\nRunning for {sym}")
            run_single(df, args, label=f"{sym}_")
        return

    symbol = args.symbol
    if symbol is None and not args.data:
        symbol = "^NSEI"

    df = load_data(path=args.data, symbol=symbol, start=args.start, end=args.end)
    run_single(df, args)


if __name__ == "__main__":  # pragma: no cover
    main()

