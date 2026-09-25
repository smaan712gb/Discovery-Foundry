"""Domain-agnostic performance metrics from per-trade and per-day PnL (currency units)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def sharpe(daily: np.ndarray, annualization: int) -> float:
    """Annualized Sharpe of daily PnL (zero-PnL days included). 0.0 if undefined."""
    if daily.size < 2:
        return 0.0
    sd = float(np.std(daily, ddof=1))
    return 0.0 if sd == 0 else float(np.mean(daily)) / sd * math.sqrt(annualization)


def sortino(daily: np.ndarray, annualization: int) -> float:
    """Annualized Sortino; downside deviation = sqrt(mean(min(x, 0)^2)) over all days."""
    if daily.size < 2:
        return 0.0
    dd = math.sqrt(float(np.mean(np.minimum(daily, 0.0) ** 2)))
    return 0.0 if dd == 0 else float(np.mean(daily)) / dd * math.sqrt(annualization)


def max_drawdown(daily: np.ndarray) -> float:
    """Largest peak-to-trough fall of cumulative PnL (a positive number), starting from 0."""
    if daily.size == 0:
        return 0.0
    equity = np.concatenate([[0.0], np.cumsum(daily)])
    return float(np.max(np.maximum.accumulate(equity) - equity))


def trade_metrics(
    net: np.ndarray,
    gross: np.ndarray,
    daily: np.ndarray,
    bars_in_market: int,
    tradeable_bars: int,
    annualization: int,
) -> dict[str, Any]:
    wins = net[net > 0]
    losses = net[net < 0]
    loss_sum = float(-losses.sum())
    pf: float | None = float(wins.sum()) / loss_sum if loss_sum > 0 else None
    return {
        "net_pnl": float(net.sum()),
        "gross_pnl": float(gross.sum()),
        "trade_count": int(net.size),
        "win_rate": float(wins.size / net.size) if net.size else 0.0,
        "profit_factor": pf,
        "avg_trade": float(net.mean()) if net.size else 0.0,
        "sharpe": sharpe(daily, annualization),
        "sortino": sortino(daily, annualization),
        "max_drawdown": max_drawdown(daily),
        "time_in_market": bars_in_market / tradeable_bars if tradeable_bars else 0.0,
        "trading_days": int(daily.size),
    }
