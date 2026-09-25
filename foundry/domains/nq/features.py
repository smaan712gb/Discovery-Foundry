"""Whitelisted features and the per-dataset market context (windows, flat bars, regimes).

Every feature value at bar t uses only bars <= t (and, for session features, only sessions
that have already closed). `tests/test_features.py` checks each feature by hand and checks
prefix invariance: computing on data[:t] gives the same values as computing on the full data.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import cast

import numpy as np
import polars as pl

from foundry.domains.nq.calendar import TradingCalendar
from foundry.domains.nq.config import EXCHANGE_TZ, RegimeConfig
from foundry.domains.nq.quality import FLAG_NAMES
from foundry.domains.nq.sessions import ETH, RTH

WINDOW_SESSIONS = ("RTH", "ETH", "both")
DOW = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
VOL_BUCKETS = ("low", "mid", "high")


@dataclass(frozen=True)
class Window:
    """Tradeable-window layout for one DSL session value."""

    window_id: np.ndarray  # int64, -1 where the bar is not in a window
    flat: np.ndarray  # bool, the bar at whose close the window's positions are flattened


@dataclass
class MarketContext:
    """Everything derived from the bars that doesn't depend on a spec, computed once."""

    frame: pl.DataFrame
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    session: np.ndarray  # str labels
    trading_date: np.ndarray  # datetime64[D]
    windows: dict[str, Window]
    vol_bucket: np.ndarray  # per bar, from earlier days only
    trend_state: np.ndarray
    dow: np.ndarray
    event_day: np.ndarray
    flags: np.ndarray  # uint32 quality bitmask
    _cache: dict[tuple[str, tuple[int, ...]], np.ndarray] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.close.shape[0])

    def feature(self, kind: str, args: tuple[int, ...]) -> np.ndarray:
        key = (kind, args)
        if key not in self._cache:
            self._cache[key] = compute_feature(self.frame, kind, args)
        return self._cache[key]

    def blocked(self, flag_names: list[str]) -> np.ndarray:
        mask = 0
        for bit, name in FLAG_NAMES.items():
            if name in flag_names:
                mask |= bit
        return np.asarray((self.flags & np.uint32(mask)) != 0, dtype=bool)


def build_context(
    bars: pl.DataFrame,
    calendar: TradingCalendar,
    bar_minutes: int,
    regimes: RegimeConfig,
    events: set[date],
) -> MarketContext:
    """`bars`: processed series rows (ts, trading_date, session, OHLCV, quality_flags), sorted."""
    s = calendar.sessions
    frame = bars.sort("ts").with_columns(
        pl.when(pl.col("session").is_in([RTH, ETH])).then(pl.col("session")).alias("seg")
    )
    rth_start = pl.col("trading_date").dt.combine(s.rth_open).dt.replace_time_zone(EXCHANGE_TZ)
    eth_start = (
        (pl.col("trading_date") - pl.duration(days=1))
        .dt.combine(s.globex_open)
        .dt.replace_time_zone(EXCHANGE_TZ)
    )
    seg_start = (
        pl.when(pl.col("seg") == RTH).then(rth_start).when(pl.col("seg") == ETH).then(eth_start)
    )
    frame = frame.with_columns(seg_start.alias("seg_start")).with_columns(
        ((pl.col("ts") - pl.col("seg_start")).dt.total_minutes()).alias("seg_minute"),
        pl.when(pl.col("seg").is_not_null())
        .then(pl.struct("trading_date", "seg").rank("dense").cast(pl.Int64))
        .alias("seg_id"),
    )

    frame = frame.with_columns(
        pl.when(pl.col("seg").is_not_null())
        .then(pl.col("volume").cum_sum().over("seg_id"))
        .alias("_cumvol")
    )
    frame = _with_prior_rth(frame)

    lo = cast(date, frame["trading_date"].min())
    hi = cast(date, frame["trading_date"].max())
    cal = calendar.frame(lo - timedelta(days=1), hi + timedelta(days=1))
    frame = frame.join(
        cal.select(
            pl.col("date").alias("trading_date"),
            "kind",
            pl.col("rth_close").alias("_rth_close"),
            pl.col("globex_close").alias("_gx_close"),
        ),
        on="trading_date",
        how="left",
    )
    eth_end_t = pl.min_horizontal(pl.lit(s.rth_open), pl.col("_gx_close"))
    has_rth = pl.col("kind").is_in(["normal", "early_close"])

    def end_dt(t: pl.Expr) -> pl.Expr:
        return pl.col("trading_date").dt.combine(t).dt.replace_time_zone(EXCHANGE_TZ)

    step = pl.duration(minutes=bar_minutes)
    specs = {
        "RTH": (pl.col("seg") == RTH, end_dt(pl.col("_rth_close"))),
        "ETH": (pl.col("seg") == ETH, end_dt(eth_end_t)),
        "both": (
            pl.col("seg").is_not_null(),
            pl.when(has_rth).then(end_dt(pl.col("_rth_close"))).otherwise(end_dt(eth_end_t)),
        ),
    }
    windows: dict[str, Window] = {}
    for name, (member, end) in specs.items():
        wid = (
            pl.when(member)
            .then(pl.col("trading_date").cast(pl.Int64) * 4 + {"RTH": 1, "ETH": 2, "both": 3}[name])
            .otherwise(-1)
        )
        flat = member & (pl.col("ts") == (end - step))
        out = frame.select(wid.alias("w"), flat.fill_null(False).alias("f"))
        windows[name] = Window(
            out["w"].to_numpy().astype(np.int64), out["f"].to_numpy().astype(bool)
        )

    vol_b, trend = _regimes(frame, regimes)
    per_bar = frame.select("trading_date").join(
        pl.DataFrame(
            {
                "trading_date": list(vol_b),
                "vb": list(vol_b.values()),
                "tr": [trend[d] for d in vol_b],
            },
            schema={"trading_date": pl.Date, "vb": pl.String, "tr": pl.String},
        ),
        on="trading_date",
        how="left",
    )
    td = frame["trading_date"]
    return MarketContext(
        frame=frame,
        open=frame["open"].to_numpy(),
        high=frame["high"].to_numpy(),
        low=frame["low"].to_numpy(),
        close=frame["close"].to_numpy(),
        session=frame["session"].to_numpy(),
        trading_date=td.to_numpy(),
        windows=windows,
        vol_bucket=per_bar["vb"].fill_null("unknown").to_numpy(),
        trend_state=per_bar["tr"].fill_null("unknown").to_numpy(),
        dow=np.array([DOW[d.weekday()] for d in td.to_list()]),
        event_day=td.is_in(sorted(events)).to_numpy() if events else np.zeros(frame.height, bool),
        flags=frame["quality_flags"].to_numpy().astype(np.uint32),
    )


def _with_prior_rth(frame: pl.DataFrame) -> pl.DataFrame:
    """Previous RTH day's high / low / last close on every bar, via a strict backward as-of join."""
    daily = (
        frame.filter(pl.col("seg") == RTH)
        .group_by("trading_date")
        .agg(
            pl.col("high").max().alias("_prior_high"),
            pl.col("low").min().alias("_prior_low"),
            pl.col("close").last().alias("_prior_close"),
        )
        .sort("trading_date")
    )
    return frame.join_asof(daily, on="trading_date", strategy="backward", allow_exact_matches=False)


def _regimes(frame: pl.DataFrame, cfg: RegimeConfig) -> tuple[dict[date, str], dict[date, str]]:
    """Per trading date: volatility bucket and trend state, both from strictly earlier RTH days."""
    rth = frame.filter(pl.col("session") == RTH)
    daily = (
        rth.with_columns(pl.col("close").log().diff().over("trading_date").alias("r"))
        .group_by("trading_date")
        .agg(pl.col("r").std().alias("vol"), pl.col("close").last())
        .sort("trading_date")
    )
    rdates: list[date] = daily["trading_date"].to_list()
    vols = daily["vol"].to_numpy()
    closes = daily["close"].to_numpy()
    all_dates: list[date] = frame["trading_date"].unique().sort().to_list()
    vb: dict[date, str] = {}
    tr: dict[date, str] = {}
    k = 0  # number of RTH dates strictly before the current date
    L, T = cfg.vol_lookback_days, cfg.trend_days  # noqa: N806
    for d in all_dates:
        while k < len(rdates) and rdates[k] < d:
            k += 1
        if k >= L + 1 and not np.isnan(vols[k - 1]):
            hist = vols[k - 1 - L : k - 1]
            q = float(np.mean(hist < vols[k - 1]))
            vb[d] = VOL_BUCKETS[0] if q < 1 / 3 else VOL_BUCKETS[1] if q < 2 / 3 else VOL_BUCKETS[2]
        else:
            vb[d] = "unknown"
        if k >= T:
            tr[d] = "up" if closes[k - 1] > float(np.mean(closes[k - T : k])) else "down"
        else:
            tr[d] = "unknown"
    return vb, tr


# ---- features ------------------------------------------------------------------------------


def compute_feature(frame: pl.DataFrame, kind: str, args: tuple[int, ...]) -> np.ndarray:
    """One feature as float64 (NaN where undefined). `frame` must come from build_context."""
    expr = _EXPRS[kind](*args)
    out = frame.select(expr.cast(pl.Float64).fill_nan(None).alias("v"))["v"]
    return out.to_numpy().astype(np.float64)


def _log_ret() -> pl.Expr:
    return (pl.col("close") / pl.col("close").shift(1)).log()


def _returns(bars: int) -> pl.Expr:
    return (pl.col("close") / pl.col("close").shift(bars)).log()


def _atr(bars: int) -> pl.Expr:
    pc = pl.col("close").shift(1)
    tr = pl.max_horizontal(
        pl.col("high") - pl.col("low"), (pl.col("high") - pc).abs(), (pl.col("low") - pc).abs()
    )
    tr = pl.when(pc.is_null()).then(pl.col("high") - pl.col("low")).otherwise(tr)
    return tr.ewm_mean(alpha=1.0 / bars, adjust=False, min_samples=bars)


def _realized_vol(bars: int) -> pl.Expr:
    return _log_ret().rolling_std(window_size=bars, min_samples=bars)


def _close_zscore(bars: int) -> pl.Expr:
    c = pl.col("close")
    sd = c.rolling_std(window_size=bars, min_samples=bars)
    return pl.when(sd > 0).then((c - c.rolling_mean(window_size=bars, min_samples=bars)) / sd)


def _vwap_distance() -> pl.Expr:
    tp = (pl.col("high") + pl.col("low") + pl.col("close")) / 3
    cum_pv = (tp * pl.col("volume")).cum_sum().over("seg_id")
    cum_v = pl.col("volume").cum_sum().over("seg_id")
    in_seg = pl.col("seg_id").is_not_null() & (cum_v > 0)
    return pl.when(in_seg).then(pl.col("close") - cum_pv / cum_v)


def _minutes_into_session() -> pl.Expr:
    return pl.when(pl.col("seg").is_not_null()).then(pl.col("seg_minute"))


def _opening_range(minutes: int, high: bool) -> pl.Expr:
    in_range = (pl.col("seg") == RTH) & (pl.col("seg_minute") < minutes)
    src = pl.when(in_range).then(pl.col("high" if high else "low"))
    agg = src.max() if high else src.min()
    level = agg.over("trading_date")
    # Available from the close of the range's last bar; the aggregate only covers bars already seen.
    ready = (pl.col("seg") == RTH) & (pl.col("seg_minute") >= minutes - 1)
    return pl.when(ready).then(level)


def _overnight(high: bool) -> pl.Expr:
    src = pl.when(pl.col("seg") == ETH).then(pl.col("high" if high else "low"))
    level = (src.max() if high else src.min()).over("trading_date")
    return pl.when(pl.col("seg") == RTH).then(level)


def _overnight_gap() -> pl.Expr:
    first_open = (
        pl.when(pl.col("seg") == RTH).then(pl.col("open")).drop_nulls().first().over("trading_date")
    )
    return pl.when(pl.col("seg") == RTH).then(first_open - pl.col("_prior_close"))


def _session_volume_ratio(days: int) -> pl.Expr:
    # Cumulative session volume against the mean at the same minute over the previous `days`
    # instances of the same session type (the current instance is excluded by the shift).
    base = (
        pl.col("_cumvol")
        .rolling_mean(window_size=days, min_samples=days)
        .shift(1)
        .over(["seg", "seg_minute"])
    )
    return pl.when(pl.col("seg").is_not_null() & (base > 0)).then(pl.col("_cumvol") / base)


_EXPRS: dict[str, Callable[..., pl.Expr]] = {
    "returns": _returns,
    "atr": _atr,
    "vwap_distance": _vwap_distance,
    "opening_range_high": lambda m: _opening_range(m, True),
    "opening_range_low": lambda m: _opening_range(m, False),
    "overnight_high": lambda: _overnight(True),
    "overnight_low": lambda: _overnight(False),
    "overnight_gap": _overnight_gap,
    "session_volume_ratio": _session_volume_ratio,
    "realized_vol": _realized_vol,
    "prior_day_high": lambda: pl.col("_prior_high"),
    "prior_day_low": lambda: pl.col("_prior_low"),
    "prior_day_close": lambda: pl.col("_prior_close"),
    "close_zscore": _close_zscore,
    "minutes_into_session": _minutes_into_session,
}
