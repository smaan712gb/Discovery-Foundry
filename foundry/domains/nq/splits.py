"""Assign each bar to search / validation / holdout by trading date."""

from __future__ import annotations

import polars as pl

from foundry.domains.nq.config import SplitConfig

SEARCH = "search"
VALIDATION = "validation"
HOLDOUT = "holdout"
UNASSIGNED = "unassigned"
PROCESSED_SPLITS: tuple[str, ...] = (SEARCH, VALIDATION)


def assign_splits(series: pl.DataFrame, s: SplitConfig) -> pl.DataFrame:
    td = pl.col("trading_date")
    split = (
        pl.when(td >= pl.lit(s.holdout_start))
        .then(pl.lit(HOLDOUT))
        .when(td.is_between(pl.lit(s.search.start), pl.lit(s.search.end)))
        .then(pl.lit(SEARCH))
        .when(td.is_between(pl.lit(s.validation.start), pl.lit(s.validation.end)))
        .then(pl.lit(VALIDATION))
        .otherwise(pl.lit(UNASSIGNED))
    )
    return series.with_columns(split.alias("split"))
