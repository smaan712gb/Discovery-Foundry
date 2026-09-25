"""Raw bar loaders. Output: one canonical frame keyed by bar START time in America/New_York.

Canonical raw-bar schema: ts, contract, open, high, low, close, volume, source_file.
Timezone and bar-label conventions come from each source's config (ADR 0002).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import polars as pl

from foundry.core.hashing import sha256_file
from foundry.domains.nq.config import EMBEDDED_TZ, EXCHANGE_TZ, SourceConfig

MONTH_CODES = {
    1: "F",
    2: "G",
    3: "H",
    4: "J",
    5: "K",
    6: "M",
    7: "N",
    8: "Q",
    9: "U",
    10: "V",
    11: "X",
    12: "Z",
}
CONTINUOUS_CONTRACT = "CONT"

TS_DTYPE = pl.Datetime("us", EXCHANGE_TZ)
RAW_SCHEMA: dict[str, pl.DataType] = {
    "ts": TS_DTYPE,
    "contract": pl.String(),
    "open": pl.Float64(),
    "high": pl.Float64(),
    "low": pl.Float64(),
    "close": pl.Float64(),
    "volume": pl.Int64(),
    "source_file": pl.String(),
}
ISSUE_SCHEMA: dict[str, pl.DataType] = {
    "ts": TS_DTYPE,
    "contract": pl.String(),
    "check": pl.String(),
    "severity": pl.String(),
    "detail": pl.String(),
}


class LoadError(Exception):
    pass


@dataclass(frozen=True)
class ContractId:
    root: str
    year: int
    month: int

    @property
    def code(self) -> str:
        return f"{self.root}{MONTH_CODES[self.month]}{self.year % 100:02d}"

    @property
    def sort_key(self) -> int:
        return self.year * 100 + self.month


def parse_contract_code(code: str) -> ContractId | None:
    """Inverse of ContractId.code, e.g. 'NQH25' -> NQ 2025-03. None for CONT."""
    m = re.fullmatch(r"(?P<root>[A-Z]+)(?P<m>[FGHJKMNQUVXZ])(?P<y>\d{2})", code)
    if m is None:
        return None
    month = {v: k for k, v in MONTH_CODES.items()}[m["m"]]
    return ContractId(m["root"], 2000 + int(m["y"]), month)


@dataclass(frozen=True)
class RawFile:
    path: Path
    source: str
    sha256: str
    rows: int
    contract: str | None


@dataclass(frozen=True)
class LoadResult:
    bars: pl.DataFrame
    issues: pl.DataFrame
    files: list[RawFile]


def empty_issues() -> pl.DataFrame:
    return pl.DataFrame(schema=ISSUE_SCHEMA)


def discover_files(src: SourceConfig) -> list[tuple[Path, re.Match[str]]]:
    if not src.directory.is_dir():
        raise LoadError(f"source {src.name}: directory not found: {src.directory}")
    rx = re.compile(src.filename_regex)
    found = []
    for p in sorted(src.directory.iterdir()):
        m = rx.match(p.name)
        if m and p.is_file():
            found.append((p, m))
    if not found:
        raise LoadError(
            f"source {src.name}: no files match {src.filename_regex!r} in {src.directory}"
        )
    return found


def contract_from_match(root: str, m: re.Match[str]) -> ContractId:
    month = int(m["month"])
    year = int(m["year"])
    if year < 100:
        year += 2000
    if not 1 <= month <= 12:
        raise LoadError(f"bad contract month {month} in {m.string!r}")
    return ContractId(root, year, month)


# ---- timestamp handling -------------------------------------------------------------------


def localize_to_exchange(
    frame: pl.DataFrame, naive_col: str, tz: str, label: str, bar_minutes: int
) -> tuple[pl.DataFrame, int]:
    """Turn a naive wall-clock column in `tz` into tz-aware ET bar starts.

    Ambiguous wall times (DST fall-back) resolve by order of appearance within each contract:
    the first occurrence is the earlier instant. Non-existent wall times (spring-forward gap)
    become null and are dropped; the count is returned so the caller can report it.
    """
    occ = pl.col(naive_col).cum_count().over(["contract", naive_col]) - 1
    ambiguous = pl.when(occ == 0).then(pl.lit("earliest")).otherwise(pl.lit("latest"))
    out = frame.with_columns(
        pl.col(naive_col)
        .dt.replace_time_zone(tz, ambiguous=ambiguous, non_existent="null")
        .alias("_aware")
    )
    nonexistent = out["_aware"].null_count() - frame[naive_col].null_count()
    out = out.filter(pl.col("_aware").is_not_null())
    return _finish_ts(out, "_aware", label, bar_minutes).drop(naive_col), nonexistent


def _finish_ts(frame: pl.DataFrame, aware_col: str, label: str, bar_minutes: int) -> pl.DataFrame:
    utc = pl.col(aware_col).dt.convert_time_zone("UTC").dt.cast_time_unit("us")
    if label == "end":
        utc = utc - pl.duration(minutes=bar_minutes)
    return frame.with_columns(utc.dt.convert_time_zone(EXCHANGE_TZ).alias("ts")).drop(aware_col)


# ---- readers ------------------------------------------------------------------------------


def _read_ninjatrader(path: Path) -> pl.DataFrame:
    frame = pl.read_csv(
        path,
        separator=";",
        has_header=False,
        new_columns=["stamp", "open", "high", "low", "close", "volume"],
        schema_overrides={
            "stamp": pl.String,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        },
    )
    return frame.with_columns(
        pl.col("stamp")
        .str.strptime(pl.Datetime("us"), "%Y%m%d %H%M%S", strict=True)
        .alias("_naive")
    ).drop("stamp")


def _read_generic(path: Path, src: SourceConfig) -> pl.DataFrame:
    assert src.columns is not None
    c = src.columns
    if src.format == "csv":
        frame = pl.read_csv(path, separator=src.csv_separator, infer_schema_length=10000)
    else:
        frame = pl.read_parquet(path)
    mapping = {
        c.timestamp: "_stamp",
        c.open: "open",
        c.high: "high",
        c.low: "low",
        c.close: "close",
        c.volume: "volume",
    }
    if c.contract:
        mapping[c.contract] = "contract"
    missing = [k for k in mapping if k not in frame.columns]
    if missing:
        raise LoadError(f"{path.name}: missing columns {missing}")
    frame = frame.select(list(mapping)).rename(mapping)
    frame = frame.with_columns(
        [pl.col(k).cast(pl.Float64) for k in ("open", "high", "low", "close", "volume")]
    )
    stamp = pl.col("_stamp")
    dtype = frame.schema["_stamp"]
    # Offset-bearing strings parse to UTC instants; naive ones stay naive for localisation.
    parse_tz = "UTC" if src.timezone == EMBEDDED_TZ else None
    if src.timestamp_format in ("unix_s", "unix_ms"):
        if src.timezone != "UTC":
            raise LoadError(f"{path.name}: unix timestamps are UTC; set timezone: UTC")
        unit: Literal["s", "ms"] = "s" if src.timestamp_format == "unix_s" else "ms"
        # Kept naive here; localize_to_exchange then applies the declared UTC zone.
        frame = frame.with_columns(
            pl.from_epoch(stamp.cast(pl.Int64), time_unit=unit).alias("_stamp")
        )
    elif isinstance(dtype, pl.Datetime):
        pass
    elif src.timestamp_format:
        frame = frame.with_columns(
            stamp.str.strptime(pl.Datetime("us", parse_tz), src.timestamp_format)
        )
    else:
        frame = frame.with_columns(stamp.str.to_datetime(time_unit="us", time_zone=parse_tz))
    return frame


def load_source(src: SourceConfig, root: str, bar_minutes: int) -> LoadResult:
    frames: list[pl.DataFrame] = []
    issues: list[pl.DataFrame] = []
    files: list[RawFile] = []
    for path, match in discover_files(src):
        contract: str | None = None
        if src.contract_from == "filename":
            contract = contract_from_match(root, match).code
        elif src.contract_from == "continuous":
            contract = CONTINUOUS_CONTRACT
        try:
            raw = (
                _read_ninjatrader(path)
                if src.format == "ninjatrader_txt"
                else _read_generic(path, src)
            )
        except (pl.exceptions.PolarsError, OSError) as exc:
            raise LoadError(f"{path}: {exc}") from exc
        if contract is not None:
            raw = raw.with_columns(pl.lit(contract).alias("contract"))
        raw = raw.with_columns(pl.col("contract").cast(pl.String))

        if "_naive" in raw.columns:
            raw, nonexistent = localize_to_exchange(
                raw, "_naive", src.timezone, src.timestamp_label, bar_minutes
            )
        else:
            dtype = raw.schema["_stamp"]
            has_tz = isinstance(dtype, pl.Datetime) and dtype.time_zone is not None
            if has_tz and src.timezone != EMBEDDED_TZ:
                raise LoadError(f"{path.name}: timestamps carry an offset; set timezone: embedded")
            if not has_tz and src.timezone == EMBEDDED_TZ:
                raise LoadError(f"{path.name}: timezone is 'embedded' but timestamps are naive")
            if has_tz:
                nonexistent = 0
                raw = _finish_ts(raw, "_stamp", src.timestamp_label, bar_minutes)
            else:
                raw = raw.with_columns(pl.col("_stamp").dt.cast_time_unit("us"))
                raw, nonexistent = localize_to_exchange(
                    raw, "_stamp", src.timezone, src.timestamp_label, bar_minutes
                )
        if nonexistent:
            issues.append(
                pl.DataFrame(
                    {
                        "ts": [None],
                        "contract": [contract or "?"],
                        "check": ["nonexistent_local_time"],
                        "severity": ["warning"],
                        "detail": [
                            f"{path.name}: {nonexistent} wall-clock times in a DST gap dropped"
                        ],
                    },
                    schema=ISSUE_SCHEMA,
                )
            )

        fractional = raw.filter(pl.col("volume") != pl.col("volume").round(0)).height
        if fractional:
            raise LoadError(f"{path.name}: {fractional} rows with non-integer volume")
        raw = raw.with_columns(
            pl.col("volume").cast(pl.Int64), pl.lit(path.name).alias("source_file")
        ).select(list(RAW_SCHEMA))
        frames.append(raw)
        files.append(
            RawFile(
                path=path,
                source=src.name,
                sha256=sha256_file(path),
                rows=raw.height,
                contract=contract,
            )
        )
    bars = pl.concat(frames) if frames else pl.DataFrame(schema=RAW_SCHEMA)
    return LoadResult(
        bars=bars, issues=pl.concat(issues) if issues else empty_issues(), files=files
    )


def load_all(sources: list[SourceConfig], root: str, bar_minutes: int) -> LoadResult:
    results = [load_source(s, root, bar_minutes) for s in sources]
    bars = pl.concat([r.bars for r in results]).cast(RAW_SCHEMA)  # type: ignore[arg-type]
    issues = pl.concat([r.issues for r in results])
    bars, dup_issues = resolve_duplicates(bars)
    files = [f for r in results for f in r.files]
    return LoadResult(bars=bars, issues=pl.concat([issues, dup_issues]), files=files)


def resolve_duplicates(bars: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Collapse exact duplicate bars; report conflicting duplicates as fatal.

    Two rows are exact duplicates when ts, contract and OHLCV all match, even if they came from
    different source files.
    """
    value_cols = ["ts", "contract", "open", "high", "low", "close", "volume"]
    before = bars.height
    deduped = bars.unique(subset=value_cols, keep="first", maintain_order=True)
    exact = before - deduped.height
    conflicts = (
        deduped.group_by(["contract", "ts"])
        .len()
        .filter(pl.col("len") > 1)
        .sort(["contract", "ts"])
    )
    issue_frames = []
    if exact:
        issue_frames.append(
            pl.DataFrame(
                {
                    "ts": [None],
                    "contract": ["*"],
                    "check": ["duplicate_exact"],
                    "severity": ["info"],
                    "detail": [f"{exact} exact duplicate bars collapsed"],
                },
                schema=ISSUE_SCHEMA,
            )
        )
    if conflicts.height:
        issue_frames.append(
            conflicts.select(
                "ts",
                "contract",
                pl.lit("duplicate_conflict").alias("check"),
                pl.lit("fatal").alias("severity"),
                pl.format(
                    "{} rows with the same timestamp but different values", pl.col("len")
                ).alias("detail"),
            )
        )
    issues = pl.concat(issue_frames) if issue_frames else empty_issues()
    return deduped.sort(["contract", "ts"]), issues
