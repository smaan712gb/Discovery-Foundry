"""DuckDB results store: specs, evaluations, trades, daily PnL.

The store is the source of the true trial count used by the deflated Sharpe ratio (phase 3):
every distinct spec ever evaluated on the search split counts, including ones that died. Rows are
only ever added, never updated or deleted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS specs (
    spec_hash VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    spec_json VARCHAR NOT NULL,
    generator VARCHAR,
    parent_hashes VARCHAR,
    engine_id VARCHAR,
    first_seen TIMESTAMP NOT NULL
);
CREATE TABLE IF NOT EXISTS evaluations (
    eval_key VARCHAR PRIMARY KEY,
    spec_hash VARCHAR NOT NULL,
    split VARCHAR NOT NULL,
    contract VARCHAR NOT NULL,
    data_hash VARCHAR NOT NULL,
    eval_config_hash VARCHAR NOT NULL,
    net_pnl DOUBLE NOT NULL,
    trade_count INTEGER NOT NULL,
    sharpe DOUBLE NOT NULL,
    metrics_json VARCHAR NOT NULL,
    breakdown_json VARCHAR NOT NULL,
    run_json VARCHAR NOT NULL,
    created_at TIMESTAMP NOT NULL
);
CREATE TABLE IF NOT EXISTS trades (
    eval_key VARCHAR NOT NULL,
    n INTEGER NOT NULL,
    entry_ts TIMESTAMPTZ NOT NULL,
    exit_ts TIMESTAMPTZ NOT NULL,
    direction INTEGER NOT NULL,
    entry_price DOUBLE NOT NULL,
    exit_price DOUBLE NOT NULL,
    gross_pnl DOUBLE NOT NULL,
    net_pnl DOUBLE NOT NULL,
    reason VARCHAR NOT NULL,
    entry_session VARCHAR NOT NULL,
    vol_bucket VARCHAR NOT NULL,
    entry_trading_date DATE NOT NULL,
    exit_trading_date DATE NOT NULL
);
CREATE TABLE IF NOT EXISTS critic_results (
    spec_hash VARCHAR NOT NULL,
    stage VARCHAR NOT NULL,
    test VARCHAR NOT NULL,
    passed BOOLEAN NOT NULL,
    numbers_json VARCHAR NOT NULL,
    created_at TIMESTAMP NOT NULL
);
CREATE TABLE IF NOT EXISTS verdicts (
    spec_hash VARCHAR NOT NULL,
    alive BOOLEAN NOT NULL,
    killed_stage VARCHAR,
    killed_by VARCHAR,
    labels VARCHAR NOT NULL,
    created_at TIMESTAMP NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id VARCHAR PRIMARY KEY,
    engine_id VARCHAR NOT NULL,
    seed BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    summary_json VARCHAR NOT NULL,
    started_at TIMESTAMP NOT NULL,
    ended_at TIMESTAMP
);
CREATE TABLE IF NOT EXISTS llm_calls (
    run_id VARCHAR NOT NULL,
    generation INTEGER NOT NULL,
    model VARCHAR NOT NULL,
    prompt_sha256 VARCHAR NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    dollars DOUBLE NOT NULL,
    returned INTEGER NOT NULL,
    valid INTEGER NOT NULL,
    error VARCHAR,
    created_at TIMESTAMP NOT NULL
);
CREATE TABLE IF NOT EXISTS daily_pnl (
    eval_key VARCHAR NOT NULL,
    trading_date DATE NOT NULL,
    net_pnl DOUBLE NOT NULL
);
"""

TRADE_COLUMNS = (
    "n",
    "entry_ts",
    "exit_ts",
    "direction",
    "entry_price",
    "exit_price",
    "gross_pnl",
    "net_pnl",
    "reason",
    "entry_session",
    "vol_bucket",
    "entry_trading_date",
    "exit_trading_date",
)


@dataclass(frozen=True)
class StoredEvaluation:
    eval_key: str
    spec_hash: str
    split: str
    contract: str
    metrics: dict[str, Any]
    breakdown: dict[str, Any]


class ResultsStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._con = duckdb.connect(str(path))
        self._con.execute(SCHEMA_SQL)

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> ResultsStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def record_spec(
        self,
        spec_hash: str,
        name: str,
        spec_json: str,
        generator: str | None = None,
        parent_hashes: list[str] | None = None,
        engine_id: str | None = None,
    ) -> None:
        self._con.execute(
            "INSERT INTO specs VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT (spec_hash) DO NOTHING",
            [
                spec_hash,
                name,
                spec_json,
                generator,
                json.dumps(parent_hashes) if parent_hashes is not None else None,
                engine_id,
                datetime.now(UTC).replace(tzinfo=None),
            ],
        )

    def get(self, eval_key: str) -> StoredEvaluation | None:
        row = self._con.execute(
            "SELECT eval_key, spec_hash, split, contract, metrics_json, breakdown_json "
            "FROM evaluations WHERE eval_key = ?",
            [eval_key],
        ).fetchone()
        if row is None:
            return None
        return StoredEvaluation(
            row[0], row[1], row[2], row[3], json.loads(row[4]), json.loads(row[5])
        )

    def record_evaluation(
        self,
        eval_key: str,
        spec_hash: str,
        split: str,
        contract: str,
        data_hash: str,
        eval_config_hash: str,
        metrics: dict[str, Any],
        breakdown: dict[str, Any],
        run: dict[str, Any],
        trades: pl.DataFrame,
        daily: pl.DataFrame,
    ) -> bool:
        """Insert an evaluation with its trades and daily PnL; False if it already exists."""
        if self.get(eval_key) is not None:
            return False
        self._con.execute("BEGIN")
        try:
            self._con.execute(
                "INSERT INTO evaluations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    eval_key,
                    spec_hash,
                    split,
                    contract,
                    data_hash,
                    eval_config_hash,
                    metrics["net_pnl"],
                    metrics["trade_count"],
                    metrics["sharpe"],
                    json.dumps(metrics, sort_keys=True),
                    json.dumps(breakdown, sort_keys=True),
                    json.dumps(run, sort_keys=True),
                    datetime.now(UTC).replace(tzinfo=None),
                ],
            )
            t = trades.select(list(TRADE_COLUMNS)).with_columns(pl.lit(eval_key).alias("eval_key"))
            self._con.register("_t", t.select("eval_key", *TRADE_COLUMNS).to_arrow())
            self._con.execute("INSERT INTO trades SELECT * FROM _t")
            self._con.unregister("_t")
            d = daily.select(pl.lit(eval_key).alias("eval_key"), "trading_date", "net_pnl")
            self._con.register("_d", d.to_arrow())
            self._con.execute("INSERT INTO daily_pnl SELECT * FROM _d")
            self._con.unregister("_d")
            self._con.execute("COMMIT")
        except Exception:
            self._con.execute("ROLLBACK")
            raise
        return True

    def trial_count(self, split: str = "search") -> int:
        """Distinct specs ever evaluated on a split: the N in the deflated Sharpe ratio."""
        row = self._con.execute(
            "SELECT COUNT(DISTINCT spec_hash) FROM evaluations WHERE split = ?", [split]
        ).fetchone()
        return int(row[0]) if row else 0

    def trial_sharpes(self, split: str, contract: str, eval_config_hash: str) -> dict[str, float]:
        """Annualized Sharpe per distinct spec on a split, base cost settings only (ADR 0008)."""
        rows = self._con.execute(
            "SELECT spec_hash, avg(sharpe) FROM evaluations WHERE split = ? AND contract = ? "
            "AND eval_config_hash = ? GROUP BY spec_hash ORDER BY spec_hash",
            [split, contract, eval_config_hash],
        ).fetchall()
        return {str(r[0]): float(r[1]) for r in rows}

    def trial_daily_matrix(
        self, split: str, contract: str, eval_config_hash: str
    ) -> tuple[list[str], pl.DataFrame]:
        """Daily PnL of every trial on a split as one column per spec (days x specs, 0-filled)."""
        long = self._con.execute(
            "SELECT e.spec_hash, d.trading_date, d.net_pnl FROM daily_pnl d "
            "JOIN evaluations e USING (eval_key) WHERE e.split = ? AND e.contract = ? "
            "AND e.eval_config_hash = ?",
            [split, contract, eval_config_hash],
        ).pl()
        if long.is_empty():
            return [], pl.DataFrame()
        wide = (
            long.pivot(
                on="spec_hash", index="trading_date", values="net_pnl", aggregate_function="mean"
            )
            .sort("trading_date")
            .fill_null(0.0)
        )
        specs = sorted(c for c in wide.columns if c != "trading_date")
        return specs, wide.select("trading_date", *specs)

    def record_critic_result(
        self, spec_hash: str, stage: str, test: str, passed: bool, numbers: dict[str, Any]
    ) -> None:
        self._con.execute(
            "INSERT INTO critic_results VALUES (?, ?, ?, ?, ?, ?)",
            [
                spec_hash,
                stage,
                test,
                passed,
                json.dumps(numbers, sort_keys=True),
                datetime.now(UTC).replace(tzinfo=None),
            ],
        )

    def record_verdict(
        self,
        spec_hash: str,
        alive: bool,
        killed_stage: str | None,
        killed_by: str | None,
        labels: list[str],
    ) -> None:
        self._con.execute(
            "INSERT INTO verdicts VALUES (?, ?, ?, ?, ?, ?)",
            [
                spec_hash,
                alive,
                killed_stage,
                killed_by,
                json.dumps(sorted(labels)),
                datetime.now(UTC).replace(tzinfo=None),
            ],
        )

    def funnel(self) -> dict[str, Any]:
        """generated -> evaluated on search -> killed by each test -> alive (latest verdicts)."""
        con = self._con
        generated = con.execute("SELECT COUNT(*) FROM specs").fetchone()
        evaluated = con.execute(
            "SELECT COUNT(DISTINCT spec_hash) FROM evaluations WHERE split = 'search'"
        ).fetchone()
        latest = (
            "SELECT * FROM verdicts QUALIFY row_number() OVER "
            "(PARTITION BY spec_hash ORDER BY created_at DESC) = 1"
        )
        kills = con.execute(
            f"SELECT killed_stage, killed_by, COUNT(*) FROM ({latest}) WHERE NOT alive "
            "GROUP BY killed_stage, killed_by ORDER BY killed_stage, killed_by"
        ).fetchall()
        alive = con.execute(f"SELECT COUNT(*) FROM ({latest}) WHERE alive").fetchone()
        critiqued = con.execute(f"SELECT COUNT(*) FROM ({latest})").fetchone()
        return {
            "generated": int(generated[0]) if generated else 0,
            "evaluated_on_search": int(evaluated[0]) if evaluated else 0,
            "critiqued": int(critiqued[0]) if critiqued else 0,
            "killed": [{"stage": s, "test": t, "count": int(n)} for s, t, n in kills],
            "alive": int(alive[0]) if alive else 0,
        }

    def has_spec_evaluation(self, spec_hash: str, split: str) -> bool:
        row = self._con.execute(
            "SELECT 1 FROM evaluations WHERE spec_hash = ? AND split = ? LIMIT 1",
            [spec_hash, split],
        ).fetchone()
        return row is not None

    def start_run(self, run_id: str, engine_id: str, seed: int, summary: dict[str, Any]) -> None:
        self._con.execute(
            "INSERT INTO runs VALUES (?, ?, ?, 'running', ?, ?, NULL)",
            [
                run_id,
                engine_id,
                seed,
                json.dumps(summary, sort_keys=True),
                datetime.now(UTC).replace(tzinfo=None),
            ],
        )

    def finish_run(self, run_id: str, status: str, summary: dict[str, Any]) -> None:
        self._con.execute(
            "UPDATE runs SET status = ?, summary_json = ?, ended_at = ? WHERE run_id = ?",
            [
                status,
                json.dumps(summary, sort_keys=True, default=str),
                datetime.now(UTC).replace(tzinfo=None),
                run_id,
            ],
        )

    def record_llm_call(
        self,
        run_id: str,
        generation: int,
        model: str,
        prompt_sha256: str,
        prompt_tokens: int,
        completion_tokens: int,
        dollars: float,
        returned: int,
        valid: int,
        error: str | None,
    ) -> None:
        self._con.execute(
            "INSERT INTO llm_calls VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                run_id,
                generation,
                model,
                prompt_sha256,
                prompt_tokens,
                completion_tokens,
                dollars,
                returned,
                valid,
                error,
                datetime.now(UTC).replace(tzinfo=None),
            ],
        )

    def kill_counts(self) -> dict[str, int]:
        """Latest-verdict kill counts as 'stage:test' -> n (aggregate only)."""
        rows = self._con.execute(
            "SELECT killed_stage, killed_by, COUNT(*) FROM (SELECT * FROM verdicts QUALIFY "
            "row_number() OVER (PARTITION BY spec_hash ORDER BY created_at DESC) = 1) "
            "WHERE NOT alive GROUP BY 1, 2"
        ).fetchall()
        return {f"{s}:{t}": int(n) for s, t, n in rows}

    def dead_on_search(self, limit: int) -> list[dict[str, Any]]:
        """Recent search-stage kills with their search metrics (no validation data)."""
        rows = self._con.execute(
            "SELECT s.spec_json, v.killed_by, e.net_pnl, e.trade_count, e.sharpe FROM verdicts v "
            "JOIN specs s USING (spec_hash) JOIN evaluations e ON e.spec_hash = v.spec_hash "
            "AND e.split = 'search' WHERE NOT v.alive AND v.killed_stage = 'search' "
            "ORDER BY v.created_at DESC LIMIT ?",
            [limit],
        ).fetchall()
        return [
            {
                "spec_json": r[0],
                "killed_by": r[1],
                "search_net_pnl": r[2],
                "search_trades": r[3],
                "search_sharpe": r[4],
            }
            for r in rows
        ]

    def best_on_search(self, limit: int) -> list[dict[str, Any]]:
        rows = self._con.execute(
            "SELECT s.spec_json, e.net_pnl, e.trade_count, e.sharpe FROM evaluations e "
            "JOIN specs s USING (spec_hash) WHERE e.split = 'search' "
            "ORDER BY e.sharpe DESC LIMIT ?",
            [limit],
        ).fetchall()
        return [
            {
                "spec_json": r[0],
                "search_net_pnl": r[1],
                "search_trades": r[2],
                "search_sharpe": r[3],
            }
            for r in rows
        ]

    def spec_json(self, spec_hash: str) -> str | None:
        row = self._con.execute(
            "SELECT spec_json FROM specs WHERE spec_hash = ?", [spec_hash]
        ).fetchone()
        return str(row[0]) if row else None

    def daily(self, eval_key: str) -> pl.DataFrame:
        return self._con.execute(
            "SELECT trading_date, net_pnl FROM daily_pnl WHERE eval_key = ? ORDER BY trading_date",
            [eval_key],
        ).pl()

    def trades(self, eval_key: str) -> pl.DataFrame:
        return self._con.execute(
            "SELECT * EXCLUDE (eval_key) FROM trades WHERE eval_key = ? ORDER BY n", [eval_key]
        ).pl()
