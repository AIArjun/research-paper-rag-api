"""
Persistent model-call ledger
============================
A small SQLite ledger that counts every attempted real-model call before the
provider is contacted. Reservations are never released: failed, timed-out or
cancelled calls stay counted. Measured token usage replaces the conservative
reservation only when the provider actually reported it.

Concurrency: every operation opens its own connection and runs inside a
BEGIN IMMEDIATE transaction, so concurrent threads (or processes sharing the
same file) cannot both pass an allowance check for the last remaining call.

Fail closed: a missing directory, unreadable, uninitialized, corrupt or
incompatible database raises LedgerError. Nothing is ever reset silently. An
existing zero-byte file is refused as truncated history rather than treated
as a fresh database; a genuinely new ledger is created only when the path
does not exist, by staging the schema in a private file and publishing it
under the final name with an exclusive link.

The ledger bounds the number of attempted calls and charged tokens. It is not
a dollar cap: money follows the provider's price list, which this file does
not know. It also needs durable storage; an ephemeral filesystem can lose it.
"""

import os
import secrets
import sqlite3
from contextlib import closing, suppress
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

from app.config import BudgetAllowances

SCHEMA_VERSION = 1
_CALL_COLUMNS = (
    "id", "day", "started_at", "finished_at", "status", "provider", "model",
    "tokens_reserved", "tokens_measured", "request_id",
)
_META_COLUMNS = ("key", "value")
# Charged tokens per row: the measured total when the provider reported one,
# otherwise the reservation. A zero measurement is treated as unreported.
_CHARGED = "COALESCE(NULLIF(tokens_measured, 0), tokens_reserved)"
_MAX_STORED_TEXT = 128


class LedgerError(RuntimeError):
    """The ledger is unavailable, uninitialized, corrupt or incompatible."""


class BudgetExhaustedError(RuntimeError):
    """An allowance would be exceeded; no provider call may be attempted."""

    def __init__(self, scope: str, kind: str, retry_after: Optional[int] = None):
        super().__init__(f"The {scope} {kind} allowance is exhausted.")
        self.scope = scope  # "daily" or "total"
        self.kind = kind  # "calls" or "tokens"
        self.retry_after = retry_after


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _seconds_until_next_utc_day(now: datetime) -> int:
    next_day = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(1, int((next_day - now).total_seconds()) + 1)


def _bounded_text(value: object) -> str:
    return value[:_MAX_STORED_TEXT] if type(value) is str else ""


class ModelCallLedger:
    """Daily and lifetime call/token allowances backed by one SQLite file."""

    def __init__(
        self,
        path: str,
        allowances: BudgetAllowances,
        clock: Optional[Callable[[], datetime]] = None,
        busy_timeout: float = 5.0,
    ):
        if not isinstance(path, str) or not path.strip():
            raise LedgerError("The ledger path is not configured.")
        self._path = path
        self._allowances = allowances
        self._clock = clock or _utc_now
        self._busy_timeout = busy_timeout
        self._open_or_initialize()

    @property
    def allowances(self) -> BudgetAllowances:
        return self._allowances

    # ─── Connection and schema ───

    def _connect(self, mode: str = "rw") -> sqlite3.Connection:
        """Open with an explicit URI mode so a vanished file is an error, never a fresh ledger.

        isolation_level=None disables implicit transactions; every write below
        opens an explicit BEGIN IMMEDIATE so the check and insert are atomic.
        """
        uri = Path(self._path).resolve().as_uri() + "?mode=" + mode
        return sqlite3.connect(uri, uri=True, timeout=self._busy_timeout, isolation_level=None)

    def _open_or_initialize(self) -> None:
        location = Path(self._path)
        if not location.parent.is_dir():
            raise LedgerError("The ledger directory does not exist.")
        if location.exists():
            if not location.is_file():
                raise LedgerError("The ledger path is not a regular file.")
            if location.stat().st_size == 0:
                # Truncated history, or a crash before the first commit: never
                # reinitialize in place, because that would replenish the budget.
                raise LedgerError("The ledger file exists but is empty; refusing to reinitialize it.")
            self._verify_existing()
            return
        if not self._create_new_database():
            # Another process published the file first; verify what it wrote.
            self._verify_existing()

    def _verify_existing(self) -> None:
        try:
            with closing(self._connect("rw")) as connection:
                self._verify_schema(connection)
        except sqlite3.Error as error:
            raise LedgerError("The ledger could not be opened.") from error

    def _create_new_database(self) -> bool:
        """Deliberate first initialization of a path that does not exist yet.

        The schema is built in a private staging file and then published under
        the final name with an exclusive hard link, so the ledger path never
        exists without a complete schema and two processes cannot both create
        it. Returns False when the final path appeared in the meantime.
        """
        staging = Path(f"{self._path}.init-{os.getpid()}-{secrets.token_hex(4)}")
        try:
            try:
                with closing(sqlite3.connect(str(staging), timeout=self._busy_timeout,
                                             isolation_level=None)) as connection:
                    self._initialize_schema(connection)
            except sqlite3.Error as error:
                raise LedgerError("The ledger could not be created.") from error
            try:
                os.link(staging, self._path)
            except FileExistsError:
                return False
            except OSError as error:
                raise LedgerError("The ledger could not be published at its path.") from error
            return True
        finally:
            with suppress(FileNotFoundError):
                staging.unlink()

    def _initialize_schema(self, connection: sqlite3.Connection) -> None:
        connection.execute("BEGIN IMMEDIATE")
        try:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version != 0:
                raise LedgerError("The ledger file is not empty.")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS ledger_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS model_calls ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, day TEXT NOT NULL, started_at TEXT NOT NULL, "
                "finished_at TEXT, status TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL, "
                "tokens_reserved INTEGER NOT NULL, tokens_measured INTEGER, request_id TEXT)"
            )
            connection.execute("CREATE INDEX IF NOT EXISTS model_calls_day ON model_calls (day)")
            connection.execute(
                "INSERT OR IGNORE INTO ledger_meta (key, value) VALUES (?, ?), (?, ?)",
                ("created_at", self._clock().isoformat(), "schema_version", str(SCHEMA_VERSION)),
            )
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise

    def _verify_schema(self, connection: sqlite3.Connection) -> None:
        integrity = connection.execute("PRAGMA quick_check").fetchone()
        if integrity is None or integrity[0] != "ok":
            raise LedgerError("The ledger failed its integrity check.")
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version != SCHEMA_VERSION:
            raise LedgerError("The ledger schema version is not supported.")
        for table, expected in (("model_calls", _CALL_COLUMNS), ("ledger_meta", _META_COLUMNS)):
            columns = tuple(row[1] for row in connection.execute(f"PRAGMA table_info({table})"))
            if columns != expected:
                raise LedgerError("The ledger schema does not match.")

    # ─── Accounting ───

    def _usage(self, connection: sqlite3.Connection, day: Optional[str] = None) -> tuple[int, int]:
        query = f"SELECT COUNT(*), COALESCE(SUM({_CHARGED}), 0) FROM model_calls"
        if day is None:
            row = connection.execute(query).fetchone()
        else:
            row = connection.execute(query + " WHERE day = ?", (day,)).fetchone()
        return int(row[0]), int(row[1])

    def reserve(self, provider: str, model: str, tokens_reserved: int, request_id: Optional[str] = None) -> int:
        """Atomically count one attempted call, or raise before any provider work.

        The reservation is charged immediately and remains charged whatever the
        provider does afterwards. Returns the ledger row id for settle().
        """
        if type(tokens_reserved) is not int or tokens_reserved < 0:
            raise LedgerError("Token reservations must be non-negative integers.")
        now = self._clock()
        day = now.strftime("%Y-%m-%d")
        allowances = self._allowances
        try:
            with closing(self._connect()) as connection:
                connection.execute("BEGIN IMMEDIATE")
                try:
                    total_calls, total_tokens = self._usage(connection)
                    day_calls, day_tokens = self._usage(connection, day)
                    retry_after = _seconds_until_next_utc_day(now)
                    if total_calls + 1 > allowances.calls_total:
                        raise BudgetExhaustedError("total", "calls")
                    if day_calls + 1 > allowances.calls_per_day:
                        raise BudgetExhaustedError("daily", "calls", retry_after)
                    if total_tokens + tokens_reserved > allowances.tokens_total:
                        raise BudgetExhaustedError("total", "tokens")
                    if day_tokens + tokens_reserved > allowances.tokens_per_day:
                        raise BudgetExhaustedError("daily", "tokens", retry_after)
                    cursor = connection.execute(
                        "INSERT INTO model_calls (day, started_at, status, provider, model, "
                        "tokens_reserved, request_id) VALUES (?, ?, 'reserved', ?, ?, ?, ?)",
                        (day, now.isoformat(), _bounded_text(provider), _bounded_text(model),
                         tokens_reserved, _bounded_text(request_id)),
                    )
                    connection.execute("COMMIT")
                    return int(cursor.lastrowid)
                except BaseException:
                    connection.execute("ROLLBACK")
                    raise
        except sqlite3.Error as error:
            raise LedgerError("The ledger could not record the call.") from error

    def settle(
        self,
        call_id: int,
        status: str,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
    ) -> None:
        """Record the outcome. Measured usage is stored only when both counts are reported."""
        if status not in {"succeeded", "failed"}:
            raise LedgerError("Unknown settlement status.")
        measured = None
        if type(input_tokens) is int and type(output_tokens) is int and input_tokens >= 0 and output_tokens >= 0:
            measured = input_tokens + output_tokens
        try:
            with closing(self._connect()) as connection:
                connection.execute("BEGIN IMMEDIATE")
                try:
                    connection.execute(
                        "UPDATE model_calls SET status = ?, finished_at = ?, tokens_measured = ? "
                        "WHERE id = ? AND status = 'reserved'",
                        (status, self._clock().isoformat(), measured, call_id),
                    )
                    connection.execute("COMMIT")
                except BaseException:
                    connection.execute("ROLLBACK")
                    raise
        except sqlite3.Error as error:
            raise LedgerError("The ledger could not record the outcome.") from error

    def summary(self) -> dict:
        """Cheap counts for readiness; no prompts or identifiers are exposed."""
        day = self._clock().strftime("%Y-%m-%d")
        try:
            with closing(self._connect("ro")) as connection:
                total_calls, total_tokens = self._usage(connection)
                day_calls, day_tokens = self._usage(connection, day)
                measured = connection.execute(
                    "SELECT COALESCE(SUM(tokens_measured), 0), COALESCE(SUM(tokens_reserved), 0), "
                    "SUM(CASE WHEN status = 'reserved' THEN 1 ELSE 0 END) FROM model_calls"
                ).fetchone()
                created = connection.execute(
                    "SELECT value FROM ledger_meta WHERE key = 'created_at'"
                ).fetchone()
        except sqlite3.Error as error:
            raise LedgerError("The ledger could not be read.") from error
        allowances = self._allowances
        return {
            "calls_today": day_calls,
            "calls_total": total_calls,
            "tokens_charged_today": day_tokens,
            "tokens_charged_total": total_tokens,
            "tokens_measured_total": int(measured[0]),
            "tokens_reserved_total": int(measured[1]),
            "calls_unsettled": int(measured[2] or 0),
            "daily_call_allowance": allowances.calls_per_day,
            "total_call_allowance": allowances.calls_total,
            "daily_token_allowance": allowances.tokens_per_day,
            "total_token_allowance": allowances.tokens_total,
            "ledger_created_at": created[0] if created else None,
        }
