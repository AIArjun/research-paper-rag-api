"""Persistent model-call ledger: durable counts, atomic allowances, fail-closed opening."""

import os
import sqlite3
import subprocess
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.config import BudgetAllowances
from app.ledger import BudgetExhaustedError, LedgerError, ModelCallLedger


def allowances(**overrides):
    values = {"calls_per_day": 3, "calls_total": 5, "tokens_per_day": 10000, "tokens_total": 20000}
    values.update(overrides)
    return BudgetAllowances(**values)


def test_reservations_survive_a_fresh_connection_and_process(tmp_path):
    path = str(tmp_path / "ledger.sqlite3")
    first = ModelCallLedger(path, allowances())
    first.reserve("openai", "fake-model", 500, "req-1")
    call = first.reserve("openai", "fake-model", 500, "req-2")
    first.settle(call, "succeeded", input_tokens=120, output_tokens=30)
    del first

    reopened = ModelCallLedger(path, allowances())
    summary = reopened.summary()
    assert summary["calls_total"] == 2 and summary["calls_today"] == 2
    # One reservation stays charged at its estimate; the settled call charges what was measured.
    assert summary["tokens_charged_total"] == 500 + 150
    assert summary["tokens_measured_total"] == 150
    assert summary["tokens_reserved_total"] == 1000
    assert summary["calls_unsettled"] == 1

    script = (
        "import sys; from app.config import BudgetAllowances; from app.ledger import ModelCallLedger; "
        f"ledger = ModelCallLedger({path!r}, BudgetAllowances(3, 5, 10000, 20000)); "
        "ledger.reserve('openai', 'fake-model', 100, 'other-process'); print(ledger.summary()['calls_total'])"
    )
    check = subprocess.run(
        [sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1],
        env={**os.environ}, capture_output=True, text=True, timeout=30,
    )
    assert check.returncode == 0, check.stderr
    assert check.stdout.strip() == "3"
    assert reopened.summary()["calls_total"] == 3


def test_concurrent_reservations_never_exceed_the_allowance(tmp_path):
    ledger = ModelCallLedger(str(tmp_path / "ledger.sqlite3"), allowances(calls_per_day=50, calls_total=7))
    start = threading.Barrier(16)
    outcomes = []
    lock = threading.Lock()

    def attempt(index):
        start.wait()
        try:
            ledger.reserve("openai", "fake-model", 10, f"req-{index}")
            outcome = "reserved"
        except BudgetExhaustedError:
            outcome = "refused"
        with lock:
            outcomes.append(outcome)

    threads = [threading.Thread(target=attempt, args=(index,)) for index in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert outcomes.count("reserved") == 7
    assert outcomes.count("refused") == 9
    assert ledger.summary()["calls_total"] == 7


@pytest.mark.parametrize("field,scope,kind", [
    ("calls_per_day", "daily", "calls"), ("calls_total", "total", "calls"),
    ("tokens_per_day", "daily", "tokens"), ("tokens_total", "total", "tokens"),
])
def test_each_allowance_refuses_before_recording_anything(tmp_path, field, scope, kind):
    ledger = ModelCallLedger(str(tmp_path / "ledger.sqlite3"), allowances(**{field: 1 if kind == "calls" else 900}))
    ledger.reserve("openai", "fake-model", 600, "first")
    with pytest.raises(BudgetExhaustedError) as refused:
        ledger.reserve("openai", "fake-model", 600, "second")
    assert refused.value.scope == scope and refused.value.kind == kind
    assert (refused.value.retry_after is not None) is (scope == "daily")
    assert ledger.summary()["calls_total"] == 1


def test_daily_allowance_resets_at_utc_midnight_but_total_does_not(tmp_path):
    now = datetime(2026, 9, 11, 23, 59, 30, tzinfo=timezone.utc)
    ledger = ModelCallLedger(str(tmp_path / "ledger.sqlite3"), allowances(calls_per_day=1, calls_total=2), clock=lambda: now)
    ledger.reserve("openai", "fake-model", 10, "day-one")
    with pytest.raises(BudgetExhaustedError) as refused:
        ledger.reserve("openai", "fake-model", 10, "day-one-again")
    assert refused.value.scope == "daily" and 1 <= refused.value.retry_after <= 31
    now = now + timedelta(minutes=1)
    ledger.reserve("openai", "fake-model", 10, "day-two")
    with pytest.raises(BudgetExhaustedError) as total:
        ledger.reserve("openai", "fake-model", 10, "day-two-again")
    assert total.value.scope == "total" and total.value.retry_after is None


def test_failed_calls_stay_counted_and_zero_measurements_keep_the_reservation(tmp_path):
    ledger = ModelCallLedger(str(tmp_path / "ledger.sqlite3"), allowances())
    failed = ledger.reserve("openai", "fake-model", 700, "failed")
    ledger.settle(failed, "failed")
    zero = ledger.reserve("openai", "fake-model", 300, "zero")
    ledger.settle(zero, "succeeded", input_tokens=0, output_tokens=0)
    summary = ledger.summary()
    assert summary["calls_total"] == 2
    assert summary["tokens_charged_total"] == 1000
    assert summary["calls_unsettled"] == 0
    with pytest.raises(LedgerError):
        ledger.settle(zero, "cancelled")


@pytest.mark.parametrize("corruption", ["garbage", "wrong_version", "wrong_schema", "directory_missing", "not_a_file"])
def test_unusable_ledgers_fail_closed_without_being_reset(tmp_path, corruption):
    path = tmp_path / "ledger.sqlite3"
    if corruption == "garbage":
        path.write_bytes(b"this is not a sqlite database at all" * 40)
    elif corruption == "wrong_version":
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE model_calls (id INTEGER PRIMARY KEY)")
            connection.execute("PRAGMA user_version = 99")
    elif corruption == "wrong_schema":
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE model_calls (id INTEGER PRIMARY KEY, secret TEXT)")
            connection.execute("CREATE TABLE ledger_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            connection.execute("PRAGMA user_version = 1")
    elif corruption == "directory_missing":
        path = tmp_path / "missing-volume" / "ledger.sqlite3"
    else:
        path.mkdir()
    before = path.read_bytes() if path.is_file() else None
    with pytest.raises(LedgerError) as error:
        ModelCallLedger(str(path), allowances())
    assert str(tmp_path) not in str(error.value)
    if before is not None:
        assert path.read_bytes() == before


def test_existing_empty_ledger_file_is_refused_not_reinitialized(tmp_path):
    path = tmp_path / "ledger.sqlite3"
    path.write_bytes(b"")
    with pytest.raises(LedgerError, match="empty"):
        ModelCallLedger(str(path), allowances())
    assert path.read_bytes() == b""
    assert sorted(entry.name for entry in tmp_path.iterdir()) == ["ledger.sqlite3"]


def test_truncated_ledger_cannot_replenish_the_budget(tmp_path):
    """Review probe: one reserved call against a total allowance of one, then truncation."""
    path = tmp_path / "ledger.sqlite3"
    ledger = ModelCallLedger(str(path), allowances(calls_total=1))
    ledger.reserve("openai", "fake-model", 10, "only-call")
    with pytest.raises(BudgetExhaustedError):
        ledger.reserve("openai", "fake-model", 10, "refused")
    path.write_bytes(b"")
    with pytest.raises(LedgerError):
        ModelCallLedger(str(path), allowances(calls_total=1))
    assert path.stat().st_size == 0
    assert sorted(entry.name for entry in tmp_path.iterdir()) == ["ledger.sqlite3"]
    # A header-only remnant of the file is refused the same way.
    path.write_bytes(b"SQLite format 3\x00")
    with pytest.raises(LedgerError):
        ModelCallLedger(str(path), allowances(calls_total=1))


def test_first_initialization_is_deliberate_and_leaves_no_staging_file(tmp_path):
    path = tmp_path / "fresh.sqlite3"
    ledger = ModelCallLedger(str(path), allowances())
    assert path.stat().st_size > 0
    assert sorted(entry.name for entry in tmp_path.iterdir()) == ["fresh.sqlite3"]
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"ledger_meta", "model_calls"} <= tables
    assert ledger.summary()["ledger_created_at"]
    ledger.reserve("openai", "fake-model", 10, "first")
    # A normal restart verifies the existing file and keeps its history.
    assert ModelCallLedger(str(path), allowances()).summary()["calls_total"] == 1
    # Publishing again is refused once the path exists, and leaves the data untouched.
    assert ledger._create_new_database() is False
    assert ModelCallLedger(str(path), allowances()).summary()["calls_total"] == 1
    assert sorted(entry.name for entry in tmp_path.iterdir()) == ["fresh.sqlite3"]


def test_concurrent_first_initialization_publishes_exactly_one_ledger(tmp_path):
    path = tmp_path / "raced.sqlite3"
    start = threading.Barrier(8, timeout=10)
    outcomes = []
    lock = threading.Lock()

    def open_and_reserve(index):
        start.wait()
        try:
            ModelCallLedger(str(path), allowances(calls_per_day=50, calls_total=50)).reserve(
                "openai", "fake-model", 1, f"race-{index}"
            )
            outcome = "ok"
        except LedgerError:
            outcome = "error"
        with lock:
            outcomes.append(outcome)

    threads = [threading.Thread(target=open_and_reserve, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert outcomes == ["ok"] * 8
    assert sorted(entry.name for entry in tmp_path.iterdir()) == ["raced.sqlite3"]
    assert ModelCallLedger(str(path), allowances(calls_per_day=50, calls_total=50)).summary()["calls_total"] == 8


def test_ledger_rows_hold_no_prompt_text_and_bound_identifiers(tmp_path):
    path = tmp_path / "ledger.sqlite3"
    ledger = ModelCallLedger(str(path), allowances())
    ledger.reserve("openai", "m" * 500, 10, "r" * 500)
    with sqlite3.connect(path) as connection:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(model_calls)")]
        row = connection.execute("SELECT model, request_id FROM model_calls").fetchone()
    assert "prompt" not in columns and "question" not in columns
    assert len(row[0]) == 128 and len(row[1]) == 128


def test_invalid_reservation_values_are_rejected(tmp_path):
    ledger = ModelCallLedger(str(tmp_path / "ledger.sqlite3"), allowances())
    with pytest.raises(LedgerError):
        ledger.reserve("openai", "fake-model", -1, "negative")
    with pytest.raises(LedgerError):
        ledger.reserve("openai", "fake-model", "10", "string")
    with pytest.raises(LedgerError):
        ModelCallLedger("", allowances())
