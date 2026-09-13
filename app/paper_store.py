"""Canonical paper data and mutation journal for one process on a local disk.

Chroma is a derived index. A pending row is committed before changing it, and
only a ready row may be served. SQLite and Chroma are not one transaction;
interrupted mutations stay visible as pending cleanup after restart.
"""

import hashlib
import json
import os
from pathlib import Path
import sqlite3


class PaperStoreError(RuntimeError):
    """Safe, categorical storage failure (never include document contents)."""


class PaperStore:
    def __init__(self, path: str, configuration: dict):
        self.path = Path(path).resolve()
        self._connection = None
        self._lock_file = None
        self._locked = False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # OS locks are released even on process death. Never unlink the lock
            # file: replacing its inode could allow two owners of the same store.
            self._lock_file = open(str(self.path) + ".lock", "a+b")
            self._lock_file.seek(0)
            if os.name == "nt":
                import msvcrt
                if self._lock_file.read(1) == b"":
                    self._lock_file.write(b"0")
                    self._lock_file.flush()
                self._lock_file.seek(0)
                msvcrt.locking(self._lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._locked = True
            self._connection = sqlite3.connect(self.path, timeout=5, check_same_thread=False)
            self._connection.execute("PRAGMA journal_mode=DELETE")
            self._connection.execute("PRAGMA synchronous=FULL")
            self._connection.execute("PRAGMA secure_delete=ON")
            if self._connection.execute("PRAGMA quick_check").fetchone() != ("ok",):
                raise PaperStoreError("corpus_corrupt")
            signature = json.dumps(configuration, sort_keys=True)
            with self._connection:
                self._connection.execute(
                    "CREATE TABLE IF NOT EXISTS configuration (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT NOT NULL)"
                )
                self._connection.execute(
                    "CREATE TABLE IF NOT EXISTS papers (paper_id TEXT PRIMARY KEY, "
                    "state TEXT NOT NULL CHECK(state IN ('pending', 'ready')), "
                    "pdf BLOB NOT NULL, payload TEXT NOT NULL, digest TEXT NOT NULL)"
                )
                existing = self._connection.execute("SELECT value FROM configuration WHERE id=1").fetchone()
                if existing is None:
                    if self._connection.execute("SELECT count(*) FROM papers").fetchone()[0]:
                        raise PaperStoreError("corpus_configuration_missing")
                    self._connection.execute("INSERT INTO configuration VALUES (1, ?)", (signature,))
                elif existing[0] != signature:
                    raise PaperStoreError("corpus_configuration_mismatch")
        except Exception as error:
            self.close()
            if isinstance(error, PaperStoreError):
                raise
            raise PaperStoreError("corpus_unavailable") from error

    def load(self, limits) -> list[dict]:
        rows = []
        total_chunks = 0
        for paper_id, state, pdf, payload, digest in self._connection.execute(
            "SELECT paper_id, state, pdf, payload, digest FROM papers ORDER BY rowid"
        ):
            if (len(rows) >= limits.max_papers or len(pdf) > limits.max_file_bytes
                    or hashlib.sha256(pdf).hexdigest() != paper_id
                    or hashlib.sha256(payload.encode()).hexdigest() != digest):
                raise PaperStoreError("corpus_integrity_or_capacity")
            data = json.loads(payload)
            paper, pages, chunks = data["paper"], data["pages"], data["chunks"]
            total_chunks += len(chunks)
            if (paper["paper_id"] != paper_id or paper["pages"] != len(pages)
                    or paper["chunks"] != len(chunks) or not pages or not chunks
                    or len(chunks) > limits.max_chunks_per_paper
                    or total_chunks > limits.max_total_chunks
                    or any(type(p["page"]) is not int or not 1 <= p["page"] <= limits.max_pdf_pages for p in pages)):
                raise PaperStoreError("corpus_integrity_or_capacity")
            rows.append({**data, "state": state})
        return rows

    def stage(self, pdf: bytes, paper: dict, pages: list[dict], chunks: list[dict]) -> None:
        payload = json.dumps({"paper": paper, "pages": pages, "chunks": chunks}, ensure_ascii=False)
        with self._connection:
            self._connection.execute(
                "INSERT INTO papers VALUES (?, 'pending', ?, ?, ?)",
                (paper["paper_id"], pdf, payload, hashlib.sha256(payload.encode()).hexdigest()),
            )

    def mark(self, paper_id: str, state: str) -> None:
        with self._connection:
            changed = self._connection.execute(
                "UPDATE papers SET state=? WHERE paper_id=?", (state, paper_id)
            ).rowcount
            if changed != 1:
                raise PaperStoreError("corpus_record_missing")

    def remove(self, paper_id: str) -> None:
        with self._connection:
            self._connection.execute("DELETE FROM papers WHERE paper_id=?", (paper_id,))

    def read_pdf(self, paper_id: str) -> bytes | None:
        row = self._connection.execute(
            "SELECT pdf FROM papers WHERE paper_id=? AND state='ready'", (paper_id,)
        ).fetchone()
        return bytes(row[0]) if row else None

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        if self._lock_file is not None:
            if self._locked:
                if os.name == "nt":
                    import msvcrt
                    self._lock_file.seek(0)
                    msvcrt.locking(self._lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
            self._lock_file.close()
            self._lock_file = None
            self._locked = False
