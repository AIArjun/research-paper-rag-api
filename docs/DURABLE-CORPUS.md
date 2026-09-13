# Durable paper corpus

Set `PAPER_STORE_PATH` to an absolute SQLite file on persistent **local** storage,
for example `/var/data/corpus/papers.sqlite3` on the existing Render disk.
The derived Chroma index is then stored at `<PAPER_STORE_PATH>.vectors`;
`VECTORSTORE_PATH` is used only when `PAPER_STORE_PATH` is empty.
The model-call ledger remains a separate file with unchanged allowances.

This is still one worker, one instance and one shared public-paper library.
It does not add private accounts, document ownership, encrypted storage,
automated backups or guaranteed recovery from loss of the disk itself.
An unset `PAPER_STORE_PATH` retains the disposable development profile.
`/health` and `/ready` report `corpus_storage` as `persistent`, `ephemeral`, or
`unavailable`. The word `persistent` identifies the configured storage mode;
only an actual restart check can establish that the platform mounted durable storage.

## What is committed

The canonical SQLite record contains the exact PDF bytes, SHA-256 identity,
canonical filename and upload time, extracted text with physical page numbers,
chunk text/offsets, and a `pending` or `ready` mutation state. A committed upload
retains these values after restart; duplicate bytes return the existing record.

1. Upload commits its pending record before writing any vectors.
2. Only after indexing succeeds does it commit `ready` and publish the paper.
3. Delete commits `pending` before removing vectors. It removes the canonical
   data only after vector deletion succeeds, then clears the in-memory mirror.

SQLite uses full synchronous commits and a rollback journal. Chroma and SQLite
are separate databases, so a process interruption can leave a pending record.
That record survives restart, is listed by `/papers` as `pending_cleanup`, is
excluded from active counts, and blocks queries until authenticated deletion is
retried successfully. A response lost after a successful commit is resolved by
listing or repeating the same upload; it is not proof that nothing was saved.

On startup, canonical checksums, configuration and capacity are validated before
publishing the library. Missing or inconsistent **known, ready** index entries
are rebuilt from saved chunks using local embeddings, in batches of at most 64.
No extraction, provider call or budget reservation is needed for this repair.
Unknown vector IDs fail closed; they are never adopted or silently deleted.
Changing the embedding model, chunk settings or storage mode requires an
explicit migration, rather than mixing incompatible index contents.

An OS file lock rejects a second process for the same canonical path. Run one
worker only. Locks release on clean shutdown and process death. Do not use NFS,
multiple machines, multiple aliases of the same file, or multiple canonical
paths pointing at one index. Back up the entire corpus directory with the
service stopped; include the separate model ledger without resetting it.

## Render migration and verification

This supersedes the ledger-only corpus instructions in `STAGE3.md` **only after
the new code is deployed and the new variable is configured**.

1. Record the deployed commit, authenticated paper list and budget counters.
   Save and hash-check each original public PDF before replacing the disposable
   deployment. Existing vector files alone cannot recover pages or the registry.
2. Keep the existing disk, model ledger path, key, access token, budgets and
   single-worker configuration. Set only
   `PAPER_STORE_PATH=/var/data/corpus/papers.sqlite3` and deploy the reviewed code.
   The entrypoint grants the existing runtime user ownership of the corpus
   directory, canonical/lock/journal files and index directory; it never changes
   ownership recursively over the disk or its unrelated contents.
3. Re-upload the verified PDFs once. This initial migration creates new upload
   timestamps; filenames, SHA-256 IDs, pages and chunk counts must match. Save the
   new complete paper-list snapshot as the restart baseline.
4. Restart the service without uploading anything. Require the same full paper
   metadata and counts, `corpus_storage=persistent`, no pending cleanup, and
   unchanged ledger identity and usage counters.
5. Use the offline deletion/crash tests below. If doing an additional live
   deletion check, use an explicitly disposable test paper, never a user's only
   copy. Do not use a paid model query to test storage.

If startup reports `corpus_unavailable` or `corpus_recovery_failed`, retain the
files, inspect the configuration/permissions/backup and fix the specific cause.
Do not delete the database or create a new empty ledger to make health green.
Rollback to older code does not understand this journal; stop writes and retain
the corpus directory for a compatible restore. Disk snapshots/backups may retain
deleted content: API deletion is logical deletion, not certified media erasure.

## Checks

`tests/test_durable_corpus.py` injects upload/deletion interruptions, failed
rollback, failed database writes, corrupt bytes, incompatible configuration,
capacity reduction and index inconsistencies. It also kills a real subprocess
after committing upload intent and recovers in a fresh process.

`scripts/verify_durable_corpus.py` uses the pinned public Transformer and RAG
PDFs with real Chroma and local embeddings across five separate processes:
populate, restart, repair missing vectors, delete, and restart an empty corpus.
It compares exact metadata/pages/chunks/index contents, original bytes, duplicate
identity and source-page evidence. There is no network-capable model object.

The real-profile workflow runs this helper with networking disabled under the
2 GiB / 1 CPU bound and records `durable-corpus.jsonl`. Its normal runtime
measurement now also sets `PAPER_STORE_PATH` under the root-owned test mount,
checking corpus permissions alongside the existing ledger/user/capability checks.
These are storage and regression checks, not a general answer-quality benchmark.
