#!/bin/sh
# Entrypoint for the real profile (Dockerfile.real runtime target).
#
# A mounted persistent disk is owned by whoever the platform mounted it for,
# usually root, and the API runs as the unprivileged user "app" (UID 10001).
# The ledger must be created inside that mount, so this script runs as root
# only long enough to create the ledger directory and hand it to "app", then
# drops to "app" with an empty capability set and no way to regain privileges
# before executing the server command. Started as a non-root user (the test
# target, or a platform that forbids root), it executes the command unchanged
# and the application still fails closed if the ledger path is not writable.
#
# Ownership changes are limited to the ledger directory, the ledger file and
# its SQLite sidecar files; a mounted volume is never chowned recursively.
set -eu

RUNTIME_USER=app

if [ "$(id -u)" -eq 0 ]; then
    if [ -n "${MODEL_CALL_LEDGER_PATH:-}" ]; then
        ledger_dir=$(dirname -- "$MODEL_CALL_LEDGER_PATH")
        mkdir -p -- "$ledger_dir"
        chown -- "$RUNTIME_USER:$RUNTIME_USER" "$ledger_dir"
        for ledger_file in "$MODEL_CALL_LEDGER_PATH" \
                           "$MODEL_CALL_LEDGER_PATH-journal" \
                           "$MODEL_CALL_LEDGER_PATH-wal" \
                           "$MODEL_CALL_LEDGER_PATH-shm"; do
            if [ -f "$ledger_file" ]; then
                chown -- "$RUNTIME_USER:$RUNTIME_USER" "$ledger_file"
            fi
        done
    fi
    if [ -n "${PAPER_STORE_PATH:-}" ]; then
        corpus_dir=$(dirname -- "$PAPER_STORE_PATH")
        mkdir -p -- "$corpus_dir" "$PAPER_STORE_PATH.vectors"
        chown -- "$RUNTIME_USER:$RUNTIME_USER" "$corpus_dir" "$PAPER_STORE_PATH.vectors"
        for corpus_file in "$PAPER_STORE_PATH" "$PAPER_STORE_PATH.lock" \
                           "$PAPER_STORE_PATH-journal" "$PAPER_STORE_PATH-wal" "$PAPER_STORE_PATH-shm"; do
            if [ -f "$corpus_file" ]; then
                chown -- "$RUNTIME_USER:$RUNTIME_USER" "$corpus_file"
            fi
        done
    fi
    # setpriv keeps the environment; give the runtime user its own home so
    # libraries that touch ~/.cache never see root's directory.
    HOME=$(getent passwd "$RUNTIME_USER" | cut -d: -f6)
    export HOME="${HOME:-/home/$RUNTIME_USER}" USER="$RUNTIME_USER" LOGNAME="$RUNTIME_USER"
    exec setpriv --reuid="$RUNTIME_USER" --regid="$RUNTIME_USER" --init-groups \
        --inh-caps=-all --bounding-set=-all --no-new-privs -- "$@"
fi

exec "$@"
