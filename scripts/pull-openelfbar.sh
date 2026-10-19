#!/usr/bin/env bash
# Copy the OpenElfbar SQLite database off an Android phone over adb.
#
# Needs a debuggable build of the app (for `run-as`); build one with
#   ./gradlew assembleRelease -Popenelfbar.debuggableRelease=true
# and `adb install -r` it over the existing install (same signing key keeps data).
#
# Usage: scripts/pull-openelfbar.sh [DEST_DIR]   (default ~/annex/Logs/OpenElfbar)
# Set ANDROID_SERIAL if more than one device is connected.
set -euo pipefail

PKG=org.youngsoftware.openelfbar
DEST=${1:-$HOME/annex/Logs/OpenElfbar}
STAMP=$(date +%Y-%m-%dT%H%M)
OUT="$DEST/openelfbar-$STAMP.db"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

mkdir -p "$DEST"

# tar the whole dir so an un-checkpointed -wal file comes along
adb exec-out run-as "$PKG" tar -cf - files/SQLite | tar -xf - -C "$TMP"
SRC="$TMP/files/SQLite/openelfbar.db"
[[ -s "$SRC" ]] || { echo "error: no database pulled (is the app debuggable?)" >&2; exit 1; }

# .backup folds the WAL into a single self-contained file
sqlite3 "$SRC" ".backup '$OUT'"
[[ $(sqlite3 "$OUT" "PRAGMA integrity_check;") == ok ]] || { echo "error: integrity check failed: $OUT" >&2; exit 1; }

ln -sf "$(basename "$OUT")" "$DEST/openelfbar-latest.db"

echo "Saved $OUT"
sqlite3 "$OUT" "SELECT 'syncs: ' || count(*) || ', last: ' || max(started_at) FROM sync_runs;"
