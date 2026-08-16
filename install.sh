#!/usr/bin/env bash
# One command from a fresh clone to a working advisor.
#
# Creates the virtualenv, installs, and offers to load the shared corpus
# snapshot so you skip an hour of harvesting and three hours of encoding.
# Safe to re-run: every step checks whether it is already done.

set -euo pipefail
cd "$(dirname "$0")"

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }
note() { printf '  %s\n' "$*"; }

# --- python -----------------------------------------------------------------

if ! command -v python3 >/dev/null; then
  echo "python3 is not installed. Install Python 3.11 or newer, then re-run." >&2
  exit 1
fi

version=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
  echo "Python $version is too old; this needs 3.11 or newer." >&2
  exit 1
fi
say "Python $version"

# --- virtualenv -------------------------------------------------------------

if [ ! -d .venv ]; then
  say "Creating the virtualenv"
  python3 -m venv .venv
else
  say "Virtualenv already exists"
fi

say "Installing (this pulls PyTorch, so it is the slow part — several minutes)"
./.venv/bin/pip install --quiet --upgrade pip
./.venv/bin/pip install --quiet -e '.[all]'
note "installed"

# --- corpus -----------------------------------------------------------------

embedded=$(./.venv/bin/python -c '
from advisor import config, db
conn = db.connect(config.load().db_path)
print(conn.execute("SELECT count(*) FROM vector_index").fetchone()[0])
conn.close()' 2>/dev/null || echo 0)

if [ "$embedded" -gt 0 ]; then
  say "Corpus already loaded: $embedded papers embedded"
elif command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
  say "Downloading the shared corpus (~151 MB)"
  note "76,000 papers, already embedded — this replaces ~4 hours of work"
  if ./.venv/bin/advisor snapshot fetch; then
    note "corpus ready"
  else
    note "download failed; you can build your own with 'advisor harvest' then 'advisor embed'"
  fi
else
  say "Skipping the corpus download"
  note "The GitHub CLI ('gh') is not installed or not logged in, and the"
  note "repository is private, so the snapshot cannot be fetched automatically."
  note ""
  note "Either install gh from https://cli.github.com and re-run this script,"
  note "or build the corpus yourself:"
  note "    source .venv/bin/activate"
  note "    advisor harvest && advisor embed"
fi

# --- done -------------------------------------------------------------------

say "Ready. Next:"
cat <<'NEXT'
    source .venv/bin/activate          # once per terminal
    advisor add 2401.12345 2024/123    # papers you have read — ten is plenty
    advisor serve                      # http://127.0.0.1:8000

  Then open the browser and press "Get new recommendations".
NEXT
