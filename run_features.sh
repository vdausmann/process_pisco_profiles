#! /usr/bin/env bash
#
# Hub wrapper for ecotaxa_write_features.py. Runs on the EcoTaxa machine,
# where the database is reachable on loopback.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${1:-}"

if [[ -z "$CONFIG_FILE" || ! -f "$CONFIG_FILE" ]]; then
    echo "Error: settings file '${CONFIG_FILE:-<none>}' not found." >&2
    exit 1
fi

cfg() {
    grep -i "^[[:space:]]*$1[[:space:]]*=" "$CONFIG_FILE" \
        | head -1 | sed 's/^[^=]*=//' | sed 's/[[:space:]]*#.*//' \
        | sed 's/^[[:space:]]*//; s/[[:space:]]*$//' || true
}

EMBEDDINGS="$(cfg embeddingsPath)"
PROJECT_ID="$(cfg projectId)"
PROJECT_ID_FROM="$(cfg projectIdFrom)"
PCA_PATH="$(cfg pcaPath)"
DSN="$(cfg dsn)"
ALLOW_PARTIAL="$(cfg allowPartial)"

for required in EMBEDDINGS PCA_PATH; do
    if [[ -z "${!required}" ]]; then
        echo "Error: $required is required." >&2
        exit 1
    fi
done

# Chaining hands over the producing task's directory, not the file in it.
if [[ -d "$EMBEDDINGS" ]]; then
    EMBEDDINGS="${EMBEDDINGS%/}/embeddings.npz"
fi

if [[ -z "$PROJECT_ID" && -z "$PROJECT_ID_FROM" ]]; then
    echo "Error: projectId or projectIdFrom is required." >&2
    exit 1
fi

ARGS=("$EMBEDDINGS" "--pca" "$PCA_PATH")
[[ -n "$PROJECT_ID" ]] && ARGS+=("--project-id" "$PROJECT_ID")
[[ -n "$PROJECT_ID_FROM" ]] && ARGS+=("--project-id-from" "$PROJECT_ID_FROM")
[[ -n "$DSN" ]] && ARGS+=("--dsn" "$DSN")
case "${ALLOW_PARTIAL,,}" in 1|true|yes|on) ARGS+=("--allow-partial") ;; esac

unset PYTHONPATH PYTHONHOME
PYTHON="${PISCO_ECOTAXA_PYTHON:-/opt/pisco/venvs/ecotaxa/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    echo "Warning: $PYTHON is not executable, falling back to the system python." >&2
    PYTHON="$(command -v python3)"
fi
# Fail on a clear message rather than a traceback from inside the script:
# a silently wrong interpreter is the confusing failure mode here.
if ! "$PYTHON" -c "import numpy, psycopg" 2>/dev/null; then
    echo "Error: $PYTHON cannot import numpy, psycopg. Set PISCO_ECOTAXA_PYTHON to an interpreter that can." >&2
    exit 1
fi

echo "=== EcoTaxa feature write ==="
echo "Interpreter: $PYTHON"
echo "Embeddings:  $EMBEDDINGS"
echo "Project:     ${PROJECT_ID:-from $PROJECT_ID_FROM}"
echo ""

exec "$PYTHON" "$SCRIPT_DIR/ecotaxa_write_features.py" "${ARGS[@]}"
