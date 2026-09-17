#! /usr/bin/env bash
#
# Hub wrapper for vit_classify_hdf5.py. Translates the rendered INI into
# flags; the script stays an ordinary CLI tool.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${1:-}"

if [[ -z "$CONFIG_FILE" || ! -f "$CONFIG_FILE" ]]; then
    echo "Error: settings file '${CONFIG_FILE:-<none>}' not found." >&2
    exit 1
fi

# A missing key yields an empty string rather than aborting: the Hub omits
# optional settings entirely, and under `set -e` a non-matching grep would
# kill this script before it printed anything.
cfg() {
    grep -i "^[[:space:]]*$1[[:space:]]*=" "$CONFIG_FILE" \
        | head -1 | sed 's/^[^=]*=//' | sed 's/[[:space:]]*#.*//' \
        | sed 's/^[[:space:]]*//; s/[[:space:]]*$//' || true
}

SOURCE_PATH="$(cfg sourcePath)"
TARGET_PATH="$(cfg targetPath)"
MODEL_PATH="$(cfg modelPath)"
ISOLATE="$(cfg isolateCrops)"
EMIT="$(cfg emitEmbeddings)"
BATCH="$(cfg batchSize)"
TOPK="$(cfg topK)"
WORKERS="$(cfg workers)"
PRECISION="$(cfg precision)"
RESTRICT_TO="$(cfg restrictToPath)"
RESTRICT_LABELS="$(cfg restrictLabels)"
CARRY_FORWARD="$(cfg carryForward)"

for required in SOURCE_PATH TARGET_PATH MODEL_PATH; do
    if [[ -z "${!required}" ]]; then
        echo "Error: $required is required." >&2
        exit 1
    fi
done

ARGS=("$SOURCE_PATH" "-o" "$TARGET_PATH" "--model" "$MODEL_PATH")
[[ -n "$BATCH" ]] && ARGS+=("--batch-size" "$BATCH")
[[ -n "$TOPK" ]] && ARGS+=("--top-k" "$TOPK")
[[ -n "$WORKERS" ]] && ARGS+=("--workers" "$WORKERS")
[[ -n "$PRECISION" ]] && ARGS+=("--precision" "$PRECISION")
[[ -n "$RESTRICT_TO" ]] && ARGS+=("--restrict-to" "$RESTRICT_TO")
[[ -n "$RESTRICT_LABELS" ]] && ARGS+=("--restrict-labels" "$RESTRICT_LABELS")
# isolation is on by default; only an explicit falsey value turns it off.
case "${ISOLATE,,}" in 0|false|no|off) ARGS+=("--no-isolate-crops") ;; esac
case "${EMIT,,}" in 1|true|yes|on) ARGS+=("--emit-embeddings") ;; esac
# carry-forward is on by default; only an explicit falsey value turns it off.
case "${CARRY_FORWARD,,}" in 0|false|no|off) ARGS+=("--no-carry-forward") ;; esac

# torch and transformers live in the venv built from the official wheel; the
# Hub server's own (nix) Python must not leak in.
unset PYTHONPATH PYTHONHOME
PYTHON="${PISCO_PYTHON:-/opt/pisco/venvs/torch/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    echo "Warning: $PYTHON is not executable, falling back to the system python." >&2
    PYTHON="$(command -v python3)"
fi
# Fail on a clear message rather than a traceback from inside the script:
# a silently wrong interpreter is the confusing failure mode here.
if ! "$PYTHON" -c "import torch, transformers, h5py" 2>/dev/null; then
    echo "Error: $PYTHON cannot import torch, transformers, h5py. Set PISCO_PYTHON to an interpreter that can." >&2
    exit 1
fi

# Models pulled from Hugging Face need a writable cache; /tmp is per-container.
export HF_HOME="${HF_HOME:-/tmp/hf}"

echo "=== ViT classification ==="
echo "Interpreter: $PYTHON"
echo "Model:       $MODEL_PATH"
echo ""

exec "$PYTHON" "$SCRIPT_DIR/vit_classify_hdf5.py" "${ARGS[@]}"
