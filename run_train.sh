#! /usr/bin/env bash
#
# Hub wrapper for training/train_vit.py. One task trains one stage; run it
# twice against the same trainset to rebuild both halves of the dual ViT.

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

DATA_PATH="$(cfg dataPath)"
STAGE="$(cfg stage)"
INIT_MODEL="$(cfg initModel)"
TARGET_PATH="$(cfg targetPath)"
EPOCHS="$(cfg epochs)"
LR="$(cfg learningRate)"
BS="$(cfg batchSize)"
SEED="$(cfg seed)"
WORKERS="$(cfg workers)"
PRECISION="$(cfg precision)"

for required in DATA_PATH INIT_MODEL TARGET_PATH; do
    if [[ -z "${!required}" ]]; then
        echo "Error: $required is required." >&2
        exit 1
    fi
done

# Chaining hands over the trainset directory, which holds both stages; `stage`
# picks one. A dataPath that is already an imagefolder is used as given.
if [[ -n "$STAGE" ]]; then
    if [[ -d "${DATA_PATH%/}/$STAGE" ]]; then
        DATA_PATH="${DATA_PATH%/}/$STAGE"
    else
        echo "Error: stage '$STAGE' requested but ${DATA_PATH%/}/$STAGE does not exist." >&2
        exit 1
    fi
fi

ARGS=("--data" "$DATA_PATH" "--out" "$TARGET_PATH" "--init" "$INIT_MODEL")
[[ -n "$EPOCHS" ]] && ARGS+=("--epochs" "$EPOCHS")
[[ -n "$LR" ]] && ARGS+=("--lr" "$LR")
[[ -n "$BS" ]] && ARGS+=("--bs" "$BS")
[[ -n "$SEED" ]] && ARGS+=("--seed" "$SEED")
[[ -n "$WORKERS" ]] && ARGS+=("--workers" "$WORKERS")
[[ -n "$PRECISION" ]] && ARGS+=("--precision" "$PRECISION")

unset PYTHONPATH PYTHONHOME
PYTHON="${PISCO_PYTHON:-/opt/pisco/venvs/torch/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    echo "Warning: $PYTHON is not executable, falling back to the system python." >&2
    PYTHON="$(command -v python3)"
fi
if ! "$PYTHON" -c "import torch, transformers, torchvision, sklearn" 2>/dev/null; then
    echo "Error: $PYTHON cannot import torch, transformers, torchvision, sklearn. Set PISCO_PYTHON to an interpreter that can." >&2
    exit 1
fi

# A checkpoint pulled from Hugging Face needs a writable cache.
export HF_HOME="${HF_HOME:-/tmp/hf}"

echo "=== ViT training ==="
echo "Interpreter: $PYTHON"
echo "Data:        $DATA_PATH"
echo "Init:        $INIT_MODEL"
echo ""

exec "$PYTHON" "$SCRIPT_DIR/training/train_vit.py" "${ARGS[@]}"
