#!/bin/bash
# Mod: vllm-tune
# Install tuned kernel configs into a running vLLM container:
#   1. Fused MoE kernel configs (configs/)
#   2. W8A8 Block FP8 dense GEMM kernel configs (fp8-configs/)
#
# Without these, vLLM falls back to default heuristics and prints:
#   "Using default MoE config. Performance might be sub-optimal!"
#   "Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal!"
#
# Configs are generated via vllm-tune:
#   ./vllm-tune.sh <model> --tp <N> --sync-mod
#
set -eo pipefail

MOD_DIR="$(cd "$(dirname "$0")" && pwd)"

# Auto-detect vLLM install path (works across Python versions)
VLLM_ROOT=$(python3 -c "import vllm, pathlib; print(pathlib.Path(vllm.__file__).parent)" 2>/dev/null) \
    || VLLM_ROOT="/usr/local/lib/python3.12/dist-packages/vllm"

# Install JSON configs from a source directory into a target directory.
# Usage: install_configs <label> <src_dir> <dst_dir>
install_configs() {
    local label="$1" src="$2" dst="$3"
    local installed=0 skipped=0 count=0

    echo "--- [vllm-tune] Installing $label..."

    if [[ ! -d "$src" ]]; then
        echo "    (no $src directory — skipping)"
        return
    fi

    for cfg in "$src"/*.json; do
        [[ -f "$cfg" ]] || continue
        count=$((count + 1))
    done

    if [[ $count -eq 0 ]]; then
        echo "    (no JSON files in $src — skipping)"
        return
    fi

    mkdir -p "$dst"

    for cfg in "$src"/*.json; do
        [[ -f "$cfg" ]] || continue
        local name
        name="$(basename "$cfg")"
        if [[ -f "$dst/$name" ]]; then
            echo "    [skip] $name (already exists)"
            skipped=$((skipped + 1))
        else
            install -m 0644 "$cfg" "$dst/$name"
            echo "    [ok]   $name"
            installed=$((installed + 1))
        fi
    done

    echo "    Installed: $installed, skipped: $skipped"
}

# 1. Fused MoE kernel configs (E=...,N=... files)
install_configs "tuned MoE kernel configs" \
    "$MOD_DIR/configs" \
    "$VLLM_ROOT/model_executor/layers/fused_moe/configs"

# 2. W8A8 Block FP8 dense GEMM kernel configs (N=...,K=... files)
install_configs "tuned FP8 GEMM kernel configs" \
    "$MOD_DIR/fp8-configs" \
    "$VLLM_ROOT/model_executor/layers/quantization/utils/configs"

echo "--- [vllm-tune] Done."

