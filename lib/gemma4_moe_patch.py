#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────
# lib/gemma4_moe_patch.py — Patch benchmark_moe.py for Gemma4 support
# ─────────────────────────────────────────────────────────────────────
#
# vLLM's upstream benchmark_moe.py does not support Gemma4 models
# (e.g. google/gemma-4-27b-it, RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic).
# Gemma4 uses `num_experts` + `top_k_experts` (not `num_experts_per_tok`)
# and falls through to the Mixtral/Llama4 `else` branch, which tries
# `config.num_local_experts` — an attribute that does not exist on
# Gemma4TextConfig — causing an AttributeError for every batch size.
#
# This script patches benchmark_moe.py in-place to add Gemma4 to the
# Qwen3 VL architecture block, with a `top_k_experts` fallback for the
# topk field.
#
# Usage (called inside the container from tune-moe.sh):
#   python3 -c "$(cat lib/gemma4_moe_patch.py)"
#
# Idempotent: re-running on an already-patched file is a no-op.
# Drift-safe: exits 0 with a warning if the upstream anchor has changed.
# ─────────────────────────────────────────────────────────────────────

from __future__ import annotations

import sys
from pathlib import Path

BENCH_PATH = Path("/tmp/vllm-bench/benchmarks/kernels/benchmark_moe.py")

# Sentinel used to detect whether the patch has already been applied.
SENTINEL = "# [vllm-tune: Gemma4ForConditionalGeneration patch]"

# The exact block in upstream benchmark_moe.py we are extending.
# Gemma4 uses the same get_text_config() pattern as Qwen3 VL models,
# but with `top_k_experts` instead of `num_experts_per_tok`.
ANCHOR = """\
    elif architecture in (
        "Qwen3VLMoeForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen3_5MoeTextConfig",
    ):
        text_config = config.get_text_config()
        E = text_config.num_experts
        topk = text_config.num_experts_per_tok
        intermediate_size = text_config.moe_intermediate_size
        hidden_size = text_config.hidden_size"""

REPLACEMENT = f"""\
    elif architecture in (
        "Qwen3VLMoeForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen3_5MoeTextConfig",
        "Gemma4ForConditionalGeneration",  {SENTINEL}
    ):
        text_config = config.get_text_config()
        E = text_config.num_experts
        # Gemma4 uses `top_k_experts`; Qwen3 uses `num_experts_per_tok`
        if hasattr(text_config, "top_k_experts"):
            topk = text_config.top_k_experts
        else:
            topk = text_config.num_experts_per_tok
        intermediate_size = text_config.moe_intermediate_size
        hidden_size = text_config.hidden_size"""


def main() -> int:
    if not BENCH_PATH.exists():
        print(f"ERROR: {BENCH_PATH} not found — has the clone step completed?",
              file=sys.stderr)
        return 1

    src = BENCH_PATH.read_text()

    if SENTINEL in src:
        print(f"[ok] {BENCH_PATH.name} already patched for Gemma4 — skipping")
        return 0

    if src.count(ANCHOR) != 1:
        # Upstream has changed the Qwen3 VL block — skip gracefully so
        # we don't corrupt benchmark_moe.py; the user will see an error
        # only if they are actually tuning a Gemma4 model.
        print(
            f"[warn] {BENCH_PATH.name}: Qwen3 VL anchor block not found or "
            f"duplicated — upstream drift likely. Gemma4 patch skipped.",
            file=sys.stderr,
        )
        return 0  # Non-fatal: other models are unaffected

    patched = src.replace(ANCHOR, REPLACEMENT, 1)
    BENCH_PATH.write_text(patched)
    print(f"[ok] {BENCH_PATH.name} patched for Gemma4ForConditionalGeneration")
    return 0


if __name__ == "__main__":
    sys.exit(main())
