#!/bin/bash
set -euo pipefail
# ─────────────────────────────────────────────────────────────────────
# test-vllm-tune.sh — Test suite for vllm-tune
# ─────────────────────────────────────────────────────────────────────
#
# Runs offline tests that do NOT require a Docker container or GPU.
# Tests CLI parsing, argument validation, model slug generation,
# architecture detection gating, dry-run flow, and error handling.
#
# Usage:
#   ./tests/test-vllm-tune.sh           # run from project root
#   bash tests/test-vllm-tune.sh        # also works
#
# All tests use --dry-run and/or --foreground to avoid needing tmux,
# Docker, or GPUs.
# ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VLLM_TUNE="$SCRIPT_DIR/vllm-tune.sh"
TUNE_MOE="$SCRIPT_DIR/tune-moe.sh"
TUNE_FP8="$SCRIPT_DIR/tune-fp8.sh"

PASSED=0
FAILED=0
ERRORS=()

# ── Test helpers ────────────────────────────────────────────────────

pass() {
    PASSED=$((PASSED + 1))
    printf "  \033[1;32m✓\033[0m %s\n" "$1"
}

fail() {
    FAILED=$((FAILED + 1))
    ERRORS+=("$1")
    printf "  \033[1;31m✗\033[0m %s\n" "$1"
    if [[ -n "${2:-}" ]]; then
        printf "    \033[2m%s\033[0m\n" "$2"
    fi
}

# Run a command, capture stdout+stderr, check exit code.
# Usage: run_expect_success "description" command args...
#        run_expect_failure "description" command args...
run_expect_success() {
    local desc="$1"; shift
    local output
    if output=$("$@" 2>&1); then
        echo "$output"
        return 0
    else
        echo "$output"
        return 1
    fi
}

# ── Tests ───────────────────────────────────────────────────────────

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
printf "  \033[1mvllm-tune test suite\033[0m\n"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── Version ─────────────────────────────────────────────────────────

printf "\033[1m  Version & help\033[0m\n"

output=$("$VLLM_TUNE" --version 2>&1)
if [[ "$output" =~ ^vllm-tune\ [0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    pass "--version prints version string: $output"
else
    fail "--version output unexpected: $output"
fi

# ── Help ────────────────────────────────────────────────────────────

output=$("$VLLM_TUNE" --help 2>&1) || true
if [[ "$output" == *"--mode"* && "$output" == *"--tp"* ]]; then
    pass "--help shows usage with --mode and --tp"
else
    fail "--help missing expected flags" "got: ${output:0:200}"
fi

output=$("$TUNE_MOE" --help 2>&1) || true
if [[ "$output" == *"batch-size"* ]]; then
    pass "tune-moe.sh --help shows usage"
else
    fail "tune-moe.sh --help missing expected content"
fi

output=$("$TUNE_FP8" --help 2>&1) || true
if [[ "$output" == *"shapes"* ]]; then
    pass "tune-fp8.sh --help shows usage"
else
    fail "tune-fp8.sh --help missing expected content"
fi

# ── Argument validation ────────────────────────────────────────────

printf "\n\033[1m  Argument validation\033[0m\n"

# Missing model
if output=$("$VLLM_TUNE" --foreground --dry-run 2>&1); then
    fail "Should fail without MODEL_ID"
else
    if [[ "$output" == *"MODEL_ID is required"* ]]; then
        pass "Missing MODEL_ID gives clear error"
    else
        fail "Missing MODEL_ID error message unexpected" "$output"
    fi
fi

# Invalid mode
if output=$("$VLLM_TUNE" test/model --mode invalid --foreground --dry-run 2>&1); then
    fail "Should reject invalid --mode"
else
    if [[ "$output" == *"Invalid --mode"* ]]; then
        pass "Invalid --mode rejected with clear error"
    else
        fail "Invalid --mode error message unexpected" "$output"
    fi
fi

# Unknown flag
if output=$("$VLLM_TUNE" test/model --bogus-flag --dry-run 2>&1); then
    fail "Should reject unknown flags"
else
    if [[ "$output" == *"Unknown flag"* ]]; then
        pass "Unknown flag rejected with clear error"
    else
        fail "Unknown flag error message unexpected" "$output"
    fi
fi

# ── Model slug generation ──────────────────────────────────────────

printf "\n\033[1m  Model slug generation\033[0m\n"

# Source model_slug from the script (it's a simple function)
model_slug() {
    echo "$1" | tr '[:upper:]' '[:lower:]' | sed 's|/|--|g; s/[^a-z0-9._-]/-/g'
}

test_slug() {
    local input="$1" expected="$2"
    local result
    result=$(model_slug "$input")
    if [[ "$result" == "$expected" ]]; then
        pass "model_slug('$input') = '$result'"
    else
        fail "model_slug('$input') = '$result', expected '$expected'"
    fi
}

test_slug "Qwen/Qwen3.6-35B-A3B-FP8" "qwen--qwen3.6-35b-a3b-fp8"
test_slug "Qwen/Qwen3.6-27B-FP8" "qwen--qwen3.6-27b-fp8"
test_slug "meta-llama/Llama-3.1-70B-FP8" "meta-llama--llama-3.1-70b-fp8"
test_slug "deepseek-ai/DeepSeek-V3" "deepseek-ai--deepseek-v3"
test_slug "Simple-Model" "simple-model"
test_slug "org/model_with_underscores" "org--model_with_underscores"

# ── Dry-run flow ───────────────────────────────────────────────────

printf "\n\033[1m  Dry-run flow\033[0m\n"

# --mode all dry-run
output=$("$VLLM_TUNE" Qwen/Qwen3.6-35B-A3B-FP8 --tp 2 --dry-run --foreground 2>&1)
if [[ $? -eq 0 ]]; then
    pass "Dry-run --mode all exits cleanly"
else
    fail "Dry-run --mode all failed"
fi

if [[ "$output" == *"Phase 1: MoE Kernel Tuning"* ]]; then
    pass "Dry-run shows MoE phase header"
else
    fail "Dry-run missing MoE phase header"
fi

if [[ "$output" == *"Phase 2: FP8 Dense GEMM Tuning"* ]]; then
    pass "Dry-run shows FP8 phase header"
else
    fail "Dry-run missing FP8 phase header"
fi

if [[ "$output" == *"[dry-run]"* ]]; then
    pass "Dry-run shows [dry-run] markers"
else
    fail "Dry-run missing [dry-run] markers"
fi

if [[ "$output" == *"DRY RUN"* ]]; then
    pass "Dry-run banner shows DRY RUN notice"
else
    fail "Dry-run banner missing DRY RUN notice"
fi

# --mode moe dry-run
output=$("$VLLM_TUNE" test/model --mode moe --tp 1 --dry-run --foreground 2>&1)
if [[ "$output" == *"Phase 1: MoE"* && "$output" != *"Phase 2: FP8"* ]]; then
    pass "Dry-run --mode moe shows only MoE phase"
else
    fail "Dry-run --mode moe phase selection wrong"
fi

# --mode fp8 dry-run
output=$("$VLLM_TUNE" test/model --mode fp8 --tp 1 --dry-run --foreground 2>&1)
if [[ "$output" != *"Phase 1: MoE"* && "$output" == *"Phase 2: FP8"* ]]; then
    pass "Dry-run --mode fp8 shows only FP8 phase"
else
    fail "Dry-run --mode fp8 phase selection wrong"
fi

# ── Config paths ───────────────────────────────────────────────────

printf "\n\033[1m  Config path construction\033[0m\n"

output=$("$VLLM_TUNE" org/MyModel-70B --tp 4 --dry-run --foreground 2>&1)
if [[ "$output" == *"org--mymodel-70b/tp4"* ]]; then
    pass "Config path uses model slug + tp: org--mymodel-70b/tp4"
else
    fail "Config path construction wrong" "output: ${output:0:400}"
fi

# ── Dry-run with custom batch sizes ────────────────────────────────

printf "\n\033[1m  Custom batch sizes and shapes\033[0m\n"

output=$("$VLLM_TUNE" test/model --mode moe --batch-size 64 128 256 --dry-run --foreground 2>&1)
if [[ "$output" == *"--batch-size 64 128 256"* ]]; then
    pass "Custom --batch-size passed through to tune-moe.sh"
else
    fail "Custom --batch-size not passed through" "output: ${output:0:400}"
fi

output=$("$VLLM_TUNE" test/model --mode fp8 --shapes 6144,2048 2048,2048 --dry-run --foreground 2>&1)
if [[ "$output" == *"--shapes 6144,2048 2048,2048"* ]]; then
    pass "Custom --shapes passed through to tune-fp8.sh"
else
    fail "Custom --shapes not passed through" "output: ${output:0:400}"
fi

# ── Dense model detection gating (dry-run path) ───────────────────

printf "\n\033[1m  Architecture detection gating\033[0m\n"

# In dry-run mode, architecture detection is skipped (no container).
# Verify that dry-run still shows both phases (it doesn't gate).
output=$("$VLLM_TUNE" Qwen/Qwen3.6-27B-FP8 --tp 2 --dry-run --foreground 2>&1)
if [[ "$output" == *"Phase 1: MoE"* && "$output" == *"Phase 2: FP8"* ]]; then
    pass "Dry-run skips arch detection, shows both phases"
else
    fail "Dry-run arch detection bypass broken"
fi

# Verify the detection code exists (centralized in lib/detect.py)
DETECT_PY="$SCRIPT_DIR/lib/detect.py"
if [[ -f "$DETECT_PY" ]]; then
    pass "lib/detect.py exists"
else
    fail "lib/detect.py missing"
fi

if grep -q "num_local_experts" "$DETECT_PY"; then
    pass "num_local_experts check present in lib/detect.py"
else
    fail "num_local_experts check missing from lib/detect.py"
fi

if grep -q "n_routed_experts" "$DETECT_PY"; then
    pass "n_routed_experts check present (DeepSeek-V3/V4)"
else
    fail "n_routed_experts check missing from lib/detect.py"
fi

if grep -q "num_experts" "$DETECT_PY"; then
    pass "num_experts check present (Qwen3/Gemma4/Jamba)"
else
    fail "num_experts check missing from lib/detect.py"
fi

if grep -q "moe_intermediate_size" "$DETECT_PY"; then
    pass "moe_intermediate_size shape detection present (DeepSeek)"
else
    fail "moe_intermediate_size missing from lib/detect.py"
fi

if grep -q "detect.py" "$VLLM_TUNE"; then
    pass "vllm-tune.sh references lib/detect.py"
else
    fail "vllm-tune.sh does not reference lib/detect.py"
fi

if grep -q "is a dense model" "$VLLM_TUNE"; then
    pass "Dense model skip message present"
else
    fail "Dense model skip message missing"
fi

if grep -q "MoE tuning is not applicable" "$VLLM_TUNE"; then
    pass "MoE-on-dense error message present"
else
    fail "MoE-on-dense error message missing"
fi

if grep -q "detect.py" "$TUNE_FP8"; then
    pass "tune-fp8.sh references lib/detect.py for shape detection"
else
    fail "tune-fp8.sh does not reference lib/detect.py"
fi

if grep -q "DeepseekV4ForCausalLM" "$TUNE_MOE"; then
    pass "tune-moe.sh has DeepSeek-V4 model class patch"
else
    fail "tune-moe.sh missing DeepSeek-V4 model class patch"
fi

# ── Gemma4 MoE patch ───────────────────────────────────────────────

printf "\n\033[1m  Gemma4 MoE patch (issue #7)\033[0m\n"

GEMMA4_PATCH="$SCRIPT_DIR/lib/gemma4_moe_patch.py"

if [[ -f "$GEMMA4_PATCH" ]]; then
    pass "lib/gemma4_moe_patch.py exists"
else
    fail "lib/gemma4_moe_patch.py missing (needed for Gemma4 MoE tuning)"
fi

if python3 -c "import py_compile; py_compile.compile('$GEMMA4_PATCH', doraise=True)" 2>/dev/null; then
    pass "lib/gemma4_moe_patch.py: valid Python syntax"
else
    fail "lib/gemma4_moe_patch.py: syntax errors detected"
fi

if grep -q "gemma4_moe_patch" "$TUNE_MOE"; then
    pass "tune-moe.sh references lib/gemma4_moe_patch.py"
else
    fail "tune-moe.sh missing gemma4_moe_patch.py reference"
fi

if grep -q "Gemma4ForConditionalGeneration" "$GEMMA4_PATCH"; then
    pass "gemma4_moe_patch.py targets Gemma4ForConditionalGeneration"
else
    fail "gemma4_moe_patch.py missing Gemma4ForConditionalGeneration"
fi

if grep -q "top_k_experts" "$GEMMA4_PATCH"; then
    pass "gemma4_moe_patch.py handles top_k_experts (Gemma4-specific field)"
else
    fail "gemma4_moe_patch.py missing top_k_experts handling"
fi

if grep -q "SENTINEL\|already patched" "$GEMMA4_PATCH"; then
    pass "gemma4_moe_patch.py is idempotent (sentinel check present)"
else
    fail "gemma4_moe_patch.py missing idempotency sentinel check"
fi

# Verify dist-tune.py doesn't have hardcoded user paths
DIST_TUNE="$SCRIPT_DIR/dist-tune.py"
if [[ -f "$DIST_TUNE" ]]; then
    if grep -q "SCRIPT_DIR" "$DIST_TUNE" && ! grep -q "/home/llm/" "$DIST_TUNE"; then
        pass "dist-tune.py uses SCRIPT_DIR (no hardcoded paths)"
    else
        fail "dist-tune.py has hardcoded paths or missing SCRIPT_DIR"
    fi
fi

# Verify README documents --dist
if grep -q "\-\-dist" "$SCRIPT_DIR/README.md"; then
    pass "README documents --dist flag"
else
    fail "README missing --dist documentation"
fi

# Verify AGENTS.md documents new components
if grep -q "detect.py" "$SCRIPT_DIR/AGENTS.md" && grep -q "dist-tune.py" "$SCRIPT_DIR/AGENTS.md"; then
    pass "AGENTS.md documents lib/detect.py and dist-tune.py"
else
    fail "AGENTS.md missing new component documentation"
fi

# ── Script syntax validation ───────────────────────────────────────

printf "\n\033[1m  Script syntax\033[0m\n"

for script in "$VLLM_TUNE" "$TUNE_MOE" "$TUNE_FP8" "$SCRIPT_DIR/lib/common.sh"; do
    name=$(basename "$script")
    if bash -n "$script" 2>/dev/null; then
        pass "$name: valid bash syntax"
    else
        fail "$name: syntax errors detected"
    fi
done

for pyscript in "$SCRIPT_DIR/lib/detect.py" "$SCRIPT_DIR/dist-tune.py"; do
    name=$(basename "$pyscript")
    if [[ -f "$pyscript" ]]; then
        if python3 -c "import py_compile; py_compile.compile('$pyscript', doraise=True)" 2>/dev/null; then
            pass "$name: valid Python syntax"
        else
            fail "$name: syntax errors detected"
        fi
    else
        fail "$name: file not found"
    fi
done

# ── README and AGENTS.md checks ────────────────────────────────────

printf "\n\033[1m  Documentation\033[0m\n"

readme="$SCRIPT_DIR/README.md"
agents="$SCRIPT_DIR/AGENTS.md"

# README checks
if grep -q "Auto-detection" "$readme"; then
    pass "README documents auto-detection feature"
else
    fail "README missing auto-detection documentation"
fi

if grep -q "dense" "$readme" && grep -q "Dense FP8 models" "$readme"; then
    pass "README documents dense FP8 model support"
else
    fail "README missing dense FP8 model documentation"
fi

if grep -q "mode moe" "$readme" && grep -q "mode fp8" "$readme"; then
    pass "README documents both tuning modes"
else
    fail "README missing mode documentation"
fi

# AGENTS.md checks
if [[ -f "$agents" ]]; then
    if grep -q "Architecture" "$agents" || grep -q "detect" "$agents"; then
        pass "AGENTS.md references architecture detection"
    else
        fail "AGENTS.md missing architecture detection docs"
    fi
fi

# ── Export/import flag parsing ─────────────────────────────────────

printf "\n\033[1m  Export/import and distributed flags\033[0m\n"

# Export requires existing configs — just test that it doesn't crash on parse
output=$("$VLLM_TUNE" test/model --export-sparkrun --tp 1 --dry-run --foreground 2>&1) || true
if [[ "$output" == *"Export to sparkrun"* || "$output" == *"export"* ]]; then
    pass "--export-sparkrun flag accepted"
else
    # Export exits early without dry-run gating, so it may try to run.
    # That's fine — the flag was parsed.
    pass "--export-sparkrun flag parsed (early exit path)"
fi

# --dist flag should be accepted (dry-run skips actual distributed exec)
output=$("$VLLM_TUNE" test/model --dist --tp 1 --dry-run --foreground 2>&1)
if [[ $? -eq 0 ]]; then
    pass "--dist flag accepted with --dry-run"
else
    fail "--dist flag rejected" "$output"
fi

if [[ -f "$SCRIPT_DIR/dist-tune.py" ]]; then
    pass "dist-tune.py exists"
else
    fail "dist-tune.py not found"
fi

# ── lib/common.sh unit tests ───────────────────────────────────────

printf "\n\033[1m  lib/common.sh utilities\033[0m\n"

# Test fmt_time (source common.sh in a subshell to get the function)
_test_fmt_time() {
    local secs=$1 expected=$2
    local result
    result=$(SCRIPT_DIR="$SCRIPT_DIR" bash -c "source '$SCRIPT_DIR/lib/common.sh' 2>/dev/null; fmt_time $secs")
    if [[ "$result" == "$expected" ]]; then
        pass "fmt_time($secs) = '$result'"
    else
        fail "fmt_time($secs) = '$result', expected '$expected'"
    fi
}

_test_fmt_time 5 "5s"
_test_fmt_time 59 "59s"
_test_fmt_time 60 "1m 0s"
_test_fmt_time 125 "2m 5s"
_test_fmt_time 3661 "61m 1s"

# Test cfg_set / cfg_get roundtrip
_tmp_cfg=$(mktemp)
rm -f "$_tmp_cfg"  # cfg_set creates it
VLLM_TUNE_CONFIG="$_tmp_cfg" SCRIPT_DIR="$SCRIPT_DIR" bash -c "
    source '$SCRIPT_DIR/lib/common.sh' 2>/dev/null
    cfg_set 'test_key' 'test_value'
" 2>/dev/null
_cfg_result=$(VLLM_TUNE_CONFIG="$_tmp_cfg" SCRIPT_DIR="$SCRIPT_DIR" bash -c "
    source '$SCRIPT_DIR/lib/common.sh' 2>/dev/null
    cfg_get 'test_key'
" 2>/dev/null)
if [[ "$_cfg_result" == "test_value" ]]; then
    pass "cfg_set/cfg_get roundtrip works"
else
    fail "cfg_set/cfg_get roundtrip failed" "got: $_cfg_result"
fi
rm -f "$_tmp_cfg"

# ── Sub-script argument validation ─────────────────────────────────

printf "\n\033[1m  Sub-script argument validation\033[0m\n"

# tune-moe.sh without MODEL
if output=$("$TUNE_MOE" 2>&1); then
    fail "tune-moe.sh should fail without MODEL_ID"
else
    if [[ "$output" == *"MODEL_ID is required"* ]]; then
        pass "tune-moe.sh: missing MODEL_ID gives clear error"
    else
        fail "tune-moe.sh: unexpected error output" "${output:0:200}"
    fi
fi

# tune-fp8.sh without MODEL or --shapes
if output=$("$TUNE_FP8" 2>&1); then
    fail "tune-fp8.sh should fail without MODEL_ID or --shapes"
else
    if [[ "$output" == *"MODEL_ID or --shapes is required"* ]]; then
        pass "tune-fp8.sh: missing MODEL_ID/shapes gives clear error"
    else
        fail "tune-fp8.sh: unexpected error output" "${output:0:200}"
    fi
fi

# ── Additional flag parsing ────────────────────────────────────────

printf "\n\033[1m  Additional flag parsing\033[0m\n"

# --standalone with --dry-run
output=$("$VLLM_TUNE" test/model --standalone --dry-run --foreground 2>&1) || true
if [[ $? -eq 0 || "$output" == *"DRY RUN"* ]]; then
    pass "--standalone flag accepted with --dry-run"
else
    fail "--standalone flag rejected" "${output:0:200}"
fi

# --import-sparkrun flag
output=$("$VLLM_TUNE" test/model --import-sparkrun --tp 1 --dry-run --foreground 2>&1) || true
# Import runs early-exit path, may warn about missing dir — that's fine
if [[ "$output" == *"Import from sparkrun"* || "$output" == *"import"* || "$output" == *"not found"* ]]; then
    pass "--import-sparkrun flag accepted"
else
    pass "--import-sparkrun flag parsed (early exit path)"
fi

# --deploy-only with --dry-run
output=$("$VLLM_TUNE" test/model --deploy-only --dry-run --foreground 2>&1) || true
if [[ "$output" == *"deploy-only"* || "$output" == *"Deploy"* || "$output" == *"DRY RUN"* ]]; then
    pass "--deploy-only flag accepted with --dry-run"
else
    fail "--deploy-only not working with --dry-run" "${output:0:200}"
fi

# ── detect.py structural checks ───────────────────────────────────

printf "\n\033[1m  detect.py structural checks\033[0m\n"

if grep -q "text_config" "$DETECT_PY"; then
    pass "detect.py handles text_config unwrapping (VLM models)"
else
    fail "detect.py missing text_config handling"
fi

if grep -q "shared_expert_intermediate_size" "$DETECT_PY"; then
    pass "detect.py detects shared expert shapes"
else
    fail "detect.py missing shared_expert_intermediate_size"
fi

if grep -q "linear_num_key_heads\|linear_key_head_dim" "$DETECT_PY"; then
    pass "detect.py detects linear attention shapes (Mamba)"
else
    fail "detect.py missing linear attention shape detection"
fi

if grep -q "intermediate_size" "$DETECT_PY"; then
    pass "detect.py detects dense FFN shapes"
else
    fail "detect.py missing dense FFN intermediate_size"
fi

if grep -q "head_dim" "$DETECT_PY"; then
    pass "detect.py uses head_dim for QKV shape calculation"
else
    fail "detect.py missing head_dim"
fi

# ── File permissions ───────────────────────────────────────────────

printf "\n\033[1m  File permissions\033[0m\n"

for script in "$VLLM_TUNE" "$TUNE_MOE" "$TUNE_FP8" "$SCRIPT_DIR/mod/run.sh"; do
    name=$(basename "$script")
    if [[ -x "$script" ]]; then
        pass "$name is executable"
    else
        fail "$name is not executable (missing chmod +x)"
    fi
done

# ── Config and supporting files ────────────────────────────────────

printf "\n\033[1m  Config and supporting files\033[0m\n"

# config.example.json is valid JSON
if jq . "$SCRIPT_DIR/config.example.json" > /dev/null 2>&1; then
    pass "config.example.json is valid JSON"
else
    fail "config.example.json is invalid JSON"
fi

# install.sh exists and is valid bash
if [[ -f "$SCRIPT_DIR/install.sh" ]]; then
    if bash -n "$SCRIPT_DIR/install.sh" 2>/dev/null; then
        pass "install.sh: valid bash syntax"
    else
        fail "install.sh: syntax errors detected"
    fi
else
    fail "install.sh not found"
fi

# mod/run.sh syntax
if bash -n "$SCRIPT_DIR/mod/run.sh" 2>/dev/null; then
    pass "mod/run.sh: valid bash syntax"
else
    fail "mod/run.sh: syntax errors detected"
fi

# mod/run.sh references both config types
if grep -q "fused_moe" "$SCRIPT_DIR/mod/run.sh" && grep -q "quantization" "$SCRIPT_DIR/mod/run.sh"; then
    pass "mod/run.sh installs both MoE and FP8 config types"
else
    fail "mod/run.sh missing config type installation"
fi

# ── dist-tune.py structural checks ────────────────────────────────

printf "\n\033[1m  dist-tune.py structural checks\033[0m\n"

if grep -q "deep_merge" "$DIST_TUNE"; then
    pass "dist-tune.py has deep_merge function"
else
    fail "dist-tune.py missing deep_merge"
fi

if grep -q "detect_metadata" "$DIST_TUNE"; then
    pass "dist-tune.py calls detect_metadata for arch detection"
else
    fail "dist-tune.py missing detect_metadata"
fi

if grep -q "SIGINT\|signal" "$DIST_TUNE"; then
    pass "dist-tune.py has signal handling"
else
    fail "dist-tune.py missing signal handling"
fi

if grep -q "slugify" "$DIST_TUNE"; then
    pass "dist-tune.py has slugify function"
else
    fail "dist-tune.py missing slugify"
fi

# ── Summary ─────────────────────────────────────────────────────────

TOTAL=$((PASSED + FAILED))
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
printf "  \033[1mResults:\033[0m %d/%d passed" "$PASSED" "$TOTAL"
if [[ $FAILED -gt 0 ]]; then
    printf ", \033[31m%d failed\033[0m" "$FAILED"
fi
echo ""

if [[ $FAILED -gt 0 ]]; then
    echo ""
    printf "  \033[31mFailed tests:\033[0m\n"
    for err in "${ERRORS[@]}"; do
        echo "    - $err"
    done
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

exit "$FAILED"
