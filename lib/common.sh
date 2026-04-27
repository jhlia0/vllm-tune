#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# lib/common.sh — Shared utilities for vllm-tune tuning scripts
# ─────────────────────────────────────────────────────────────────────
#
# Sourced by tune-moe.sh and tune-fp8.sh to avoid duplicating:
#   - Container validation and tini detection
#   - jq dependency check
#   - Retry loop with per-item crash-safe backup
#   - Cumulative JSON merge (jq -s '.[0] * .[1]')
#   - Zombie detection and cleanup
#   - Cache-clearing between rounds
#   - Report generation (Markdown)
#   - Final summary output
#
# Usage (from a tuning script):
#   SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
#   source "$SCRIPT_DIR/lib/common.sh"
#
# The sourcing script must define these before calling shared functions:
#   CONTAINER       — Docker container name (default: vllm_node)
#   CONFIGS_DIR     — Where to write final JSON configs
#   HOST_BACKUP_DIR — Host-side incremental backup directory
#   CONTAINER_SAVE_DIR — In-container temp dir for benchmark output
#   MAX_RETRIES     — Max retry count per tuning item (default: 1)
# ─────────────────────────────────────────────────────────────────────

# ── Signal handling ─────────────────────────────────────────────────

# Ctrl+C should cancel the entire tuning run, not just skip one batch.
# Without this trap, SIGINT kills the docker exec subprocess but the
# bash loop treats it as a failure and moves to the next iteration.
CANCELLED=false
_on_sigint() {
    CANCELLED=true
    echo ""
    echo "  ⛔ Cancelled by user (Ctrl+C)"
    exit 130
}
trap _on_sigint INT TERM

# ── Formatting ──────────────────────────────────────────────────────

fmt_time() {
    local secs=$1
    if [[ $secs -ge 60 ]]; then
        echo "$((secs/60))m $((secs%60))s"
    else
        echo "${secs}s"
    fi
}

# ── Config file ─────────────────────────────────────────────────────

# Central config file — lives next to the scripts, gitignored.
VLLM_TUNE_CONFIG="${VLLM_TUNE_CONFIG:-${SCRIPT_DIR:-$(cd "$(dirname "$0")" && pwd)}/config.json}"

# Read a value from config.json.  Returns empty string if key missing.
# Usage: cfg_get "spark_vllm_docker"
#        cfg_get "sync_targets" (returns JSON array)
cfg_get() {
    local key="$1"
    [[ -f "$VLLM_TUNE_CONFIG" ]] || return 0
    jq -r --arg k "$key" '.[$k] // empty' "$VLLM_TUNE_CONFIG" 2>/dev/null
}

# Read an array from config.json as newline-separated values.
# Usage: mapfile -t targets < <(cfg_get_array "sync_targets")
cfg_get_array() {
    local key="$1"
    [[ -f "$VLLM_TUNE_CONFIG" ]] || return 0
    jq -r --arg k "$key" '.[$k] // [] | .[]' "$VLLM_TUNE_CONFIG" 2>/dev/null
}

# Write a scalar value to config.json (creates file if needed).
# Usage: cfg_set "container" "vllm_node"
cfg_set() {
    local key="$1" value="$2"
    if [[ -f "$VLLM_TUNE_CONFIG" ]]; then
        local tmp="${VLLM_TUNE_CONFIG}.tmp"
        jq --arg k "$key" --arg v "$value" '.[$k] = $v' "$VLLM_TUNE_CONFIG" > "$tmp" \
            && mv "$tmp" "$VLLM_TUNE_CONFIG"
    else
        jq -n --arg k "$key" --arg v "$value" '{($k): $v}' > "$VLLM_TUNE_CONFIG"
    fi
}

# Write a numeric value to config.json.
# Usage: cfg_set_num "tp" 2
cfg_set_num() {
    local key="$1" value="$2"
    if [[ -f "$VLLM_TUNE_CONFIG" ]]; then
        local tmp="${VLLM_TUNE_CONFIG}.tmp"
        jq --arg k "$key" --argjson v "$value" '.[$k] = $v' "$VLLM_TUNE_CONFIG" > "$tmp" \
            && mv "$tmp" "$VLLM_TUNE_CONFIG"
    else
        jq -n --arg k "$key" --argjson v "$value" '{($k): $v}' > "$VLLM_TUNE_CONFIG"
    fi
}

# Write an array of strings to config.json.
# Usage: cfg_set_array "sync_targets" "/path/a" "/path/b"
cfg_set_array() {
    local key="$1"; shift
    local json_array
    json_array=$(printf '%s\n' "$@" | jq -R . | jq -s .)
    if [[ -f "$VLLM_TUNE_CONFIG" ]]; then
        local tmp="${VLLM_TUNE_CONFIG}.tmp"
        jq --arg k "$key" --argjson v "$json_array" '.[$k] = $v' "$VLLM_TUNE_CONFIG" > "$tmp" \
            && mv "$tmp" "$VLLM_TUNE_CONFIG"
    else
        jq -n --arg k "$key" --argjson v "$json_array" '{($k): $v}' > "$VLLM_TUNE_CONFIG"
    fi
}

# ── Validation ──────────────────────────────────────────────────────

# Verify the target Docker container is running.
# Exits with error if not found.
check_container() {
    if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$CONTAINER"; then
        echo "Error: Container '$CONTAINER' is not running." >&2
        echo "Start it first with: ./start-vllm.sh <model>" >&2
        exit 1
    fi
}

# Detect tini/docker-init inside the container for use as a subreaper.
# Without this, benchmark child processes become zombies under PID 1
# (sleep infinity) because sleep doesn't call wait(). tini -s sets
# PR_SET_CHILD_SUBREAPER so orphaned grandchildren get reparented to
# tini (which reaps them) instead of PID 1.
#
# Sets global: INIT_WRAPPER (empty string if no init found)
detect_init_wrapper() {
    INIT_WRAPPER=""
    for init_bin in /usr/bin/tini /sbin/tini /usr/bin/docker-init /sbin/docker-init; do
        if docker exec "$CONTAINER" test -x "$init_bin" 2>/dev/null; then
            INIT_WRAPPER="$init_bin -s --"
            return
        fi
    done
}

# Verify jq is installed on the host (required for JSON config merging).
check_jq() {
    if ! command -v jq &>/dev/null; then
        echo "Error: jq is required for merging configs. Install it with: sudo apt install jq" >&2
        exit 1
    fi
}

# Run all pre-flight checks: container, init wrapper, jq.
# Creates CONFIGS_DIR and HOST_BACKUP_DIR if they don't exist.
preflight() {
    check_container
    detect_init_wrapper
    check_jq
    mkdir -p "$HOST_BACKUP_DIR" "$CONFIGS_DIR"
}

# ── Zombie tracking ────────────────────────────────────────────────

# Count zombie processes parented by the container's PID 1.
# Returns the count via stdout.
count_zombies() {
    local container_pid count
    container_pid=$(docker inspect --format '{{.State.Pid}}' "$CONTAINER" 2>/dev/null) || { echo 0; return; }
    count=$(ps --ppid "$container_pid" -o stat= 2>/dev/null | grep -c '^Z' || true)
    echo "${count:-0}"
}

# If zombies have accumulated, restart the container to reap them.
cleanup_zombies() {
    local zombies
    zombies=$(count_zombies)
    if [[ $zombies -gt 0 ]]; then
        echo
        echo "  🧟 $zombies zombie process(es) detected — restarting container to reap..."
        docker restart "$CONTAINER" >/dev/null 2>&1 && echo "  ✅ Container restarted, zombies cleared." \
            || echo "  ⚠ Container restart failed — zombies will persist until manual restart." >&2
    fi
}

# Clear OS page/slab caches between tuning rounds to prevent memory
# pressure from skewing results.
#
# Configurable via environment:
#   DROP_CACHES_CMD — Command to drop caches (default: sync && echo 3 > /proc/sys/vm/drop_caches)
#   PEER_NODES      — Space-separated list of SSH-reachable peer nodes to also clear
clear_caches() {
    local drop_cmd="${DROP_CACHES_CMD:-sync && echo 3 > /proc/sys/vm/drop_caches}"

    echo "  🧹 Clearing memory caches..."
    sudo bash -c "$drop_cmd" 2>/dev/null && echo "    ✓ $(hostname)" || true

    for peer in ${PEER_NODES:-}; do
        ssh -o ConnectTimeout=5 -o BatchMode=yes "$peer" \
            "sudo bash -c '$drop_cmd'" 2>/dev/null \
            && echo "    ✓ $peer" \
            || echo "    ⚠ $peer (skipped)" >&2
    done
}

# ── Crash-safe JSON merge ───────────────────────────────────────────

# After each tuning iteration, copy results from the container and
# merge them into both the cumulative backup and the production config
# directory. This ensures no progress is lost on crash.
#
# Arguments:
#   $1 — Subdirectory name inside CONTAINER_SAVE_DIR (e.g. "moe-configs-tuning")
#
# Globals used: CONTAINER, CONTAINER_SAVE_DIR, HOST_BACKUP_DIR, CONFIGS_DIR
merge_results() {
    local subdir_name="$1"
    local staging_dir
    staging_dir=$(mktemp -d)

    docker cp "$CONTAINER:$CONTAINER_SAVE_DIR/" "$staging_dir/" 2>/dev/null || true

    for sf in "$staging_dir/$subdir_name"/*.json "$staging_dir"/*.json; do
        [[ -f "$sf" ]] || continue
        local base cumulative target
        base=$(basename "$sf")
        cumulative="$HOST_BACKUP_DIR/$base"
        target="$CONFIGS_DIR/$base"

        # Merge into cumulative backup (additive — never loses keys)
        if [[ -f "$cumulative" ]]; then
            jq -s '.[0] * .[1]' "$cumulative" "$sf" > "$cumulative.tmp" \
                && mv "$cumulative.tmp" "$cumulative"
        else
            cp "$sf" "$cumulative"
        fi

        # Also merge cumulative into production configs
        if [[ -f "$target" ]]; then
            jq -s '.[0] * .[1]' "$target" "$cumulative" > "$target.tmp" \
                && mv "$target.tmp" "$target"
        else
            cp "$cumulative" "$target"
        fi
    done
    rm -rf "$staging_dir"
    echo "  📦 Saved to $CONFIGS_DIR"
}

# ── Retry loop ──────────────────────────────────────────────────────

# Run a tuning command with retry logic.
# Returns 0 if any attempt succeeds, 1 if all fail.
#
# Arguments:
#   $1    — Human-readable label for log messages (e.g. "batch_size=64")
#   $2... — The command to execute
run_with_retry() {
    local label="$1"; shift
    local attempt

    for attempt in $(seq 0 "$MAX_RETRIES"); do
        if [[ $attempt -gt 0 ]]; then
            printf "  ⟳ Retry %d/%d for %s...\n" "$attempt" "$MAX_RETRIES" "$label"
            sleep 2
        fi

        if "$@"; then
            return 0
        else
            if [[ $attempt -lt $MAX_RETRIES ]]; then
                echo "  ⚠ Attempt $((attempt + 1)) failed, will retry..."
            fi
        fi
    done
    return 1
}

# ── Post-round housekeeping ─────────────────────────────────────────

# Run after each tuning item: clear caches and check for zombies.
post_round() {
    clear_caches

    local zombies
    zombies=$(count_zombies)
    if [[ $zombies -gt 0 ]]; then
        echo "  💀 $zombies zombie(s) in container (will clean up at end)"
    fi
    echo
}

# ── Report generation ───────────────────────────────────────────────

# Generate a Markdown tuning report.
#
# Arguments:
#   $1 — Report title (e.g. "MoE Tuning Report")
#   $2 — Item column header (e.g. "Batch Size" or "Shape (N,K)")
#   $3 — Total tuning time in seconds
#   $4 — Space-separated list of all items tuned
#   $5 — Space-separated list of failed items
#   $6 — Re-run command template for failed items
#
# Globals used: CONFIGS_DIR, MODEL, TP, CONTAINER
# Reads from associative array: TIMINGS (item -> elapsed seconds)
#
# Output: Writes to $CONFIGS_DIR/tuning-report.md
generate_report() {
    local title="$1" item_header="$2" total_secs="$3"
    local items_str="$4" failed_str="$5" rerun_cmd="$6"

    local report_file="$CONFIGS_DIR/tuning-report.md"
    local timestamp
    timestamp=$(date -Iseconds)

    read -ra items <<< "$items_str"
    read -ra failed <<< "$failed_str"
    local total=${#items[@]}
    local succeeded=$((total - ${#failed[@]}))

    cat > "$report_file" <<EOF
# $title

Generated: $timestamp

## Configuration

| Parameter | Value |
|-----------|-------|
EOF

    [[ -n "${MODEL:-}" ]] && echo "| Model | \`$MODEL\` |" >> "$report_file"
    cat >> "$report_file" <<EOF
| TP size | ${TP:-2} |
| Container | $CONTAINER |
| Total time | $(fmt_time "$total_secs") |

## Results

| $item_header | Status | Time |
|$(printf -- '-%.0s' {1..20})|--------|------|
EOF

    for item in "${items[@]}"; do
        local elapsed=${TIMINGS[$item]:-0}
        local status="✅ OK"
        for fb in "${failed[@]+${failed[@]}}"; do
            if [[ "$fb" == "$item" ]]; then
                status="❌ Failed"
                break
            fi
        done
        echo "| $item | $status | $(fmt_time "$elapsed") |" >> "$report_file"
    done

    cat >> "$report_file" <<EOF

## Summary

- **Succeeded:** $succeeded/$total
- **Failed:** ${#failed[@]}/$total$(
    [[ ${#failed[@]} -gt 0 ]] && echo " (${failed[*]})"
)
EOF

    if [[ ${#failed[@]} -gt 0 && -n "$rerun_cmd" ]]; then
        cat >> "$report_file" <<EOF

## Re-run failed

\`\`\`bash
$rerun_cmd
\`\`\`
EOF
    fi

    echo "  📝 Report: $report_file"
}

# ── Final summary ──────────────────────────────────────────────────

# Print a styled summary banner at the end of tuning.
#
# Arguments:
#   $1 — Count of succeeded items
#   $2 — Total item count
#   $3 — Total elapsed seconds
#   $4 — Item noun (e.g. "batch sizes" or "shapes")
#   $5 — Space-separated list of failed items (may be empty)
#   $6 — Re-run command for failed items (may be empty)
print_summary() {
    local succeeded="$1" total="$2" elapsed="$3" noun="$4"
    local failed_str="$5" rerun_cmd="$6"

    read -ra failed <<< "$failed_str"

    echo
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    printf "  \033[1mDone!\033[0m  "
    printf "%d/%d %s tuned in %s" "$succeeded" "$total" "$noun" "$(fmt_time "$elapsed")"
    if [[ ${#failed[@]} -gt 0 ]]; then
        printf "\n  \033[31mFailed:\033[0m %s" "${failed[*]}"
        [[ -n "$rerun_cmd" ]] && printf "\n  Re-run: %s" "$rerun_cmd"
    fi
    echo
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}
