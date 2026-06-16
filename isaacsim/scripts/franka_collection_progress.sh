#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

RUN_PREFIX="franka_20env_500epi_full"
SESSION_PREFIX="franka"
OUTPUT_DIR="scripts/runs/collected_data"
TAIL_LINES=3
SHOW_GPU=1
COUNT_IMAGES=0

usage() {
    cat <<'EOF'
Usage:
  scripts/franka_collection_progress.sh [options]

Options:
  --run-prefix NAME      Prefix used by start_franka_multigpu_screen.sh.
                          Default: franka_20env_500epi_full
  --session-prefix NAME  Screen session prefix. Default: franka
  --output-dir DIR       Collector output dir. Default: scripts/runs/collected_data
  --tail-lines N         Recent progress lines per shard. Default: 3
  --count-images         Also count camera/mask/depth files. This can be slower on large runs.
  --no-gpu              Skip nvidia-smi summary.
  -h, --help             Show this help.
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

last_match() {
    local pattern="$1"
    local file="$2"
    if command -v rg >/dev/null 2>&1; then
        rg -e "$pattern" "$file" | tail -n "$TAIL_LINES" || true
    else
        grep -E "$pattern" "$file" | tail -n "$TAIL_LINES" || true
    fi
}

single_last_match() {
    local pattern="$1"
    local file="$2"
    if command -v rg >/dev/null 2>&1; then
        rg -e "$pattern" "$file" | tail -n 1 || true
    else
        grep -E "$pattern" "$file" | tail -n 1 || true
    fi
}

count_files() {
    local dir="$1"
    local glob="$2"
    if [[ -d "$dir" ]]; then
        find "$dir" -maxdepth 1 -type f -name "$glob" | wc -l | tr -d ' '
    else
        printf '0'
    fi
}

count_real_schema_sessions() {
    local dir="$1"
    if [[ -d "$dir" ]]; then
        find "$dir" -mindepth 1 -maxdepth 1 -type d -name 'session_ep*_env*' |
            while IFS= read -r session_dir; do
                [[ -s "$session_dir/frames.csv" ]] && basename "$session_dir"
            done |
            wc -l |
            tr -d ' '
    else
        printf '0'
    fi
}

count_real_schema_episodes() {
    local dir="$1"
    if [[ -d "$dir" ]]; then
        find "$dir" -mindepth 1 -maxdepth 1 -type d -name 'session_ep*_env*' |
            while IFS= read -r session_dir; do
                [[ -s "$session_dir/frames.csv" ]] && basename "$session_dir"
            done |
            sed -E 's/^session_ep([0-9]+)_env[0-9]+$/\1/' |
            sort -u |
            wc -l |
            tr -d ' '
    else
        printf '0'
    fi
}

percent() {
    local done="$1"
    local total="$2"
    if ((total <= 0)); then
        printf 'n/a'
    else
        awk -v d="$done" -v t="$total" 'BEGIN { printf "%.1f%%", (d * 100.0 / t) }'
    fi
}

config_value() {
    local config="$1"
    local key="$2"
    python3 - "$config" "$key" <<'PY'
import json
import sys
path, key = sys.argv[1], sys.argv[2]
try:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    value = data.get(key, "")
    print(value)
except Exception:
    print("")
PY
}

while (($#)); do
    case "$1" in
        --run-prefix)
            RUN_PREFIX="${2:?missing value for --run-prefix}"
            shift 2
            ;;
        --session-prefix)
            SESSION_PREFIX="${2:?missing value for --session-prefix}"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="${2:?missing value for --output-dir}"
            shift 2
            ;;
        --tail-lines)
            TAIL_LINES="${2:?missing value for --tail-lines}"
            shift 2
            ;;
        --count-images)
            COUNT_IMAGES=1
            shift
            ;;
        --no-gpu)
            SHOW_GPU=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

[[ "$TAIL_LINES" =~ ^[0-9]+$ ]] || die "--tail-lines must be an integer"

cd "$ISAAC_ROOT"
if [[ "$OUTPUT_DIR" = /* ]]; then
    OUTPUT_PATH="$OUTPUT_DIR"
else
    OUTPUT_PATH="$ISAAC_ROOT/$OUTPUT_DIR"
fi
SCREEN_LIST="$(screen -ls 2>/dev/null || true)"

printf 'Franka collection progress\n'
printf '  time: %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')"
printf '  run_prefix: %s\n' "$RUN_PREFIX"
printf '  output_dir: %s\n' "$OUTPUT_PATH"

printf '\nScreen sessions:\n'
if command -v rg >/dev/null 2>&1; then
    printf '%s\n' "$SCREEN_LIST" | rg "${SESSION_PREFIX}_shard[0-9]+_gpu[0-9]+" || printf '  no matching screen sessions\n'
else
    printf '%s\n' "$SCREEN_LIST" | grep -E "${SESSION_PREFIX}_shard[0-9]+_gpu[0-9]+" || printf '  no matching screen sessions\n'
fi

if ((SHOW_GPU)) && command -v nvidia-smi >/dev/null 2>&1; then
    printf '\nGPU summary:\n'
    gpu_summary="$(
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
            --format=csv,noheader,nounits 2>&1
    )" || {
        printf '  nvidia-smi query failed: %s\n' "$(printf '%s\n' "$gpu_summary" | head -n 1)"
        gpu_summary=""
    }
    if [[ -n "$gpu_summary" ]]; then
        printf '%s\n' "$gpu_summary" |
            awk -F', ' '{ printf "  gpu%s %s: util=%s%% mem=%s/%s MiB power=%s W temp=%s C\n", $1, $2, $3, $4, $5, $6, $7 }'
    fi
fi

shopt -s nullglob
logs=("$OUTPUT_PATH/${RUN_PREFIX}"_shard*_gpu*.log)
shopt -u nullglob

if ((${#logs[@]} == 0)); then
    printf '\nNo matching logs found for prefix: %s\n' "$RUN_PREFIX"
    exit 0
fi

total_done=0
total_target=0
total_env_done=0
total_env_target=0

printf '\nShards:\n'
for log in "${logs[@]}"; do
    run_name="$(basename "$log" .log)"
    data_dir="$OUTPUT_PATH/$run_name"
    config="$data_dir/collection_config.json"
    episodes=""
    num_envs=""

    if [[ -f "$config" ]]; then
        episodes="$(config_value "$config" "episodes")"
        num_envs="$(config_value "$config" "num_envs")"
    fi

    if [[ -z "$episodes" ]]; then
        header="$(single_last_match 'num_envs: [0-9]+, episodes: [0-9]+' "$log")"
        episodes="$(printf '%s\n' "$header" | sed -E 's/.*episodes: ([0-9]+).*/\1/')"
        [[ "$episodes" =~ ^[0-9]+$ ]] || episodes=0
    fi
    if [[ -z "$num_envs" ]]; then
        header="$(single_last_match 'num_envs: [0-9]+, episodes: [0-9]+' "$log")"
        num_envs="$(printf '%s\n' "$header" | sed -E 's/.*num_envs: ([0-9]+).*/\1/')"
        [[ "$num_envs" =~ ^[0-9]+$ ]] || num_envs=0
    fi

    legacy_episode_count="$(count_files "$data_dir" 'episode_*.npz')"
    real_episode_count="$(count_real_schema_episodes "$data_dir")"
    real_session_count="$(count_real_schema_sessions "$data_dir")"
    if ((real_session_count > 0)); then
        episode_count="$real_episode_count"
        env_done="$real_session_count"
    else
        episode_count="$legacy_episode_count"
        env_done=$((legacy_episode_count * num_envs))
    fi
    problem_pattern='Traceback|Exception|Fatal|(^|[^A-Za-z])ERROR([^A-Za-z]|$)|\[Error\]'
    latest="$(single_last_match "Queued episode [0-9]+|Collection complete|${problem_pattern}" "$log")"
    recent="$(last_match "Episode [0-9]+ step [0-9]+|Episode [0-9]+: reset|Queued episode [0-9]+|Collection complete|${problem_pattern}" "$log")"

    screen_status="unknown"
    if [[ "$run_name" =~ shard([0-9]+)_gpu([0-9]+) ]]; then
        session="${SESSION_PREFIX}_shard${BASH_REMATCH[1]}_gpu${BASH_REMATCH[2]}"
        if [[ "$SCREEN_LIST" == *".${session}"* ]]; then
            screen_status="running (${session})"
        else
            screen_status="not running (${session})"
        fi
    fi

    progress_total="$episodes"
    [[ "$progress_total" =~ ^[0-9]+$ ]] || progress_total=0
    [[ "$num_envs" =~ ^[0-9]+$ ]] || num_envs=0

    total_done=$((total_done + episode_count))
    total_target=$((total_target + progress_total))
    total_env_done=$((total_env_done + env_done))
    total_env_target=$((total_env_target + progress_total * num_envs))

    printf '\n  %s\n' "$run_name"
    printf '    screen: %s\n' "$screen_status"
    printf '    data: %s\n' "$data_dir"
    printf '    episodes: %s/%s (%s), envs=%s, env-sessions=%s/%s (%s)\n' \
        "$episode_count" "$progress_total" "$(percent "$episode_count" "$progress_total")" "$num_envs" \
        "$env_done" "$((progress_total * num_envs))" "$(percent "$env_done" "$((progress_total * num_envs))")"
    if [[ -n "$latest" ]]; then
        printf '    latest: %s\n' "$latest"
    fi
    if ((COUNT_IMAGES)); then
        printf '    files: world=%s wrist=%s world_mask=%s wrist_mask=%s wrist_depth=%s\n' \
            "$(count_files "$data_dir/world_camera" '*.npy')" \
            "$(count_files "$data_dir/wrist_camera" '*.npy')" \
            "$(count_files "$data_dir/world_camera_mask" '*.npy')" \
            "$(count_files "$data_dir/wrist_camera_mask" '*.npy')" \
            "$(count_files "$data_dir/wrist_camera_depth" '*.npy')"
    fi
    if [[ -n "$recent" ]]; then
        printf '    recent:\n'
        printf '%s\n' "$recent" | sed 's/^/      /'
    fi
done

printf '\nAggregate:\n'
printf '  shard episodes: %s/%s (%s)\n' "$total_done" "$total_target" "$(percent "$total_done" "$total_target")"
printf '  env-episodes: %s/%s (%s)\n' "$total_env_done" "$total_env_target" "$(percent "$total_env_done" "$total_env_target")"
