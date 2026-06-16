#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REAL_ISAAC_ROOT="${REAL_ISAAC_ROOT:-/home/james/isaacsim}"

TOTAL_ENVS=20
EPISODES=500
TARGET_ENV_EPISODES=0
GPUS="0,1"
SHARDS_PER_GPU=2
RUN_PREFIX=""
SESSION_PREFIX="franka"
OUTPUT_DIR="scripts/runs/collected_data"
ENV_USD="$REAL_ISAAC_ROOT/USDFiles/franka_env.usd"
SAVE_WORKERS=8
MAX_PENDING_SAVES=2048
STAGGER_SECONDS=2
CAMERA_WIDTH=848
CAMERA_HEIGHT=480
CAMERA_WARMUP_RENDER_STEPS=10
DRY_RUN=0
EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Usage:
  scripts/start_franka_multigpu_screen.sh [options] [-- extra collector args]

Defaults match the current full-data throughput run:
  20 total envs, 500 episodes per shard, GPUs 0,1, 2 shards per GPU.

Options:
  --total-envs N          Total parallel environments split across all shards. Default: 20
  --episodes N           Episodes per shard. Default: 500
  --target-env-episodes N Total env-episodes to collect. Overrides --episodes using ceil(N / total-envs).
  --gpus LIST            Comma-separated Isaac GPU indices. Default: 0,1
  --shards-per-gpu N     Screen/Isaac processes per GPU. Default: 2
  --run-prefix NAME      Output/log prefix. Default: franka_${total_envs}env_${episodes}epi_full
  --session-prefix NAME  Screen session prefix. Default: franka
  --output-dir DIR       Collector output dir. Default: scripts/runs/collected_data
  --env-usd PATH          Environment USD. Default: /home/james/isaacsim/USDFiles/franka_env.usd
  --save-workers N       Async save worker threads per shard. Default: 8
  --max-pending-saves N  Async save queue limit per shard. Default: 2048
  --stagger-seconds N    Delay between shard launches. Default: 2
  --camera-width N        World and wrist camera width. Default: 848
  --camera-height N       World and wrist camera height. Default: 480
  --camera-warmup-steps N Rendered camera warmup frames. Default: 10
  --dry-run              Print commands without starting screen sessions.
  -h, --help             Show this help.

Examples:
  scripts/start_franka_multigpu_screen.sh
  scripts/start_franka_multigpu_screen.sh --total-envs 24 --gpus 0,1 --shards-per-gpu 3
  scripts/start_franka_multigpu_screen.sh --target-env-episodes 10000 --output-dir /media/data
  scripts/start_franka_multigpu_screen.sh -- --steps-per-episode 360 --save-image-interval 10
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

while (($#)); do
    case "$1" in
        --total-envs)
            TOTAL_ENVS="${2:?missing value for --total-envs}"
            shift 2
            ;;
        --episodes)
            EPISODES="${2:?missing value for --episodes}"
            shift 2
            ;;
        --target-env-episodes)
            TARGET_ENV_EPISODES="${2:?missing value for --target-env-episodes}"
            shift 2
            ;;
        --gpus)
            GPUS="${2:?missing value for --gpus}"
            shift 2
            ;;
        --shards-per-gpu)
            SHARDS_PER_GPU="${2:?missing value for --shards-per-gpu}"
            shift 2
            ;;
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
        --env-usd)
            ENV_USD="${2:?missing value for --env-usd}"
            shift 2
            ;;
        --save-workers)
            SAVE_WORKERS="${2:?missing value for --save-workers}"
            shift 2
            ;;
        --max-pending-saves)
            MAX_PENDING_SAVES="${2:?missing value for --max-pending-saves}"
            shift 2
            ;;
        --stagger-seconds)
            STAGGER_SECONDS="${2:?missing value for --stagger-seconds}"
            shift 2
            ;;
        --camera-width)
            CAMERA_WIDTH="${2:?missing value for --camera-width}"
            shift 2
            ;;
        --camera-height)
            CAMERA_HEIGHT="${2:?missing value for --camera-height}"
            shift 2
            ;;
        --camera-warmup-steps)
            CAMERA_WARMUP_RENDER_STEPS="${2:?missing value for --camera-warmup-steps}"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            EXTRA_ARGS=("$@")
            break
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

[[ "$TOTAL_ENVS" =~ ^[0-9]+$ ]] || die "--total-envs must be an integer"
[[ "$EPISODES" =~ ^[0-9]+$ ]] || die "--episodes must be an integer"
[[ "$TARGET_ENV_EPISODES" =~ ^[0-9]+$ ]] || die "--target-env-episodes must be an integer"
[[ "$SHARDS_PER_GPU" =~ ^[0-9]+$ ]] || die "--shards-per-gpu must be an integer"
[[ "$SAVE_WORKERS" =~ ^[0-9]+$ ]] || die "--save-workers must be an integer"
[[ "$MAX_PENDING_SAVES" =~ ^[0-9]+$ ]] || die "--max-pending-saves must be an integer"
[[ "$STAGGER_SECONDS" =~ ^[0-9]+$ ]] || die "--stagger-seconds must be an integer"
[[ "$CAMERA_WIDTH" =~ ^[0-9]+$ ]] || die "--camera-width must be an integer"
[[ "$CAMERA_HEIGHT" =~ ^[0-9]+$ ]] || die "--camera-height must be an integer"
[[ "$CAMERA_WARMUP_RENDER_STEPS" =~ ^[0-9]+$ ]] || die "--camera-warmup-steps must be an integer"
((TOTAL_ENVS > 0)) || die "--total-envs must be > 0"
((EPISODES > 0)) || die "--episodes must be > 0"
((SHARDS_PER_GPU > 0)) || die "--shards-per-gpu must be > 0"
((SAVE_WORKERS > 0)) || die "--save-workers must be > 0"
((MAX_PENDING_SAVES > 0)) || die "--max-pending-saves must be > 0"
((CAMERA_WIDTH > 0)) || die "--camera-width must be > 0"
((CAMERA_HEIGHT > 0)) || die "--camera-height must be > 0"
[[ -f "$ENV_USD" ]] || die "environment USD not found: $ENV_USD"

if ((TARGET_ENV_EPISODES > 0)); then
    EPISODES=$(((TARGET_ENV_EPISODES + TOTAL_ENVS - 1) / TOTAL_ENVS))
fi

IFS=',' read -r -a GPU_IDS <<< "$GPUS"
(( ${#GPU_IDS[@]} > 0 )) || die "--gpus must contain at least one GPU index"
for gpu in "${GPU_IDS[@]}"; do
    [[ "$gpu" =~ ^[0-9]+$ ]] || die "invalid GPU index in --gpus: $gpu"
done

RUN_PREFIX="${RUN_PREFIX:-franka_${TOTAL_ENVS}env_${EPISODES}epi_full}"
TOTAL_SHARDS=$((${#GPU_IDS[@]} * SHARDS_PER_GPU))
((TOTAL_SHARDS > 0)) || die "computed shard count is zero"

BASE_ENVS=$((TOTAL_ENVS / TOTAL_SHARDS))
EXTRA_ENVS=$((TOTAL_ENVS % TOTAL_SHARDS))
if ((BASE_ENVS == 0)); then
    die "--total-envs is smaller than shard count (${TOTAL_SHARDS}); reduce --shards-per-gpu or GPU count"
fi

command -v screen >/dev/null 2>&1 || die "screen is not installed or not in PATH"
[[ -x "$SCRIPT_DIR/run_franka_collection_quiet.sh" ]] || die "missing executable: $SCRIPT_DIR/run_franka_collection_quiet.sh"

if [[ "$OUTPUT_DIR" = /* ]]; then
    OUTPUT_PATH="$OUTPUT_DIR"
else
    OUTPUT_PATH="$ISAAC_ROOT/$OUTPUT_DIR"
fi

mkdir -p "$OUTPUT_PATH"
cd "$ISAAC_ROOT"

SCREEN_LIST="$(screen -ls 2>/dev/null || true)"

printf 'Starting Franka collection shards\n'
printf '  total_envs=%s episodes_per_shard=%s target_env_episodes=%s gpus=%s shards_per_gpu=%s total_shards=%s\n' \
    "$TOTAL_ENVS" "$EPISODES" "$((TOTAL_ENVS * EPISODES))" "$GPUS" "$SHARDS_PER_GPU" "$TOTAL_SHARDS"
printf '  run_prefix=%s output_dir=%s\n' "$RUN_PREFIX" "$OUTPUT_PATH"
printf '  env_usd=%s camera=%sx%s warmup_steps=%s\n' "$ENV_USD" "$CAMERA_WIDTH" "$CAMERA_HEIGHT" "$CAMERA_WARMUP_RENDER_STEPS"
printf '  data channels: full collector defaults; no data fields are skipped\n'

shard_index=0
for gpu in "${GPU_IDS[@]}"; do
    for ((local_shard = 0; local_shard < SHARDS_PER_GPU; local_shard++)); do
        shard_envs="$BASE_ENVS"
        if ((shard_index < EXTRA_ENVS)); then
            shard_envs=$((shard_envs + 1))
        fi

        shard_id="$(printf '%02d' "$shard_index")"
        session_name="${SESSION_PREFIX}_shard${shard_id}_gpu${gpu}"
        run_name="${RUN_PREFIX}_shard${shard_id}_gpu${gpu}"
        log_path="${OUTPUT_DIR}/${run_name}.log"
        log_path_abs="${OUTPUT_PATH}/${run_name}.log"

        if [[ "$SCREEN_LIST" == *".${session_name}"* ]]; then
            die "screen session already exists: ${session_name}"
        fi

        cmd=(
            scripts/run_franka_collection_quiet.sh
            --headless
            --env-usd "$ENV_USD"
            --num-envs "$shard_envs"
            --episodes "$EPISODES"
            --active-gpu "$gpu"
            --max-gpu-count 1
            --output-dir "$OUTPUT_DIR"
            --world-camera-width "$CAMERA_WIDTH"
            --world-camera-height "$CAMERA_HEIGHT"
            --wrist-camera-width "$CAMERA_WIDTH"
            --wrist-camera-height "$CAMERA_HEIGHT"
            --camera-warmup-render-steps "$CAMERA_WARMUP_RENDER_STEPS"
            --save-workers "$SAVE_WORKERS"
            --max-pending-saves "$MAX_PENDING_SAVES"
            --run-name "$run_name"
        )
        if ((${#EXTRA_ARGS[@]})); then
            cmd+=("${EXTRA_ARGS[@]}")
        fi

        printf '\n[%s]\n' "$session_name"
        printf '  gpu=%s envs=%s run=%s\n' "$gpu" "$shard_envs" "$run_name"
        printf '  log=%s\n' "$log_path_abs"
        printf '  command:'
        printf ' %q' "${cmd[@]}"
        printf '\n'

        if ((DRY_RUN == 0)); then
            screen -dmS "$session_name" bash -lc "cd '$ISAAC_ROOT' && $(printf '%q ' "${cmd[@]}")> '$log_path_abs' 2>&1"
            if ((STAGGER_SECONDS > 0)); then
                sleep "$STAGGER_SECONDS"
            fi
        fi

        shard_index=$((shard_index + 1))
    done
done

if ((DRY_RUN)); then
    printf '\nDry run only; no screen sessions were started.\n'
else
    printf '\nStarted %s screen sessions. Check progress with:\n' "$TOTAL_SHARDS"
    printf '  scripts/franka_collection_progress.sh --run-prefix %q --session-prefix %q --output-dir %q\n' "$RUN_PREFIX" "$SESSION_PREFIX" "$OUTPUT_DIR"
fi
