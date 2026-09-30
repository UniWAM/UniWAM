#!/usr/bin/env bash

set -euo pipefail

MOTUS_ROOT="${MOTUS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

usage() {
    cat <<'EOF'
Usage:
  scripts/libero/eval_libero.sh [options]

Options:
  --checkpoint, -c PATH       Model .pt file or containing pytorch_model directory
  --config PATH               Model config (default: configs/libero_history_future_noise.yaml)
  --benchmark-mode MODE       plus|original (default: plus)
  --task, -t SUITE           libero_spatial|libero_object|libero_goal|libero_10|all
                              (default: libero_goal)
  --num-trials, -n N         trials per task (default: original=50, plus=1)
  --max-tasks N              maximum tasks per suite; -1 means all (default: -1)
  --port, -p PORT            policy server port (default: 9883)
  --host HOST                policy client host (default: 127.0.0.1)
  --gpu-id, -g ID            policy server GPU id (default: 0)
  --seed N                   evaluation seed (default: 7)
  --num-steps-wait N         initial no-op steps (default: 10)
  --num-inference-steps N    diffusion inference steps (default: 10)
  --mujoco-gl BACKEND        egl|osmesa (default: egl)
  --output-root PATH         output root (default: outputs/libero_<mode>)
  --run-name NAME            output subdirectory name (default: checkpoint step name)
  --motus-python PATH        Motus Python executable
  --libero-plus-python PATH  LIBERO-plus Python executable
  --libero-plus-root PATH    LIBERO repository root (selected automatically by mode)
  --libero-config-path PATH  LIBERO config directory (selected automatically by mode)
  --wan-path PATH            WAN model directory
  --vlm-path PATH            Qwen3-VL model directory
  --action-stats PATH        Raw OSC action statistics (required for plus)
  --server-wait-seconds N    server startup timeout (default: 600)
  --tail-lines N             log lines printed on failure (default: 120)
  --help, -h                 show this help

Examples:
  # Smoke test: one task, one trial
  scripts/libero/eval_libero.sh --checkpoint /path/to/model/pytorch_model --action-stats /path/to/model/raw_osc_action_stats.json --task libero_goal --max-tasks 1 --num-trials 1

  # Official original LIBERO smoke test
  scripts/libero/eval_libero.sh --checkpoint /path/to/checkpoint --benchmark-mode original --task libero_goal --max-tasks 1 --num-trials 1

  # Full original LIBERO evaluation: 50 trials per task
  scripts/libero/eval_libero.sh --checkpoint /path/to/checkpoint --benchmark-mode original --task all --num-trials 50
EOF
}

CHECKPOINT="${CHECKPOINT:-}"
CONFIG_PATH="${CONFIG_PATH:-configs/libero_history_future_noise.yaml}"
BENCHMARK_MODE="${BENCHMARK_MODE:-plus}"
TASK="${TASK:-libero_goal}"
NUM_TRIALS="${NUM_TRIALS:-}"
MAX_TASKS="${MAX_TASKS:--1}"
PORT="${PORT:-9883}"
CLIENT_HOST="${CLIENT_HOST:-127.0.0.1}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-7}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
MUJOCO_GL="${MUJOCO_GL:-egl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
RUN_NAME="${RUN_NAME:-}"
MOTUS_PYTHON="${MOTUS_PYTHON:-python}"
LIBERO_PLUS_PYTHON="${LIBERO_PLUS_PYTHON:-python}"
WAN_PATH="${WAN_PATH:-${MOTUS_ROOT}/pretrained_models/Wan2.2-TI2V-5B}"
VLM_PATH="${VLM_PATH:-${MOTUS_ROOT}/pretrained_models/Qwen3-VL-2B-Instruct}"
ACTION_STATS="${ACTION_STATS:-}"
ACTION_STATS_FLAG=0
LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-}"
SERVER_WAIT_SECONDS="${SERVER_WAIT_SECONDS:-600}"
TAIL_LINES="${TAIL_LINES:-120}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint|-c) CHECKPOINT="$2"; shift 2 ;;
        --config) CONFIG_PATH="$2"; shift 2 ;;
        --benchmark-mode) BENCHMARK_MODE="$2"; shift 2 ;;
        --task|-t) TASK="$2"; shift 2 ;;
        --num-trials|-n) NUM_TRIALS="$2"; shift 2 ;;
        --max-tasks) MAX_TASKS="$2"; shift 2 ;;
        --port|-p) PORT="$2"; shift 2 ;;
        --host) CLIENT_HOST="$2"; shift 2 ;;
        --gpu-id|-g) GPU_ID="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --num-steps-wait) NUM_STEPS_WAIT="$2"; shift 2 ;;
        --num-inference-steps) NUM_INFERENCE_STEPS="$2"; shift 2 ;;
        --mujoco-gl) MUJOCO_GL="$2"; shift 2 ;;
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        --run-name) RUN_NAME="$2"; shift 2 ;;
        --motus-python) MOTUS_PYTHON="$2"; shift 2 ;;
        --libero-plus-python) LIBERO_PLUS_PYTHON="$2"; shift 2 ;;
        --libero-plus-root) LIBERO_PLUS_ROOT="$2"; shift 2 ;;
        --libero-config-path) LIBERO_CONFIG_PATH="$2"; shift 2 ;;
        --wan-path) WAN_PATH="$2"; shift 2 ;;
        --vlm-path) VLM_PATH="$2"; shift 2 ;;
        --action-stats) ACTION_STATS="$2"; ACTION_STATS_FLAG=1; shift 2 ;;
        --server-wait-seconds) SERVER_WAIT_SECONDS="$2"; shift 2 ;;
        --tail-lines) TAIL_LINES="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "${BENCHMARK_MODE}" in
    plus)
        [[ "${ACTION_STATS_FLAG}" -eq 1 && -f "${ACTION_STATS}" ]] || {
            echo "LIBERO-plus requires --action-stats pointing to its matching raw_osc_action_stats.json" >&2
            exit 2
        }
        LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-${MOTUS_ROOT}/../LIBERO-plus}"
        LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${MOTUS_ROOT}/outputs/.libero_configs/plus}"
        OUTPUT_ROOT="${OUTPUT_ROOT:-${MOTUS_ROOT}/outputs/libero_plus}"
        NUM_TRIALS="${NUM_TRIALS:-1}"
        ;;
    original)
        LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-${MOTUS_ROOT}/../LIBERO}"
        LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${MOTUS_ROOT}/outputs/.libero_configs/original}"
        OUTPUT_ROOT="${OUTPUT_ROOT:-${MOTUS_ROOT}/outputs/libero_original}"
        NUM_TRIALS="${NUM_TRIALS:-50}"
        ;;
    *) echo "Unsupported benchmark mode: ${BENCHMARK_MODE}" >&2; exit 2 ;;
esac

case "${TASK}" in
    libero_spatial|libero_object|libero_goal|libero_10|all) ;;
    *) echo "Unsupported task suite: ${TASK}" >&2; exit 2 ;;
esac
case "${MUJOCO_GL}" in
    osmesa|egl) ;;
    *) echo "Unsupported MuJoCo backend: ${MUJOCO_GL}" >&2; exit 2 ;;
esac

[[ -n "${CHECKPOINT}" && ( -d "${CHECKPOINT}" || -f "${CHECKPOINT}" ) ]] || { echo "Checkpoint not found: ${CHECKPOINT}" >&2; exit 1; }
[[ -f "${MOTUS_ROOT}/${CONFIG_PATH}" || -f "${CONFIG_PATH}" ]] || { echo "Config not found: ${CONFIG_PATH}" >&2; exit 1; }
command -v "${MOTUS_PYTHON}" >/dev/null || { echo "Motus Python not found: ${MOTUS_PYTHON}" >&2; exit 1; }
command -v "${LIBERO_PLUS_PYTHON}" >/dev/null || { echo "LIBERO Python not found: ${LIBERO_PLUS_PYTHON}" >&2; exit 1; }
[[ -d "${LIBERO_PLUS_ROOT}" ]] || { echo "LIBERO-plus root not found: ${LIBERO_PLUS_ROOT}" >&2; exit 1; }
for asset_dir in assets bddl_files init_files; do
    [[ -d "${LIBERO_PLUS_ROOT}/libero/libero/${asset_dir}" ]] || {
        echo "LIBERO benchmark asset missing: ${LIBERO_PLUS_ROOT}/libero/libero/${asset_dir}" >&2
        exit 1
    }
done
if [[ ! -f "${LIBERO_CONFIG_PATH}/config.yaml" ]]; then
    mkdir -p "${LIBERO_CONFIG_PATH}"
    printf '%s\n' \
        "assets: ${LIBERO_PLUS_ROOT}/libero/libero/assets" \
        "bddl_files: ${LIBERO_PLUS_ROOT}/libero/libero/bddl_files" \
        "benchmark_root: ${LIBERO_PLUS_ROOT}/libero/libero" \
        "datasets: ${LIBERO_PLUS_ROOT}/libero/datasets" \
        "init_states: ${LIBERO_PLUS_ROOT}/libero/libero/init_files" \
        >"${LIBERO_CONFIG_PATH}/config.yaml"
    echo "Generated LIBERO config: ${LIBERO_CONFIG_PATH}/config.yaml"
fi
CHECKPOINT="$(readlink -f "${CHECKPOINT}")"
if [[ -z "${RUN_NAME}" ]]; then
    RUN_NAME="$(basename "$(dirname "${CHECKPOINT}")")"
fi
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
RESULTS_DIR="${RUN_DIR}/results"
LOG_DIR="${RUN_DIR}/logs"
SERVER_LOG="${LOG_DIR}/server.log"
mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

server_pid=""
cleanup() {
    local exit_code=$?
    if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
        echo "Stopping policy server (PID ${server_pid})..."
        kill "${server_pid}" 2>/dev/null || true
        for _ in {1..10}; do
            kill -0 "${server_pid}" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "${server_pid}" 2>/dev/null; then
            kill -9 "${server_pid}" 2>/dev/null || true
        fi
        wait "${server_pid}" 2>/dev/null || true
    fi
    if [[ ${exit_code} -ne 0 ]]; then
        echo "Evaluation failed. Recent server log:" >&2
        tail -n "${TAIL_LINES}" "${SERVER_LOG}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

wait_for_server() {
    local waited=0
    while (( waited < SERVER_WAIT_SECONDS )); do
        kill -0 "${server_pid}" 2>/dev/null || return 1
        if bash -c ">/dev/tcp/127.0.0.1/${PORT}" 2>/dev/null; then
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done
    return 1
}

echo "Starting Motus LIBERO evaluation:"
echo "  BENCHMARK_MODE=${BENCHMARK_MODE}"
echo "  LIBERO_ROOT=${LIBERO_PLUS_ROOT}"
echo "  CHECKPOINT=${CHECKPOINT}"
echo "  CONFIG_PATH=${CONFIG_PATH}"
echo "  TASK=${TASK}"
echo "  NUM_TRIALS=${NUM_TRIALS}"
echo "  MAX_TASKS=${MAX_TASKS}"
echo "  GPU_ID=${GPU_ID}"
echo "  MUJOCO_GL=${MUJOCO_GL}"
echo "  RUN_DIR=${RUN_DIR}"

action_stats_args=()
if [[ -n "${ACTION_STATS}" ]]; then
    action_stats_args=(--action-stats "${ACTION_STATS}")
fi

(
    cd "${MOTUS_ROOT}"
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
    export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
    export PYTHONPATH="${MOTUS_ROOT}:${PYTHONPATH:-}"
    exec "${MOTUS_PYTHON}" -m examples.libero_plus.policy_server \
        --checkpoint "${CHECKPOINT}" \
        --config "${CONFIG_PATH}" \
        --wan-path "${WAN_PATH}" \
        --vlm-path "${VLM_PATH}" \
        "${action_stats_args[@]}" \
        --num-inference-steps "${NUM_INFERENCE_STEPS}" \
        --port "${PORT}"
) >"${SERVER_LOG}" 2>&1 &
server_pid=$!

echo "Policy server PID ${server_pid}; waiting for port ${PORT}..."
if ! wait_for_server; then
    echo "Policy server failed to become ready." >&2
    tail -n "${TAIL_LINES}" "${SERVER_LOG}" >&2 || true
    exit 1
fi
echo "Policy server is ready."

run_suite() {
    local suite="$1"
    local eval_log="${LOG_DIR}/${suite}.log"
    echo "Evaluating ${suite}: ${NUM_TRIALS} trial(s) per task..."
    if ! (
        cd "${MOTUS_ROOT}"
        export LIBERO_HOME="${LIBERO_PLUS_ROOT}"
        export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH}"
        export PYTHONPATH="${LIBERO_PLUS_ROOT}:${MOTUS_ROOT}:${PYTHONPATH:-}"
        export TOKENIZERS_PARALLELISM=false
        export MUJOCO_GL="${MUJOCO_GL}"
        export PYOPENGL_PLATFORM="${MUJOCO_GL}"
        exec "${LIBERO_PLUS_PYTHON}" examples/libero_plus/eval_libero.py \
            --host "${CLIENT_HOST}" \
            --port "${PORT}" \
            --benchmark-mode "${BENCHMARK_MODE}" \
            --task-suite-name "${suite}" \
            --num-trials-per-task "${NUM_TRIALS}" \
            --max-tasks "${MAX_TASKS}" \
            --num-steps-wait "${NUM_STEPS_WAIT}" \
            --num-inference-steps "${NUM_INFERENCE_STEPS}" \
            --seed "${SEED}" \
            --output-dir "${RESULTS_DIR}"
    ) >"${eval_log}" 2>&1; then
        echo "Evaluation failed for ${suite}. Recent eval log:" >&2
        tail -n "${TAIL_LINES}" "${eval_log}" >&2 || true
        return 1
    fi
    echo "Finished ${suite}:"
    echo "  log=${eval_log}"
    echo "  results=${RESULTS_DIR}/${suite}/results.json"
}

if [[ "${TASK}" == "all" ]]; then
    for suite in libero_spatial libero_object libero_goal libero_10; do
        run_suite "${suite}"
    done
else
    run_suite "${TASK}"
fi

echo "LIBERO evaluation complete. Outputs: ${RUN_DIR}"
