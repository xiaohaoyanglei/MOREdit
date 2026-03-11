#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace"
PROJECT_DIR="${ROOT}/MOREdit"
CONFIG_PATH="${PROJECT_DIR}/configs/train_10k_mhpv2.yaml"
OUT_BASE="${PROJECT_DIR}/output"
TS="$(date +%Y%m%d-%H%M%S)"
LAUNCH_LOG="${OUT_BASE}/direct_launch_${TS}.log"

mkdir -p "${OUT_BASE}"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-all}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

BEFORE_FILE="$(mktemp)"
find "${OUT_BASE}" -mindepth 1 -maxdepth 1 -type d -printf '%p\n' | sort > "${BEFORE_FILE}"

echo "[direct-launch] config=${CONFIG_PATH}" | tee -a "${LAUNCH_LOG}"
python -m MOREdit.train --config "${CONFIG_PATH}" >> "${LAUNCH_LOG}" 2>&1 &
TRAIN_PID=$!
echo "[direct-launch] train_pid=${TRAIN_PID}" | tee -a "${LAUNCH_LOG}"

RUN_DIR=""
while true; do
  if ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
    echo "[direct-launch] train exited before run dir creation" | tee -a "${LAUNCH_LOG}"
    rm -f "${BEFORE_FILE}"
    wait "${TRAIN_PID}" || true
    exit 1
  fi
  CANDIDATE="$(
    find "${OUT_BASE}" -mindepth 1 -maxdepth 1 -type d -printf '%p\n' | sort | \
    comm -13 "${BEFORE_FILE}" - | tail -n 1
  )"
  if [[ -n "${CANDIDATE}" && -f "${CANDIDATE}/run_metadata.json" ]]; then
    RUN_DIR="${CANDIDATE}"
    break
  fi
  sleep 2
done
rm -f "${BEFORE_FILE}"

echo "[direct-launch] run_dir=${RUN_DIR}" | tee -a "${LAUNCH_LOG}"
echo "${RUN_DIR}" > "${OUT_BASE}/latest_run_dir.txt"

nohup "${PROJECT_DIR}/scripts/watch_train_until_done.sh" "${TRAIN_PID}" "${RUN_DIR}" 60 >> "${LAUNCH_LOG}" 2>&1 &
WATCH_PID=$!
echo "[direct-launch] watch_pid=${WATCH_PID}" | tee -a "${LAUNCH_LOG}"

nohup "${PROJECT_DIR}/scripts/watch_checkpoints_and_infer.sh" "${TRAIN_PID}" "${RUN_DIR}" "${CONFIG_PATH}" 10000 1000 >> "${LAUNCH_LOG}" 2>&1 &
INFER_WATCH_PID=$!
echo "[direct-launch] infer_watch_pid=${INFER_WATCH_PID}" | tee -a "${LAUNCH_LOG}"

wait "${TRAIN_PID}"
EXIT_CODE=$?
echo "[direct-launch] train exit code=${EXIT_CODE}" | tee -a "${LAUNCH_LOG}"
exit "${EXIT_CODE}"
