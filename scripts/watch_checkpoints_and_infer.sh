#!/usr/bin/env bash
set -euo pipefail

TRAIN_PID="${1:-}"
RUN_DIR="${2:-}"
CONFIG_PATH="${3:-/workspace/MOREdit/configs/train_10k_mhpv2.yaml}"
MAX_STEP="${4:-10000}"
STEP_INTERVAL="${5:-1000}"

if [[ -z "${TRAIN_PID}" || -z "${RUN_DIR}" ]]; then
  echo "usage: $0 <train_pid> <run_dir> [config_path] [max_step] [step_interval]" >&2
  exit 1
fi

LOG_FILE="${RUN_DIR}/mask_infer_watch.log"
MASK_IMAGE="/workspace/dataset/mhpv2_triples_en/images/train_1004.jpg"
MASK_PROMPT="the 1st person from the left"

echo "[mask-watch] train_pid=${TRAIN_PID} run_dir=${RUN_DIR} max_step=${MAX_STEP} interval=${STEP_INTERVAL}" | tee -a "${LOG_FILE}"

for STEP in $(seq "${STEP_INTERVAL}" "${STEP_INTERVAL}" "${MAX_STEP}"); do
  WEIGHT_PATH="${RUN_DIR}/weights/lora_step_$(printf '%06d' "${STEP}").pt"
  while true; do
    if [[ -f "${WEIGHT_PATH}" ]]; then
      break
    fi
    if ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
      echo "[mask-watch] train exited before checkpoint ${STEP}" | tee -a "${LOG_FILE}"
      exit 0
    fi
    sleep 15
  done

  OUT_DIR="${RUN_DIR}/mask_infer/step_$(printf '%06d' "${STEP}")"
  mkdir -p "${OUT_DIR}"
  echo "[mask-watch] infer step=${STEP}" | tee -a "${LOG_FILE}"
  python -m MOREdit.inference.run_soft_mask \
    --config "${CONFIG_PATH}" \
    --lora-weights "${WEIGHT_PATH}" \
    --image "${MASK_IMAGE}" \
    --pointer-prompt "${MASK_PROMPT}" \
    --edit-prompt "" \
    --mask-mode peak_region \
    --output-dir "${OUT_DIR}" >> "${LOG_FILE}" 2>&1 || {
      echo "[mask-watch] inference failed at step=${STEP}" | tee -a "${LOG_FILE}"
    }
done

echo "[mask-watch] completed all scheduled inference checkpoints" | tee -a "${LOG_FILE}"
