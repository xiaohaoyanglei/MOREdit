#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace"
PROJECT_DIR="${ROOT}/MOREdit"
CONFIG_PATH="${PROJECT_DIR}/configs/train_10k_mhpv2.yaml"
OUT_BASE="${PROJECT_DIR}/output"
RUN_TS="$(date +%Y%m%d-%H%M%S)"
LAUNCH_LOG="${PROJECT_DIR}/output/launch_${RUN_TS}.log"

mkdir -p "${OUT_BASE}"

cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "[launcher] config=${CONFIG_PATH}" | tee -a "${LAUNCH_LOG}"
echo "[launcher] output_base=${OUT_BASE}" | tee -a "${LAUNCH_LOG}"

EXISTING_RUNS_FILE="$(mktemp)"
find "${OUT_BASE}" -mindepth 1 -maxdepth 1 -type d -printf '%p\n' | sort > "${EXISTING_RUNS_FILE}"

python -m MOREdit.train --config "${CONFIG_PATH}" >> "${LAUNCH_LOG}" 2>&1 &
TRAIN_PID=$!
echo "[launcher] train_pid=${TRAIN_PID}" | tee -a "${LAUNCH_LOG}"

RUN_DIR=""
while true; do
  if ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
    echo "[launcher] train process exited before creating run directory" | tee -a "${LAUNCH_LOG}"
    rm -f "${EXISTING_RUNS_FILE}"
    wait "${TRAIN_PID}" || true
    exit 1
  fi
  CANDIDATE="$(
    find "${OUT_BASE}" -mindepth 1 -maxdepth 1 -type d -printf '%p\n' | sort | \
    comm -13 "${EXISTING_RUNS_FILE}" - | tail -n1
  )"
  if [[ -n "${CANDIDATE}" && -f "${CANDIDATE}/run_metadata.json" ]]; then
    RUN_DIR="${CANDIDATE}"
    break
  fi
  sleep 2
done
rm -f "${EXISTING_RUNS_FILE}"

if [[ -z "${RUN_DIR}" ]]; then
  echo "[launcher] failed to detect run directory in ${OUT_BASE}" | tee -a "${LAUNCH_LOG}"
  kill "${TRAIN_PID}" || true
  exit 1
fi

echo "[launcher] run_dir=${RUN_DIR}" | tee -a "${LAUNCH_LOG}"
echo "${RUN_DIR}" > "${PROJECT_DIR}/output/latest_run_dir.txt"

MASK_IMAGE="/workspace/dataset/mhpv2_triples_en/images/train_1004.jpg"
MASK_PROMPT="the 1st person from the left"

for STEP in $(seq 1000 1000 10000); do
  WEIGHT_PATH="${RUN_DIR}/weights/lora_step_$(printf '%06d' "${STEP}").pt"
  while true; do
    if [[ -f "${WEIGHT_PATH}" ]]; then
      break
    fi
    if ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
      break
    fi
    sleep 10
  done

  if [[ ! -f "${WEIGHT_PATH}" ]]; then
    echo "[launcher] checkpoint missing at step=${STEP}, stopping mask inference loop" | tee -a "${LAUNCH_LOG}"
    break
  fi

  echo "[launcher] running mask inference for step=${STEP}" | tee -a "${LAUNCH_LOG}"
  INFER_OUT="${RUN_DIR}/mask_infer/step_$(printf '%06d' "${STEP}")"
  mkdir -p "${INFER_OUT}"
  python -m MOREdit.inference.run_soft_mask \
    --config "${CONFIG_PATH}" \
    --lora-weights "${WEIGHT_PATH}" \
    --image "${MASK_IMAGE}" \
    --pointer-prompt "${MASK_PROMPT}" \
    --edit-prompt "" \
    --mask-mode peak_region \
    --output-dir "${INFER_OUT}" >> "${LAUNCH_LOG}" 2>&1 || {
      echo "[launcher] mask inference failed at step=${STEP}" | tee -a "${LAUNCH_LOG}"
    }
done

echo "[launcher] waiting for train process to finish" | tee -a "${LAUNCH_LOG}"
wait "${TRAIN_PID}"
EXIT_CODE=$?
echo "[launcher] train exit code=${EXIT_CODE}" | tee -a "${LAUNCH_LOG}"
exit "${EXIT_CODE}"
