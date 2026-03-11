#!/usr/bin/env bash
set -euo pipefail

PID="${1:-}"
RUN_DIR="${2:-/workspace/MOREdit/output/20260307-082747}"
INTERVAL="${3:-60}"

if [[ -z "${PID}" ]]; then
  echo "usage: $0 <train_pid> [run_dir] [interval_sec]" >&2
  exit 1
fi

TS="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="/workspace/MOREdit/output"
LOG_FILE="${OUT_DIR}/watch_${TS}.log"
ALERT_FILE="${OUT_DIR}/watch_${TS}.alerts.log"
LATEST_LINK="${OUT_DIR}/watch_latest.log"
LATEST_ALERT_LINK="${OUT_DIR}/watch_latest.alerts.log"

ln -sf "${LOG_FILE}" "${LATEST_LINK}"
ln -sf "${ALERT_FILE}" "${LATEST_ALERT_LINK}"

echo "[watch] start ts=${TS} pid=${PID} run_dir=${RUN_DIR} interval=${INTERVAL}s" | tee -a "${LOG_FILE}"

last_ckpt_count=0
last_ckpt_step=0
stuck_minutes=0

while true; do
  now="$(date '+%F %T')"
  if ! kill -0 "${PID}" 2>/dev/null; then
    echo "[watch][${now}] train process exited pid=${PID}" | tee -a "${LOG_FILE}"
    break
  fi

  stat_line="$(ps -o etimes=,pcpu=,pmem=,stat= -p "${PID}" | xargs)"
  gpu_line="$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader | head -n1)"
  app_line="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | rg "^${PID}," || true)"

  ckpt_count=0
  latest_ckpt=""
  if [[ -d "${RUN_DIR}/weights" ]]; then
    ckpt_count="$(find "${RUN_DIR}/weights" -maxdepth 1 -type f -name 'lora_step_*.pt' | wc -l | xargs)"
    latest_ckpt="$(find "${RUN_DIR}/weights" -maxdepth 1 -type f -name 'lora_step_*.pt' | sort | tail -n1)"
  fi
  if [[ -n "${latest_ckpt}" ]]; then
    step_str="$(basename "${latest_ckpt}" | sed -E 's/^lora_step_([0-9]+)\.pt$/\1/')"
    last_ckpt_step="$((10#${step_str}))"
  fi

  echo "[watch][${now}] pid=${PID} ps='${stat_line}' gpu='${gpu_line}' app='${app_line:-none}' ckpts=${ckpt_count} last_step=${last_ckpt_step}" | tee -a "${LOG_FILE}"

  if [[ "${ckpt_count}" -eq "${last_ckpt_count}" ]]; then
    stuck_minutes=$((stuck_minutes + 1))
  else
    stuck_minutes=0
    last_ckpt_count="${ckpt_count}"
  fi

  if [[ -z "${app_line}" ]]; then
    echo "[alert][${now}] no CUDA context detected for pid=${PID}" | tee -a "${ALERT_FILE}"
  fi
  if [[ "${stuck_minutes}" -ge 45 ]]; then
    echo "[alert][${now}] no new checkpoint for ${stuck_minutes} minutes (last_step=${last_ckpt_step})" | tee -a "${ALERT_FILE}"
    stuck_minutes=0
  fi

  sleep "${INTERVAL}"
done

echo "[watch] done ts=$(date '+%F %T')" | tee -a "${LOG_FILE}"
