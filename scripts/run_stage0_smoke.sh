#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
isaaclab_root="${ISAACLAB_ROOT:-/home/engineer/IsaacLab}"
isaacsim_root="${ISAACSIM_ROOT:-$isaaclab_root/_isaac_sim}"
pegasus_root="${PEGASUS_ROOT:-$isaaclab_root/external/PegasusSimulator}"
pegasus_extension="$pegasus_root/extensions/pegasus.simulator"

if [[ ! -x "$isaacsim_root/python.sh" ]]; then
  echo "Isaac Sim is missing: expected executable $isaacsim_root/python.sh" >&2
  exit 2
fi
if [[ ! -d "$pegasus_extension/pegasus/simulator" ]]; then
  echo "Pegasus Python extension is missing: $pegasus_extension" >&2
  exit 2
fi
if ! "$isaacsim_root/python.sh" -c "import isaaclab" >/dev/null 2>&1; then
  echo "Isaac Lab is not installed in the Isaac Sim Python interpreter." >&2
  echo "Run: cd $isaaclab_root && ./_isaac_sim/python.sh -m pip install --no-build-isolation -e source/isaaclab" >&2
  exit 2
fi
if ! "$isaacsim_root/python.sh" -c "import h5py" >/dev/null 2>&1; then
  echo "Isaac Lab runtime dependency h5py is missing from Isaac Sim Python." >&2
  echo "Run: $isaacsim_root/python.sh -m pip install h5py==3.14.0" >&2
  exit 2
fi
if ! nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA driver is not usable; nvidia-smi failed." >&2
  exit 2
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Ignoring CUDA_VISIBLE_DEVICES because Isaac Sim uses its own GPU enumeration." >&2
  unset CUDA_VISIBLE_DEVICES
fi

# Workstation GPU policy:
#   0, 1: RTX 3090 compute/simulation GPUs
#   2:    T1000 Xorg/desktop-only GPU (unsupported by Isaac Sim)
gpu_index="${VACAN_GPU_INDEX:-1}"
case "$gpu_index" in
  0|1) ;;
  2)
    echo "GPU 2 is the T1000 desktop adapter and is excluded from VACaN simulation." >&2
    echo "Use the default GPU 1, or explicitly set VACAN_GPU_INDEX=0 for the other RTX 3090." >&2
    exit 2
    ;;
  *)
    echo "Invalid VACAN_GPU_INDEX=$gpu_index; this host allows RTX 3090 indices 0 or 1." >&2
    exit 2
    ;;
esac
gpu_name="$(nvidia-smi --id="$gpu_index" --query-gpu=name --format=csv,noheader | head -n 1)"
if [[ "$gpu_name" != *"RTX 3090"* ]]; then
  echo "Refusing GPU $gpu_index ($gpu_name): VACaN requires an RTX 3090 on this workstation." >&2
  exit 2
fi

# GPU 1 is the configured RTX-backed Xorg presentation device. Keep GUI mode
# by default; set VACAN_HEADLESS=1 explicitly for compute-only execution.
runtime_args=("$@")
if [[ "${VACAN_HEADLESS:-0}" == "1" ]]; then
  has_headless=0
  for argument in "${runtime_args[@]}"; do
    if [[ "$argument" == "--headless" ]]; then
      has_headless=1
      break
    fi
  done
  if [[ "$has_headless" -eq 0 ]]; then
    runtime_args+=(--headless)
  fi
fi

echo "[INFO] VACaN simulation GPU: $gpu_index ($gpu_name); headless: ${VACAN_HEADLESS:-0}"

export PYTHONPATH="$project_root/src:$pegasus_extension:${PYTHONPATH:-}"
mkdir -p "$project_root/artifacts/stage0"
for run_number in 1 2; do
  result_path="$project_root/artifacts/stage0/smoke-run-${run_number}.json"
  log_path="$project_root/artifacts/stage0/smoke-run-${run_number}.log"
  # Prevent an old successful result from being mistaken for this process.
  : > "$result_path"
  : > "$log_path"
  if ! timeout --signal=INT --kill-after=30s "${VACAN_STAGE0_TIMEOUT:-300}" \
      "$isaacsim_root/python.sh" \
      "$project_root/scripts/isaacsim/stage0_integration_smoke.py" \
      --config "$project_root/configs/isaacsim/stage0_smoke.yaml" \
      --result "$result_path" \
      --gpu "$gpu_index" \
      "${runtime_args[@]}" 2>&1 | tee "$log_path"; then
    echo "Stage 0 smoke process $run_number failed; not starting another process." >&2
    exit 1
  fi
  if [[ ! -f "$result_path" ]] || ! grep -q '"status": "pass"' "$result_path"; then
    echo "Stage 0 smoke result $run_number did not pass; not starting another process." >&2
    exit 1
  fi
  if [[ "$run_number" -eq 1 ]]; then
    sleep 10
  fi
done
echo "PASS: two independent Isaac Sim processes terminated cleanly."
