cat > ~/cleanup_gpu.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

echo "==> Current GPU processes"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv || true
echo

# Kill only YOUR GPU-bound processes
PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)

if [ -z "${PIDS}" ]; then
  echo "No GPU PIDs found."
else
  echo "==> Gracefully stopping your GPU PIDs..."
  for p in ${PIDS}; do
    u=$(ps -o user= -p "${p}" 2>/dev/null | tr -d ' ' || true)
    if [ "${u}" = "${USER}" ]; then
      echo "TERM ${p} ($(ps -o comm= -p ${p} 2>/dev/null))"
      kill -TERM "${p}" 2>/dev/null || true
    fi
  done

  sleep 5

  echo "==> Force-killing survivors..."
  for p in ${PIDS}; do
    u=$(ps -o user= -p "${p}" 2>/dev/null | tr -d ' ' || true)
    if [ "${u}" = "${USER}" ] && kill -0 "${p}" 2>/dev/null; then
      echo "KILL ${p} ($(ps -o comm= -p ${p} 2>/dev/null))"
      kill -9 "${p}" 2>/dev/null || true
    fi
  done
fi

# Also stop common launchers so they don't respawn workers
echo "==> Stopping common launchers..."
pkill -f "accelerate.*launch" || true
pkill -f "torchrun"           || true
pkill -f "finetune_eval.py"   || true

echo
echo "==> Remaining users of /dev/nvidia* (if any):"
fuser -v /dev/nvidia* || true

echo
echo "==> GPU status after cleanup:"
nvidia-smi || true
SH
