#!/usr/bin/env bash
set -euo pipefail
SIM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SIM_ROOT"
export PYTHONPATH="$SIM_ROOT/.runtime/photo-simulator:$SIM_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
exec "$SIM_ROOT/.venv/bin/python" -B -m photo_simulator.server --port "${SIMULATOR_PORT:-8787}" --device "${SIMULATOR_DEVICE:-auto}"
