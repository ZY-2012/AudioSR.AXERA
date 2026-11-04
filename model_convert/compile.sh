#!/bin/bash
set -e
cd "$(dirname "$0")/.."
IMAGE="${PULSAR2_IMAGE:-docker-registry.aitsw.axera-tech.com/pulsar2:8.0}"
HASP_ARGS=(-v /var/hasplm:/var/hasplm)
if [ -n "${HASP_HOME:-}" ]; then HASP_ARGS+=(-v "${HASP_HOME}:/root/.hasplm"); fi
for name in "${@}"; do
  echo "=== compile ${name} ==="
  docker run --rm --network host -v "$PWD:/workspace" "${HASP_ARGS[@]}" -e HASP_HOME=/root/.hasplm \
    "$IMAGE" -lc "set -e; mkdir -p /root/.hasplm/installed/32434 && cp -f /root/*.v2c /root/.hasplm/installed/32434/ 2>/dev/null || true; PATH=/usr/local/bin/.venv/bin:/opt/pulsar2:\$PATH; pulsar2 build --config /workspace/model_convert/configs/${name}.json" 2>&1 | tee "build/compile_${name}.log"
  python3 scripts/quant_report.py "$name" 2>&1 | tee -a "build/compile_${name}.log"
  if [ -f "build/compile/${name}_work/quant/debug/precision_analysis_table.txt" ]; then
    echo "=== end-to-end precision table: ${name} ===" | tee -a "build/compile_${name}.log"
    cat "build/compile_${name}_work/quant/debug/precision_analysis_table.txt" | tee -a "build/compile_${name}.log"
  fi
  if ! python3 scripts/precision_gate.py "${name}" 2>&1 | tee -a "build/compile_${name}.log"; then
    echo "=== PRECISION GATE FAILED: ${name}; stopping further compilation ===" | tee -a "build/compile_${name}.log"
    exit 1
  fi
done
find build/compile -name '*.axmodel' -ls
