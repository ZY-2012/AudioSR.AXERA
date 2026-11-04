#!/bin/bash
# AudioSR.AXERA one-click demo.
#   bash run_ax650.sh [input.wav] [output.wav] [--python]
set -e
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=/soc/lib:${LD_LIBRARY_PATH:-}
export PYTHONDONTWRITEBYTECODE=1
INPUT="${1:-samples/listen/16k/speech_2_input_16k.wav}"
OUTPUT="${2:-output.wav}"
if [ $# -ge 2 ]; then shift 2; else shift $#; fi
MODE=cpp; EXTRA=()
for a in "$@"; do
  case "$a" in
    --python) MODE=python ;;
    --cpp) MODE=cpp ;;
    *) EXTRA+=("$a") ;;
  esac
done
if [ ! -x ./bin/audiosr_ax650 ]; then MODE=python; fi
if [ "$MODE" = cpp ]; then
  exec ./bin/audiosr_ax650 models "$INPUT" "$OUTPUT" "${EXTRA[@]}"
else
  exec python3 python/demo.py "$INPUT" -o "$OUTPUT" --models models --ddim models/audiosr_ddim_10.npz "${EXTRA[@]}"
fi
