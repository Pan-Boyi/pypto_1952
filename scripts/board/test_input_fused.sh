#!/usr/bin/env bash
set -eo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "${1:-}" == "--runtime-env" ]]; then
    if [[ $# -lt 2 ]]; then echo "--runtime-env requires a board environment script" >&2; exit 2; fi
    if source "$2"; then :; else echo "Board runtime environment failed: $2" >&2; exit 1; fi
    shift 2
fi
set -eo pipefail
exec "${PYTHON:-python3}" -B "$script_dir/board_verify.py" --operator input_fused "$@"
