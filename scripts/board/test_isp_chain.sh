#!/usr/bin/env bash
set -eo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "${1:-}" == "--runtime-env" ]]; then
    if [[ $# -lt 2 ]]; then
        echo "--runtime-env requires a board environment script" >&2
        exit 2
    fi
    source "$2"
    shift 2
fi
exec "${PYTHON:-python3}" -I -S -B "$script_dir/board_chain.py" "$@"
