#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "$script_dir/.."
exec "${PYTHON:-python3}" -B -m pypto_1952.eight_head_profile "$@"
