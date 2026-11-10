#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "$script_dir/.."
exec "${PYTHON:-python3}" -m pypto_1952.dec_enc_profile "$@"
