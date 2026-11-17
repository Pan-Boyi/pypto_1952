#!/usr/bin/env bash
# Board host: keep each release and its result separate; stop on the first failed test.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ $# -lt 1 || "${1:-}" == "--help" ]]; then
    echo "Usage: $0 NEW_RESULT_ROOT [--runtime-env BOARD_SET_ENV] [--device ID] [--verify-only]"
    exit 0
fi
output_root=$("${PYTHON:-python3}" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$1")
shift
verify_only=false
for arg in "$@"; do
    if [[ "$arg" == "--verify-only" ]]; then verify_only=true; fi
done
mkdir -p -- "$(dirname -- "$output_root")"
mkdir -- "$output_root"
for op in dec_enc_fused ffm_fused input_fused; do
    (cd -- "$script_dir/$op" && sha256sum -c SHA256SUMS)
    release="$output_root/$op/release"
    mkdir -p -- "$release"
    tar -xzf "$script_dir/$op/runtime.tar.gz" -C "$release"
    bash "$release/runtime/test_$op.sh" "$@" --output "$output_root/$op/result"
done
if [[ "$verify_only" == true ]]; then
    echo "All three packages verified; device execution not run. Results: $output_root"
else
    echo "All three operator tests passed. Results: $output_root"
fi
