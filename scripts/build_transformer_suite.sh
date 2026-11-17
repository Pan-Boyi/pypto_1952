#!/usr/bin/env bash
# Compiler host: build each operator in a fresh process, then collect portable board packages.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ $# -lt 1 || "${1:-}" == "--help" ]]; then
    echo "Usage: $0 NEW_OUTPUT_ROOT --gym-root GYM --toolkit-root CANN --pto-root PTO [build options]"
    exit 0
fi
output_root=$("${PYTHON:-python3}" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$1")
shift
mkdir -p -- "$(dirname -- "$output_root")"
mkdir -- "$output_root"
suite="$output_root/board_suite"
mkdir -- "$suite"
for op in dec_enc_fused ffm_fused input_fused; do
    bash "$script_dir/build_transformer_ops.sh" build --op "$op" \
        --output "$output_root/$op" "$@" 2>&1 | tee "$output_root/$op.log"
    mkdir -- "$suite/$op"
    cp -- "$output_root/$op/board/runtime.tar.gz" "$output_root/$op/board/SHA256SUMS" "$suite/$op/"
done
cp -- "$script_dir/board/test_transformer_suite.sh" "$suite/test_transformer_suite.sh"
tar -czf "$output_root/board_suite.tar.gz" -C "$output_root" board_suite
(cd -- "$output_root" && sha256sum board_suite.tar.gz > board_suite.SHA256SUMS)
echo "Board suite: $output_root/board_suite.tar.gz"
echo "Checksum:   $output_root/board_suite.SHA256SUMS"
