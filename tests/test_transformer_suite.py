"""Shell orchestration boundaries: no compiler, transport or NPU is required."""

import hashlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest

from pypto_1952.transformer_ops import OPERATORS


class TransformerSuiteTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.scripts = self.root / 'scripts with spaces'
        self.scripts.mkdir()
        (self.scripts / 'board').mkdir()
        repo = Path(__file__).resolve().parents[1]
        for name in ('build_transformer_suite.sh', 'board/test_transformer_suite.sh'):
            shutil.copy2(repo / 'scripts' / name, self.scripts / name)
        self.events = self.root / 'events.txt'
        self.env = dict(os.environ, PYTHON=sys.executable, SUITE_EVENTS=str(self.events))

    def fake_archive(self, op):
        root = self.root / 'fixture' / op
        root.mkdir(parents=True)
        script = ('#!/bin/bash\nset -eu\n'
                  'printf "%s\\n" "' + op + ' $*" >> "$SUITE_EVENTS"\n'
                  'if [[ "${SUITE_FAIL:-}" == "' + op + '" ]]; then exit 17; fi\n'
                  'while [[ "$1" != "--output" ]]; do shift; done\n'
                  'mkdir "$2"\nprintf \'{"status":"passed"}\\n\' > "$2/validation.json"\n')
        archive = root / 'runtime.tar.gz'
        with tarfile.open(archive, 'w:gz') as tar:
            data = script.encode()
            member = tarfile.TarInfo('runtime/test_' + op + '.sh')
            member.mode, member.size = 0o755, len(data)
            tar.addfile(member, io.BytesIO(data))
        (root / 'SHA256SUMS').write_text(hashlib.sha256(archive.read_bytes()).hexdigest() + '  runtime.tar.gz\n')
        return root

    def build(self, fail=''):
        for op in OPERATORS:
            self.fake_archive(op)
        (self.scripts / 'build_transformer_ops.sh').write_text('''#!/bin/bash
set -eu
op=''; output=''
while [[ $# -gt 0 ]]; do
  case "$1" in
    --op) op=$2; shift 2 ;;
    --output) output=$2; shift 2 ;;
    *) shift ;;
  esac
done
printf '%s\n' "$op" >> "$SUITE_EVENTS"
if [[ "$op" == "${SUITE_FAIL:-}" ]]; then exit 17; fi
mkdir -p "$output/board"
cp "$SUITE_FIXTURES/$op/runtime.tar.gz" "$SUITE_FIXTURES/$op/SHA256SUMS" "$output/board/"
''')
        self.output = self.root / 'build output'
        env = dict(self.env, SUITE_FIXTURES=str(self.root / 'fixture'), SUITE_FAIL=fail)
        return subprocess.run(['bash', str(self.scripts / 'build_transformer_suite.sh'), str(self.output),
                               '--toolkit-root', '/sdk path', '--pto-root', '/pto path'],
                              env=env, capture_output=True, text=True)

    def board(self, fail='', verify_only=False):
        output = self.root / ('board result' + fail)
        result = subprocess.run(['bash', str(self.output / 'board_suite/test_transformer_suite.sh'), str(output),
                                 '--runtime-env', '/board/runtime env.sh', '--device', '2'] +
                                (['--verify-only'] if verify_only else []),
                                env=dict(self.env, SUITE_FAIL=fail), capture_output=True, text=True)
        return result, output

    def test_builds_three_archives_and_executes_each_standalone_board_entry(self):
        built = self.build()
        self.assertEqual(built.returncode, 0, built.stderr)
        self.assertEqual(self.events.read_text().splitlines(), list(OPERATORS))
        with tarfile.open(self.output / 'board_suite.tar.gz') as tar:
            names = set(tar.getnames())
        self.assertIn('board_suite/test_transformer_suite.sh', names)
        self.assertTrue(all('board_suite/' + op + '/runtime.tar.gz' in names for op in OPERATORS))
        verified = subprocess.run(['sha256sum', '-c', 'board_suite.SHA256SUMS'], cwd=self.output,
                                  capture_output=True, text=True)
        self.assertEqual(verified.returncode, 0, verified.stderr)
        result, output = self.board()
        self.assertEqual(result.returncode, 0, result.stderr)
        for op in OPERATORS:
            self.assertTrue((output / op / 'result/validation.json').is_file())
        self.assertIn('All three operator tests passed', result.stdout)

    def test_compile_failure_stops_later_operators_and_never_emits_suite(self):
        built = self.build(fail='ffm_fused')
        self.assertEqual(built.returncode, 17)
        self.assertEqual(self.events.read_text().splitlines(), ['dec_enc_fused', 'ffm_fused'])
        self.assertFalse((self.output / 'board_suite.tar.gz').exists())

    def test_board_failure_stops_and_cannot_print_success(self):
        self.assertEqual(self.build().returncode, 0)
        result, output = self.board(fail='ffm_fused')
        self.assertEqual(result.returncode, 17)
        self.assertNotIn('All three operator tests passed', result.stdout)
        self.assertFalse((output / 'input_fused').exists())

    def test_corrupted_package_fails_checksum_before_launch(self):
        self.assertEqual(self.build().returncode, 0)
        (self.output / 'board_suite/dec_enc_fused/runtime.tar.gz').write_bytes(b'corrupt')
        result, output = self.board()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((output / 'dec_enc_fused').exists())

    def test_verify_only_never_claims_device_precision_passed(self):
        self.assertEqual(self.build().returncode, 0)
        result, _ = self.board(verify_only=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('device execution not run', result.stdout)
        self.assertNotIn('All three operator tests passed', result.stdout)


if __name__ == '__main__':
    unittest.main()
