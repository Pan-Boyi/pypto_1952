"""Exercise actual multi-output execution and board acceptance with synthetic ACL memory."""

from contextlib import redirect_stderr, redirect_stdout
import ctypes as ct
import importlib.util
import io
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import unittest
from unittest.mock import patch

from pypto_1952 import host, offline_run, runner, transformer_ops
from tests import test_host as fixtures
from tests.test_host import half
from tests.test_offline import ArchiveACL


class TwoOutputACL(ArchiveACL):
    def aclmdlGetNumOutputs(self, _):
        return 2

    def aclmdlGetOutputNameByIndex(self, _, index):
        return b'wrong' if self.mismatch == 'output_name' else (b'z', b'uv')[index]

    def aclmdlGetInputDataType(self, _, index):
        return 3 if index == 1 else 1

    def aclmdlGetInputSizeByIndex(self, _, index):
        return 16 if index == 1 else 8

    def aclmdlGetOutputDataType(self, *_):
        return 1

    def aclmdlExecute(self, _, inputs, outputs):
        self.executions += 1
        y = struct.unpack('<4e', ct.string_at(*self.buffers[self.datasets[inputs][0]]))
        bias = struct.unpack('<4i', ct.string_at(*self.buffers[self.datasets[inputs][1]]))
        values = [a + b / 65536 for a, b in zip(y, bias)]
        for index in range(1 if self.mode == 'no_write_uv' else 2):
            payload = half([v if index == 0 else -v for v in values])
            address, size = self.buffers[self.datasets[outputs][index]]
            ct.memmove(address, payload, size)
        return 0


class TransformerBoardTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.HostTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        bias = f.case_dir / 'x.bin'
        bias.write_bytes(struct.pack('<4i', 65536, -65536, 32768, 131072))
        f.case['inputs'][0].update(dtype='int32', bytes=16, sha256=host.sha256(bias))
        (f.case_dir / 'z.bin').write_bytes(half([6, 5, 7.5, 10]))
        (f.case_dir / 'uv.bin').write_bytes(half([-6, -5, -7.5, -10]))
        f.case['outputs'][0]['sha256'] = host.sha256(f.case_dir / 'z.bin')
        f.case['outputs'].append({**f.case['outputs'][0], 'name': 'uv', 'file': 'uv.bin',
                                  'sha256': host.sha256(f.case_dir / 'uv.bin')})
        f.case['operator'] = 'ffm_fused'
        f.case_path.write_text(json.dumps(f.case))
        f.binding['case_sha256'] = host.sha256(f.case_path)
        f.binding['output_name_bindings'].append({'logical_name': 'uv', 'om_declared_names': ['uv']})
        f.binding_path.write_text(json.dumps(f.binding))
        shutil.rmtree(f.bundle)
        run_profile = __import__('pypto_1952.run_profile', fromlist=['_make_bundle'])
        run_profile._make_bundle(f.case_path, f.om, f.binding_path, f.bundle)
        (f.build / 'build_result.json').write_text(json.dumps({'status': 'bundle-built', 'target_soc': 'MC62CM12AA'}))

    def execute(self, mode='normal', mismatch=None):
        acl = TwoOutputACL(mode=mode, mismatch=mismatch)
        result = self.fixture.root / ('execution-' + mode + str(mismatch))
        with patch.object(runner, 'load_acl', return_value=acl), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            rc = runner.main(['--execution', str(self.fixture.request_path), '--output', str(result)])
        self.assertEqual(acl.memory, {})
        self.assertEqual(acl.events[-1], 'finalize')
        return rc, result

    def test_q16_inputs_and_both_outputs_pass_actual_runner_and_acceptance(self):
        rc, result = self.execute()
        self.assertEqual(rc, 0)
        report = host.evaluate(self.fixture.build, result)
        self.assertEqual(report['status'], 'passed')
        self.assertEqual(set(report['output_comparisons']), {'z', 'uv'})
        execution = json.loads((result / 'execution.json').read_text())
        self.assertEqual(execution['inputs'][1]['acl_dtype'], 3)
        self.assertEqual(len(execution['outputs']), 2)
        self.assertEqual((result / 'actual_1.bin').read_bytes(), half([-6, -5, -7.5, -10]))

    def test_second_unwritten_output_cannot_pass(self):
        rc, result = self.execute(mode='no_write_uv')
        self.assertEqual(rc, 0)
        report = host.evaluate(self.fixture.build, result)
        self.assertEqual(report['status'], 'failed')
        self.assertFalse(report['output_comparisons']['uv']['dual_sentinel_consistent'])

    def test_wrong_output_name_is_rejected_before_execution_and_cleans_up(self):
        rc, result = self.execute(mismatch='output_name')
        self.assertEqual(rc, 1)
        self.assertEqual(json.loads((result / 'execution.json').read_text())['execution_calls'], 0)

    def test_board_package_has_isolated_verification_and_real_dual_output_acceptance(self):
        _, result = self.execute()
        package = self.fixture.root / 'board-package'
        with redirect_stdout(io.StringIO()):
            transformer_ops.package(self.fixture.build, package)
        runtime = package / 'runtime'
        checked = subprocess.run([sys.executable, '-I', '-S', '-B', str(runtime / 'board_verify.py'),
                                  '--operator', 'ffm_fused', '--verify-only'], capture_output=True, text=True)
        self.assertEqual(checked.returncode, 0, checked.stderr)
        self.assertEqual(json.loads(checked.stdout)['status'], 'verified')
        with patch.dict(sys.modules, {'acceptance': host, 'offline_run': offline_run}):
            spec = importlib.util.spec_from_file_location('_test_board_verify', runtime / 'board_verify.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        output = self.fixture.root / 'board-result'

        def board_execute(args):
            output.mkdir()
            shutil.copytree(result, output / 'results')
            return 0

        with patch.object(offline_run, 'main', side_effect=board_execute), redirect_stdout(io.StringIO()):
            self.assertEqual(module.main(['--operator', 'ffm_fused', '--output', str(output)]), 0)
        verdict = json.loads((output / 'validation.json').read_text())
        self.assertEqual(verdict['status'], 'passed')
        self.assertEqual(verdict['precision_checked_on'], 'board')
        (runtime / 'validation/uv.bin').write_bytes(b'changed')
        with redirect_stderr(io.StringIO()):
            self.assertEqual(module.main(['--operator', 'ffm_fused', '--verify-only']), 1)


if __name__ == '__main__':
    unittest.main()
