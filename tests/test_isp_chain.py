"""Regression tests for the portable single-OM ISP validation package."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pypto_1952 import board_chain, host

SPAN = ('input_fused', 'GEMMFusedKernel', 'dec_enc_fused', 'ffm_fused')
IMAGE_SHAPE = (1, 4, 1080, 1920)
BRIGHTNESS_RULE = 'test_mean'
LAYOUT = SimpleNamespace(STAGES=('preprocess', *SPAN, 'postprocess'), IMAGE_SHAPE=IMAGE_SHAPE,
                         BRIGHTNESS_RULE=BRIGHTNESS_RULE, selected_stages=lambda start, end: SPAN)


def write_json(path, value):
    path.write_text(json.dumps(value) + '\n')


class ISPChainBoardTests(unittest.TestCase):
    def setUp(self):
        work = tempfile.TemporaryDirectory()
        self.addCleanup(work.cleanup)
        self.root = Path(work.name) / 'isp_chain'
        self.root.mkdir()
        self.output = self.root.parent / 'results'
        self.output.mkdir()
        for name in ('run.sh', 'board_chain.py', 'chain_layout.py', 'runner.py', 'acceptance.py'):
            (self.root / name).touch()
        (self.root / 'model.om').write_bytes(b'single static OM')

        def tensor(name, value):
            path = self.root / (name + '.bin')
            path.write_bytes(struct.pack('<e', value))
            return {'name': name, 'file': path.name, 'bytes': 2, 'shape': [1], 'dtype': 'fp16',
                    'sha256': host.sha256(path)}

        case = {'schema_version': 1, 'operator': 'ISPChain', 'inputs': [tensor('input_fused_x', 1.)],
                'outputs': [tensor('ffm_fused_yout', 0.25), tensor('ffm_fused_uvout', 0.5)],
                'atol': 0.003, 'rtol': 0.003}
        self.manifest = {'schema_version': 2, 'kind': 'pypto_1952_isp_chain', 'stages': list(SPAN),
                         'start': SPAN[0], 'end': SPAN[-1], 'seed': 1, 'image_shape': list(IMAGE_SHAPE),
                         'brightness_rule': BRIGHTNESS_RULE, 'target_soc': 'MC62CM12AA'}
        case['configuration'] = dict({key: self.manifest[key] for key in ('seed', 'start', 'end', 'stages')},
                                     height=1080, width=1920, brightness_rule=BRIGHTNESS_RULE,
                                     fixture='isp_chain_single_om_fp16')
        write_json(self.root / 'case.json', case)
        binding = {'om_sha256': host.sha256(self.root / 'model.om'),
                   'case_sha256': host.sha256(self.root / 'case.json'), 'runtime_dependencies': 'device_kernel_only',
                   'target_soc': 'MC62CM12AA',
                   'output_name_bindings': [{'logical_name': name, 'om_declared_names': [name]}
                                            for name in ('ffm_fused_yout', 'ffm_fused_uvout')]}
        write_json(self.root / 'binding.json', binding)
        request = host.create_execution(self.root / 'case.json', self.root / 'binding.json')
        write_json(self.root / 'execution.json', request)
        self._inventory()

    def _inventory(self):
        self.manifest['files'] = {p.name: host.sha256(p) for p in self.root.iterdir() if p.name != 'chain.json'}
        write_json(self.root / 'chain.json', self.manifest)

    def _exercise(self, bad_precision=False):
        manifest, case, binding, request = board_chain.verify_release(self.root, LAYOUT)

        def execute(root, output, device):
            self.assertEqual(list(root.glob('*.om')), [root / 'model.om'])
            self.assertEqual(device, 0)
            result = output / 'results'
            result.mkdir()
            rounds = []
            for index, (stem, sentinel) in enumerate((('actual', 127), ('actual_second', 128)), 1):
                records = []
                for i, expected in enumerate(case['outputs']):
                    payload = (root / expected['file']).read_bytes()
                    if bad_precision and i == 1:
                        payload = struct.pack('<e', 0.75)
                    binary = result / (stem + ('_' + str(i) if i else '') + '.bin')
                    binary.write_bytes(payload)
                    records.append({'file': binary.name, 'bytes': 2, 'sha256': host.sha256(binary)})
                rounds.append({'index': index, 'sentinel': sentinel, 'outputs': records})
            write_json(result / 'execution.json', {
                'schema_version': 1, 'kind': 'pypto_pro_execution_result', 'status': 'executed',
                'request_sha256': host.sha256(root / 'execution.json'), 'om_sha256': binding['om_sha256'],
                'execution_calls': 2, 'inputs': [dict(item, acl_dtype=1) for item in case['inputs']],
                'outputs': [dict(item, model_name=item['name'], acl_dtype=1) for item in case['outputs']],
                'rounds': rounds,
            })

        with patch.object(board_chain, '_run_model', side_effect=execute) as run:
            verdict = board_chain.run_chain(self.root, self.output, manifest, case, binding, request, 0)
        run.assert_called_once_with(self.root, self.output, 0)
        self.assertEqual(verdict, json.loads((self.output / 'validation.json').read_text()))
        self.assertEqual(verdict['stages'], list(SPAN))
        return verdict

    def test_one_om_compares_both_final_outputs(self):
        verdict = self._exercise()
        self.assertEqual(verdict['status'], 'passed')
        self.assertEqual(list(verdict['output_comparisons']), ['ffm_fused_yout', 'ffm_fused_uvout'])

    def test_final_output_mismatch_fails_the_span(self):
        self.assertEqual(self._exercise(bad_precision=True)['status'], 'failed')

    def test_multiple_oms_rejected_even_if_listed_in_inventory(self):
        (self.root / 'additional.om').write_bytes(b'extra OM')
        self._inventory()
        with self.assertRaisesRegex(ValueError, 'exactly one model.om'):
            board_chain.verify_release(self.root, LAYOUT)

    def test_board_flags_only_check_compiled_endpoints(self):
        with patch.object(board_chain, '__file__', str(self.root / 'board_chain.py')):
            with redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(board_chain.main(LAYOUT, ['--verify-only']), 0)
            self.assertEqual(json.loads(stdout.getvalue())['selected_stages'], list(SPAN))
            with redirect_stderr(io.StringIO()) as stderr:
                self.assertEqual(board_chain.main(LAYOUT, ['--verify-only', '--end', 'postprocess']), 1)
            self.assertIn('must match the compiled span', stderr.getvalue())


if __name__ == '__main__':
    unittest.main()
