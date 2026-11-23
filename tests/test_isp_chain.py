"""Regression tests for board handoffs, range selection and the final NV12 check."""

import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pypto_1952 import board_chain, host

LAYOUT = SimpleNamespace(STAGES=('preprocess', 'input_fused', 'GEMMFusedKernel',
                                 'dec_enc_fused', 'ffm_fused', 'postprocess'))
Y4 = [0.5, 0.625, 0.25, 0.375, 0.125, 0.75, 0.875, 1.]
UV = [0.5, 0.75, 0.625, 0.25]
NV12 = [0.5, 0.25, 0.625, 0.375, 0.125, 0.875, 0.75, 1., 0.5, 0.625, 0.75, 0.25]


def fp16(values):
    return struct.pack('<' + str(len(values)) + 'e', *values)


class ISPChainBoardTests(unittest.TestCase):
    def setUp(self):
        work = tempfile.TemporaryDirectory()
        self.addCleanup(work.cleanup)
        self.root = Path(work.name)
        self.manifest = {'brightness_rule': 'test', 'image_shape': [1, 4, 1, 2]}
        (self.root / 'chain.json').write_text(json.dumps(self.manifest))
        self.stages = {}
        self._add_stage('ffm_fused', {'x': ([1, 16, 1, 2], [0.] * 32)},
                        {'yout': ([1, 4, 1, 2], Y4), 'uvout': ([1, 2, 1, 2], UV)}, 0.003)
        self._add_stage('postprocess', {'y4': ([1, 4, 1, 2], Y4), 'uv': ([1, 2, 1, 2], UV)},
                        {'out': ([1, 12], NV12)}, 0.)

    def _add_stage(self, name, inputs, outputs, atol):
        directory = self.root / name
        directory.mkdir()
        (directory / 'model.om').write_bytes(name.encode())

        def records(values):
            for key, (shape, value) in values.items():
                path = directory / (key + '.bin')
                path.write_bytes(fp16(value))
                yield {'name': key, 'file': path.name, 'bytes': path.stat().st_size,
                       'shape': shape, 'dtype': 'fp16', 'sha256': host.sha256(path)}

        case = {'operator': name, 'inputs': list(records(inputs)), 'outputs': list(records(outputs)),
                'atol': atol, 'rtol': 0.}
        binding = {'om_sha256': host.sha256(directory / 'model.om')}
        request = {'om': {'file': 'model.om', 'sha256': binding['om_sha256']},
                   'outputs': [{'name': key, 'model_names': [key]} for key in outputs]}
        self.stages[name] = directory, case, binding, request

    def _run(self, start, bad_repack=False):
        destination = self.root / 'results'
        destination.mkdir()
        changed = 0.5 + 1 / 2048 if start == 'ffm_fused' else 0.5
        actual_y4 = fp16([changed] + Y4[1:])
        out = [changed] + NV12[1:]
        if bad_repack:
            out[1] += 1 / 4096  # within inherited FFM tolerance but incorrect Y packing
        called = []

        def execute(root, execution, result, device, log_path):
            del root, device, log_path
            op = result.parent.name
            called.append(op)
            request = json.loads(execution.read_text())
            _, case, binding, _ = self.stages[op]
            inputs = [dict(record, sha256=item['sha256'], acl_dtype=1)
                      for record, item in zip(case['inputs'], request['inputs'])]
            if op == 'postprocess':
                payload = execution.parent / request['inputs'][0]['file']
                self.assertEqual(payload.read_bytes(), actual_y4)
                self.assertEqual(request['inputs'][0]['sha256'], host.sha256(payload))
            values = (actual_y4, fp16(UV)) if op == 'ffm_fused' else (fp16(out),)
            result.mkdir()
            rounds = []
            for index, (stem, sentinel) in enumerate((('actual', 127), ('actual_second', 128)), 1):
                records = []
                for i, value in enumerate(values):
                    path = result / (stem + ('_' + str(i) if i else '') + '.bin')
                    path.write_bytes(value)
                    records.append({'file': path.name, 'bytes': len(value), 'sha256': host.sha256(path)})
                rounds.append({'index': index, 'sentinel': sentinel, 'outputs': records})
            report = {'schema_version': 1, 'kind': 'pypto_pro_execution_result', 'status': 'executed',
                      'request_sha256': host.sha256(execution), 'om_sha256': binding['om_sha256'],
                      'execution_calls': 2, 'inputs': inputs,
                      'outputs': [dict(record, model_name=record['name'], acl_dtype=1) for record in case['outputs']],
                      'rounds': rounds}
            (result / 'execution.json').write_text(json.dumps(report))

        span = board_chain.selected_stages(LAYOUT, start, 'postprocess')
        with patch.object(board_chain, '_run_stage', side_effect=execute):
            verdict = board_chain.run_chain(self.root, destination, self.manifest, self.stages, LAYOUT, span, 0)
        self.assertEqual(called, list(span))
        post = json.loads((destination / 'postprocess/validation.json').read_text())
        return verdict, post

    def test_range_uses_actual_inputs_and_requires_exact_nv12_repack(self):
        verdict, post = self._run('ffm_fused')
        self.assertEqual(verdict['status'], 'passed')
        self.assertEqual(post['input_sources']['y4']['source'], 'ffm_fused.yout')
        self.assertEqual(post['atol'], 0.003)
        self.assertTrue(post['exact_device_yuv_repack']['passed'])

    def test_single_operator_uses_packaged_inputs_and_exact_golden(self):
        verdict, post = self._run('postprocess')
        self.assertEqual(verdict['status'], 'passed')
        self.assertEqual(post['input_sources']['y4']['source'], 'packaged_cpu_input')
        self.assertEqual(post['atol'], 0.)

    def test_wrong_repack_fails_even_within_chain_tolerance(self):
        verdict, post = self._run('ffm_fused', bad_repack=True)
        self.assertTrue(post['comparisons'][0]['passed'])
        self.assertEqual(verdict['first_failed'], 'postprocess')
        self.assertEqual(verdict['status'], 'failed')

    def test_reversed_ranges_fail(self):
        with self.assertRaisesRegex(ValueError, '--start must be at or before --end'):
            board_chain.selected_stages(LAYOUT, 'postprocess', 'ffm_fused')


if __name__ == '__main__':
    unittest.main()
