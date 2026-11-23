"""Validate an inclusive range of ISP OMs using AscendCL and the Python standard library."""

import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

if __package__:
    from . import host as acceptance
else:
    # The release can run with Python -I -S on the board.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import acceptance


DIRECT_INPUTS = {
    'input_fused': {'x': ('preprocess', 'out')},
    'dec_enc_fused': {'x': ('preprocess', 'out')},
    'ffm_fused': {'x': ('dec_enc_fused', 'output')},
    'postprocess': {'y4': ('ffm_fused', 'yout'), 'uv': ('ffm_fused', 'uvout')},
}


def selected_stages(layout, start, end):
    first, last = layout.STAGES.index(start), layout.STAGES.index(end)
    if first > last:
        raise ValueError('--start must be at or before --end')
    return layout.STAGES[first:last + 1]


def _write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def linked_inputs(layout, op, outputs):
    """Return (source name, payload) for producers that ran in the selected range."""
    linked = {}
    for name, (producer, output) in DIRECT_INPUTS.get(op, {}).items():
        if producer in outputs:
            linked[name] = (producer + '.' + output, outputs[producer][output])
    if op == 'GEMMFusedKernel' and 'input_fused' in outputs:
        features = outputs['input_fused']['features'].read_bytes()
        linked.update((name, ('input_fused.features', payload))
                      for name, payload in layout.head_inputs(features).items())
    if op == 'dec_enc_fused' and 'GEMMFusedKernel' in outputs:
        y = outputs['GEMMFusedKernel']['y'].read_bytes()
        linked.update((name, ('GEMMFusedKernel.y', payload))
                      for name, payload in layout.dec_coefficients(y).items())
    return linked


def _abi(record):
    return record['dtype'], tuple(record['shape'])


def verify_release(root, layout):
    """Verify inventory, each case/OM binding and every CPU golden handoff."""
    manifest = json.loads((root / 'chain.json').read_text())
    if (manifest.get('kind') != 'pypto_1952_isp_chain' or manifest.get('schema_version') != 1
            or manifest['stages'] != list(layout.STAGES) or tuple(manifest['image_shape']) != layout.IMAGE_SHAPE
            or manifest['brightness_rule'] != layout.BRIGHTNESS_RULE):
        raise ValueError('Unsupported ISP chain manifest')
    files = manifest['files']
    if not {'run.sh', 'board_chain.py', 'chain_layout.py', 'runner.py', 'acceptance.py'} <= files.keys():
        raise ValueError('Incomplete ISP chain release')
    actual = set()
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('ISP chain release contains a symlink')
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != set(files) | {'chain.json'}:
        raise ValueError('ISP chain release has missing or unlisted files')
    for name, expected in files.items():
        if acceptance.sha256(acceptance.relative_file(root, name)) != expected:
            raise ValueError('ISP chain SHA256 mismatch: ' + name)

    stages, goldens = {}, {}
    for op in layout.STAGES:
        prefix = 'stages/' + op
        directory = root / prefix
        case = json.loads((directory / 'case.json').read_text())
        binding = json.loads((directory / 'binding.json').read_text())
        request = acceptance.create_execution(directory / 'case.json', directory / 'binding.json')
        cfg = case['configuration']
        if (case['schema_version'] != 1 or case['operator'] != op
                or cfg['fixture'] != 'isp_chain_fp16' or cfg['brightness_rule'] != layout.BRIGHTNESS_RULE
                or [cfg['height'], cfg['width']] != list(layout.IMAGE_SHAPE[2:]) or cfg['seed'] != manifest['seed']
                or binding['om_sha256'] != files[prefix + '/model.om']
                or binding['target_soc'] != manifest['target_soc']):
            raise ValueError('Case and OM binding disagree: ' + op)
        names = set()
        for record in case['inputs'] + case['outputs']:
            filename = record['name'] + '.bin'
            if (Path(filename).name != filename or record['file'] != filename or filename in names
                    or files[prefix + '/' + filename] != record['sha256']
                    or (directory / filename).stat().st_size != record['bytes']
                    or acceptance.tensor_bytes(record) != record['bytes']):
                raise ValueError('Invalid tensor in ' + op + ': ' + record['name'])
            names.add(filename)
        if any(not math.isfinite(case[key]) or case[key] < 0 for key in ('atol', 'rtol')):
            raise ValueError('Invalid comparison tolerance: ' + op)
        stages[op] = (directory, case, binding, request)
        goldens[op] = {r['name']: directory / r['file'] for r in case['outputs']}

    for op in layout.STAGES:
        inputs = {record['name']: record for record in stages[op][1]['inputs']}
        for name, (producer, output) in DIRECT_INPUTS.get(op, {}).items():
            src = next(r for r in stages[producer][1]['outputs'] if r['name'] == output)
            if _abi(src) != _abi(inputs[name]) or src['dtype'] != 'fp16':
                raise ValueError('FP16 handoff ABI differs: ' + producer + ' -> ' + op + '.' + name)
        if op == 'GEMMFusedKernel':
            for index in range(8):
                if _abi(inputs['x' + str(index)]) != ('fp16', (1 if index < 4 else 16, 128)):
                    raise ValueError('Invalid FP16 GEMMFusedKernel head input shape')
        if op == 'dec_enc_fused':
            for name, shape in layout.DEC_COEFF_SHAPES.items():
                if _abi(inputs[name]) != ('fp16', shape):
                    raise ValueError('Invalid FP16 DecEnc coefficient shape: ' + name)
        for name, (source, payload) in linked_inputs(layout, op, goldens).items():
            digest = acceptance.sha256(payload) if isinstance(payload, Path) else hashlib.sha256(payload).hexdigest()
            if digest != inputs[name]['sha256']:
                raise ValueError('Golden handoff differs: ' + source + ' -> ' + op + '.' + name)
    for op, name, shape in (('input_fused', 'features', layout.FEATURES_SHAPE),
                            ('GEMMFusedKernel', 'y', layout.GEMM_OUTPUT_SHAPE)):
        record = next(r for r in stages[op][1]['outputs'] if r['name'] == name)
        if _abi(record) != ('fp16', shape):
            raise ValueError('Invalid FP16 bridge output shape: ' + op)
    return manifest, stages


def _run_stage(root, execution, result, device, log_path):
    command = [sys.executable, '-I', '-S', '-B', str(root / 'runner.py'), '--execution', str(execution),
               '--output', str(result), '--device', str(device)]
    with log_path.open('w') as log:
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
    if code:
        report = result / 'execution.json'
        error = json.loads(report.read_text()).get('error') if report.is_file() else None
        raise RuntimeError((error or 'ACL runner exit code ' + str(code)) + '. See ' + str(log_path))


def _nv12_from_actual(y4, uv, height, width, output):
    """Compute the exact NV12 layout from the board FFM outputs."""
    def samples(path):
        words = array('H')
        words.frombytes(path.read_bytes())
        if sys.byteorder != 'little':
            words.byteswap()
        # Binary16 values are finite here; clamp in sample order without a dtype cast.
        return array('H', (0 if word & 0x8000 else min(word, 0x3C00) for word in words))

    y4, uv = samples(y4), samples(uv)
    pixels = height * width
    line = array('H', [0]) * (2 * width)

    def interleave(stream, planes, first, second, row):
        a, b = first * pixels + row * width, second * pixels + row * width
        line[::2] = planes[a:a + width]
        line[1::2] = planes[b:b + width]
        if sys.byteorder != 'little':
            line.byteswap()
        stream.write(line.tobytes())

    with output.open('wb') as stream:
        for row in range(height):
            interleave(stream, y4, 0, 1, row)
            interleave(stream, y4, 2, 3, row)
        for row in range(height):
            interleave(stream, uv, 0, 1, row)


def run_chain(root, output, manifest, stages, layout, span, device):
    """Compare each executed output to the continuous CPU golden and route real outputs."""
    report = {'schema_version': 1, 'status': 'failed', 'device': device, 'start': span[0], 'end': span[-1],
              'precision_checked_on': 'board', 'manifest_sha256': acceptance.sha256(root / 'chain.json'),
              'brightness_rule': manifest['brightness_rule'], 'stages': {}, 'first_failed': None}
    outputs = {}
    try:
        for op in span:
            directory, case, binding, request = stages[op]
            stage = output / op
            request_dir = stage / 'request'
            request_dir.mkdir(parents=True)
            shutil.copyfile(directory / 'model.om', request_dir / 'model.om')
            linked = linked_inputs(layout, op, outputs)
            inputs, sources = [], {}
            for record in case['inputs']:
                name = record['name']
                source, payload = linked.get(name, ('packaged_cpu_input', directory / record['file']))
                destination = request_dir / record['file']
                if isinstance(payload, Path):
                    shutil.copyfile(payload, destination)
                else:
                    destination.write_bytes(payload)
                if destination.stat().st_size != record['bytes']:
                    raise ValueError('Invalid handoff size for ' + op + '.' + name)
                digest = acceptance.sha256(destination)
                inputs.append(dict(record, sha256=digest))
                sources[name] = {'source': source, 'sha256': digest}

            execution = request_dir / 'execution.json'
            request = dict(request, inputs=[{key: record[key] for key in ('name', 'file', 'sha256')}
                                            for record in inputs])
            _write_json(execution, request)
            result = stage / 'results'
            _run_stage(root, execution, result, device, stage / 'runner.log')
            compared_case = dict(case, inputs=inputs)
            if op == 'postprocess' and 'ffm_fused' in outputs:
                # Packing is exact, but its CPU chain golden also includes the FFM numerical tolerance.
                ffm_case = stages['ffm_fused'][1]
                compared_case.update(atol=ffm_case['atol'], rtol=ffm_case['rtol'])
            verdict = acceptance.evaluate_case(compared_case, binding, request, directory, result,
                                               acceptance.sha256(execution))
            verdict.update(precision_checked_on='board', input_sources=sources)
            actual = {record['name']: result / ('actual' + ('_' + str(i) if i else '') + '.bin')
                      for i, record in enumerate(case['outputs'])}
            if op == 'postprocess' and 'ffm_fused' in outputs:
                expected = stage / 'nv12_from_device_yuv.bin'
                yuv = outputs['ffm_fused']
                _nv12_from_actual(yuv['yout'], yuv['uvout'], *manifest['image_shape'][2:], expected)
                verdict['exact_device_yuv_repack'] = acceptance.compare_files(actual['out'], expected, 0, 0, 'fp16')
                if not verdict['exact_device_yuv_repack']['passed']:
                    verdict['status'] = 'failed'
            _write_json(stage / 'validation.json', verdict)
            report['stages'][op] = {'status': verdict['status'], 'report': op + '/validation.json',
                                    'om_sha256': binding['om_sha256']}
            print(op + ': ' + verdict['status'].upper(), flush=True)
            if verdict['status'] != 'passed' and report['first_failed'] is None:
                report['first_failed'] = op
            if not verdict['dual_sentinel_consistent'] or any(v['nonfinite'] for v in verdict['comparisons']):
                raise ValueError(op + ' output is incomplete, nondeterministic or nonfinite')
            outputs[op] = actual
        if report['first_failed'] is None:
            report['status'] = 'passed'
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, OverflowError) as error:
        report['first_failed'] = report['first_failed'] or op
        report['error'] = str(error)
        print('Board execution stopped at ' + op + ': ' + str(error), file=sys.stderr)
    _write_json(output / 'validation.json', report)
    return report


def main(layout, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', choices=layout.STAGES, default=layout.STAGES[0])
    parser.add_argument('--end', choices=layout.STAGES, default=layout.STAGES[-1])
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--output', type=Path, help='New result directory outside the release')
    parser.add_argument('--verify-only', action='store_true', help='Validate package and CPU handoffs without ACL')
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    try:
        if sys.version_info < (3, 7, 9) or args.device < 0:
            raise ValueError('Need Python 3.7.9 or later and a nonnegative device ID')
        span = selected_stages(layout, args.start, args.end)
        manifest, stages = verify_release(root, layout)
        if args.verify_only:
            print(json.dumps({'status': 'verified', 'selected_stages': span, 'device_execution': 'not_run'}))
            return 0
        if args.output is None:
            parser.error('--output is required for device execution')
        output = args.output.expanduser().resolve()
        if output == root or root in output.parents:
            raise ValueError('Place results outside the extracted release')
        output.mkdir(parents=True, exist_ok=False)
        report = run_chain(root, output, manifest, stages, layout, span, args.device)
        print(json.dumps({'status': report['status'], 'first_failed': report['first_failed'],
                          'report': str(output / 'validation.json')}))
        return 0 if report['status'] == 'passed' else 1
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print('ISP chain validation failed: ' + str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    import chain_layout

    raise SystemExit(main(chain_layout))
