"""Execute and validate a compiled ISP span with one static OM."""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

if __package__:
    from . import host as acceptance
else:
    # The release supports Python -I -S on the board.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import acceptance


def verify_release(root, layout):
    """Check the single OM, its case and the package inventory."""
    manifest = json.loads((root / 'chain.json').read_text())
    if (manifest.get('kind') != 'pypto_1952_isp_chain' or manifest.get('schema_version') != 2
            or tuple(manifest['image_shape']) != layout.IMAGE_SHAPE
            or manifest['brightness_rule'] != layout.BRIGHTNESS_RULE):
        raise ValueError('Expected a single-OM ISP chain package')
    stages = layout.selected_stages(manifest['start'], manifest['end'])
    if tuple(manifest['stages']) != stages:
        raise ValueError('Package stages differ from its compiled start/end')

    files = manifest['files']
    required = {'run.sh', 'board_chain.py', 'chain_layout.py', 'runner.py', 'acceptance.py',
                'model.om', 'case.json', 'binding.json', 'execution.json'}
    if not isinstance(files, dict) or not required <= files.keys():
        raise ValueError('Incomplete ISP chain release')
    actual = set()
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('ISP chain release contains a symlink')
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != set(files) | {'chain.json'}:
        raise ValueError('ISP chain release has missing or unlisted files')
    if {name for name in files if name.endswith('.om')} != {'model.om'}:
        raise ValueError('An ISP span must contain exactly one model.om')
    for name, expected in files.items():
        if acceptance.sha256(acceptance.relative_file(root, name)) != expected:
            raise ValueError('ISP chain SHA256 mismatch: ' + name)

    case = json.loads((root / 'case.json').read_text())
    binding = json.loads((root / 'binding.json').read_text())
    request = acceptance.create_execution(root / 'case.json', root / 'binding.json')
    config = case['configuration']
    if (case['schema_version'] != 1 or case['operator'] != 'ISPChain'
            or config['fixture'] != 'isp_chain_single_om_fp16'
            or config['brightness_rule'] != manifest['brightness_rule']
            or [config['height'], config['width']] != list(layout.IMAGE_SHAPE[2:])
            or any(config[key] != manifest[key] for key in ('seed', 'start', 'end', 'stages'))
            or binding['om_sha256'] != files['model.om']
            or binding['target_soc'] != manifest['target_soc']
            or request != json.loads((root / 'execution.json').read_text())):
        raise ValueError('Case, request and OM binding disagree')
    seen = set()
    for record in case['inputs'] + case['outputs']:
        filename = record['name'] + '.bin'
        if (Path(filename).name != filename or record['file'] != filename or filename in seen
                or files.get(filename) != record['sha256']
                or (root / filename).stat().st_size != record['bytes']
                or acceptance.tensor_bytes(record) != record['bytes']):
            raise ValueError('Invalid case tensor: ' + record['name'])
        seen.add(filename)
    if any(not math.isfinite(case[key]) or case[key] < 0 for key in ('atol', 'rtol')):
        raise ValueError('Invalid comparison tolerance')
    return manifest, case, binding, request


def _run_model(root, output, device):
    command = [sys.executable, '-I', '-S', '-B', str(root / 'runner.py'), '--execution',
               str(root / 'execution.json'), '--output', str(output / 'results'), '--device', str(device)]
    with (output / 'runner.log').open('w') as log:
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
    if code:
        execution = output / 'results/execution.json'
        error = json.loads(execution.read_text()).get('error') if execution.is_file() else None
        raise RuntimeError((error or 'ACL runner exit code ' + str(code)) + '. See ' + str(output / 'runner.log'))


def run_chain(root, output, manifest, case, binding, request, device):
    """Run the same OM for both sentinel rounds, then compare the end output."""
    try:
        _run_model(root, output, device)
        report = acceptance.evaluate_case(case, binding, request, root, output / 'results',
                                          acceptance.sha256(root / 'execution.json'))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, OverflowError) as error:
        report = {'schema_version': 1, 'status': 'failed', 'error': str(error)}
    report.update(start=manifest['start'], end=manifest['end'], stages=manifest['stages'],
                  om_sha256=binding['om_sha256'], device=device, precision_checked_on='board')
    (output / 'validation.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def main(layout, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', choices=layout.STAGES, help='Require the compiled start to match')
    parser.add_argument('--end', choices=layout.STAGES, help='Require the compiled end to match')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--output', type=Path, help='New result directory outside the release')
    parser.add_argument('--verify-only', action='store_true', help='Validate package and case without ACL')
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    try:
        if sys.version_info < (3, 7, 9) or args.device < 0:
            raise ValueError('Need Python 3.7.9 or later and a nonnegative device ID')
        manifest, case, binding, request = verify_release(root, layout)
        if ((args.start is not None and args.start != manifest['start'])
                or (args.end is not None and args.end != manifest['end'])):
            raise ValueError('The board flags must match the compiled span. Select a new span on the compiler host.')
        if args.verify_only:
            print(json.dumps({'status': 'verified', 'selected_stages': manifest['stages'],
                              'om_sha256': binding['om_sha256'], 'device_execution': 'not_run'}))
            return 0
        if args.output is None:
            parser.error('--output is required for device execution')
        output = args.output.expanduser().resolve()
        if output == root or root in output.parents:
            raise ValueError('Place results outside the extracted release')
        output.mkdir(parents=True, exist_ok=False)
        report = run_chain(root, output, manifest, case, binding, request, args.device)
        summary = {'status': report['status'], 'start': manifest['start'], 'end': manifest['end'],
                   'report': str(output / 'validation.json')}
        if 'error' in report:
            summary['error'] = report['error']
        print(json.dumps(summary))
        return 0 if report['status'] == 'passed' else 1
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print('ISP chain validation failed: ' + str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    import chain_layout

    raise SystemExit(main(chain_layout))
