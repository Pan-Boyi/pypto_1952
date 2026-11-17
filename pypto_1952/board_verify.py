"""Portable on-board accuracy verification; Python 3.7.9+ and AscendCL only."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import acceptance
import offline_run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operator', required=True)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    try:
        manifest = offline_run.verify(root)
        case_file = root / 'validation/case.json'
        case = json.loads(case_file.read_text())
        binding = json.loads((root / 'validation/binding.json').read_text())
        request_file = root / 'bundle/execution.json'
        request = json.loads(request_file.read_text())
        if (case['operator'] != args.operator or manifest['operator'] != args.operator
                or acceptance.sha256(case_file) != binding['case_sha256']
                or binding['case_sha256'] != manifest['case_sha256']
                or binding['om_sha256'] != manifest['om_sha256']):
            raise ValueError('Validation contract does not match this operator and OM')
        for record in case['outputs']:
            path = acceptance.relative_file(root / 'validation', record['file'])
            if path.stat().st_size != record['bytes'] or acceptance.sha256(path) != record['sha256']:
                raise ValueError('Golden changed: ' + record['name'])
        if args.verify_only:
            print(json.dumps({'status': 'verified', 'operator': args.operator,
                              'om_sha256': manifest['om_sha256'], 'device_execution': 'not_run'}))
            return 0
        if args.output is None:
            parser.error('--output must be a new directory outside runtime/')
        args.output = args.output.expanduser().resolve()
        rc = offline_run.main(['--output', str(args.output), '--device', str(args.device)])
        if rc:
            return rc
        report = acceptance.evaluate_case(case, binding, request, root / 'validation',
                                          args.output / 'results', acceptance.sha256(request_file))
        report['precision_checked_on'] = 'board'
        (args.output / 'validation.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(args.operator + ': ' + report['status'].upper())
        for name, result in report['output_comparisons'].items():
            errors = [v['max_absolute_error'] for v in result['comparisons']]
            print('{}: max_abs_error={} mismatches={} finite={}'.format(
                name, None if any(value is None for value in errors) else max(errors),
                sum(v['mismatches'] for v in result['comparisons']),
                all(v['nonfinite'] == 0 for v in result['comparisons'])))
        print('Detailed result: ' + str(args.output / 'validation.json'))
        return 0 if report['status'] == 'passed' else 1
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print('Board validation failed: ' + str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
