"""Compile the three canonical 1952 ISP operators and package independent board tests."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile

from . import host, offline, offline_run
from .isp_1952_validation import ROOT, TRANSFORMER_ROOT
from .provider import resolve_gym_root

HERE = Path(__file__).resolve().parent
OPERATORS = ('dec_enc_fused', 'ffm_fused', 'input_fused')


def package(build, output):
    root, case, binding, _ = host.validate_build(build)
    if case['operator'] not in OPERATORS:
        raise ValueError('Expected one of the three transformer ISP operators')
    report = offline.package(root, output)
    output = Path(output).resolve()
    runtime = output / 'runtime'
    validation = runtime / 'validation'
    validation.mkdir()
    for name in ('case.json', 'binding.json'):
        shutil.copy2(root / 'case' / name, validation / name)
    for record in case['outputs']:
        shutil.copy2(host.relative_file(root / 'case', record['file']), validation / record['file'])
    shutil.copy2(HERE / 'host.py', runtime / 'acceptance.py')
    shutil.copy2(HERE / 'board_verify.py', runtime / 'board_verify.py')
    script_name = 'test_' + case['operator'] + '.sh'
    shutil.copy2(HERE.parent / 'scripts/board' / script_name, runtime / script_name)
    manifest = json.loads((runtime / 'runtime.json').read_text())
    manifest['validation_entry'] = script_name
    manifest['files'] = {p.relative_to(runtime).as_posix(): host.sha256(p)
                         for p in sorted(runtime.rglob('*')) if p.is_file() and p.name != 'runtime.json'}
    (runtime / 'runtime.json').write_text(json.dumps(manifest, indent=2) + '\n')
    offline_run.verify(runtime)
    archive = output / 'runtime.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(runtime, arcname='runtime')
    report.update(status='board-validation-package-built', validation_entry='runtime/' + script_name,
                  runtime_manifest_sha256=host.sha256(runtime / 'runtime.json'),
                  archive={'file': archive.name, 'sha256': host.sha256(archive), 'bytes': archive.stat().st_size})
    (output / 'package_result.json').write_text(json.dumps(report, indent=2) + '\n')
    (output / 'SHA256SUMS').write_text(report['archive']['sha256'] + '  runtime.tar.gz\n')
    print(json.dumps(report, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('build', 'package'))
    parser.add_argument('--op', choices=OPERATORS)
    parser.add_argument('--gym-root', type=Path, default=Path(os.environ.get('PYPTO_GYM_ROOT', HERE.parents[1] / 'pypto-gym')))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--build', type=Path)
    args, extra = parser.parse_known_args(argv)
    try:
        if args.command == 'build':
            if args.op is None:
                parser.error('build requires --op')
            from .run_profile import main as build
            gym = resolve_gym_root(args.gym_root)
            legacy = gym / TRANSFORMER_ROOT / args.op / 'provider.py'
            provider = legacy if legacy.is_file() else ROOT / 'provider.py'
            arguments = ['build', '--provider', str(provider), '--gym-root', str(gym),
                         '--op', args.op, '--height', '1080', '--width', '1920',
                         '--block-dim', '8', '--output', str(args.output), *extra]
            if build(arguments):
                return 1
            package(args.output, args.output / 'board')
        else:
            if args.build is None or extra:
                parser.error('package requires --build and no build options')
            package(args.build, args.output)
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print('Transformer workflow failed: ' + str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
