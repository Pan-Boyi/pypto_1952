"""Build and package the six FP16 ISP OMs with continuous CPU golden data."""

import argparse
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile

from . import board_chain, host
from .provider import load_provider

HERE = Path(__file__).resolve().parent
VALIDATION = Path('ops/pypto_pro/experimental/ops_transformer/isp_1952_validation')


def _compile(args, output, assets, layout):
    _, golden = load_provider(assets / 'chain_golden.py')
    golden.make_cases(output / 'golden', args.seed)
    _, _, height, width = layout.IMAGE_SHAPE
    for op in layout.STAGES:
        config = output / (op + '.json')
        config.write_text(json.dumps({'chain_case': str(output / 'golden' / op / 'case.json'),
                                      'seed': args.seed}) + '\n')
        log = output / (op + '.log')
        command = [sys.executable, '-B', '-m', 'pypto_1952.run_profile', 'build',
                   '--op', op, '--provider', str(assets / 'chain_provider.py'), '--case-config', str(config),
                   '--toolkit-root', str(args.toolkit_root), '--pto-root', str(args.pto_root),
                   '--target-soc', args.target_soc, '--height', str(height), '--width', str(width),
                   '--dtype', 'fp16', '--block-dim', '8', '--output', str(output / 'build' / op)]
        print('Building ' + op + ' (log: ' + str(log) + ')', flush=True)
        with log.open('w') as stream:
            result = subprocess.run(command, cwd=str(HERE.parent), stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError('OM build failed for ' + op + '. See ' + str(log))


def _package(args, output, assets, layout):
    release = output / 'isp_chain'
    release.mkdir()
    for source, name in ((HERE.parent / 'scripts/board/test_isp_chain.sh', 'run.sh'),
                         (HERE / 'board_chain.py', 'board_chain.py'),
                         (HERE / 'runner.py', 'runner.py'),
                         (HERE / 'host.py', 'acceptance.py'),
                         (assets / 'chain_layout.py', 'chain_layout.py')):
        shutil.copy2(source, release / name)
    for op in layout.STAGES:
        build, case, _, _ = host.validate_build(output / 'build' / op)
        destination = release / 'stages' / op
        destination.mkdir(parents=True)
        shutil.copy2(build / 'bundle/model.om', destination / 'model.om')
        for name in ['case.json', 'binding.json'] + [r['file'] for r in case['inputs'] + case['outputs']]:
            shutil.copy2(build / 'case' / name, destination / name)

    transformer = assets.parent
    sources = [transformer / op / (op + '.py') for op in layout.STAGES if op != 'GEMMFusedKernel']
    sources += [transformer / 'dec_enc_fused/brightness.py', transformer / 'postprocess/reference_nv12.py']
    sources += sorted(assets.glob('*.py'))
    gemm = args.gym_root.resolve() / 'ops/pro/experimental/gemm_fused_kernel'
    sources += [gemm / name for name in ('kernel.py', 'packing.py', 'reference.py', 'provider.py')]
    manifest = {'schema_version': 1, 'kind': 'pypto_1952_isp_chain', 'stages': list(layout.STAGES),
                'image_shape': list(layout.IMAGE_SHAPE), 'seed': args.seed, 'target_soc': args.target_soc,
                'brightness_rule': layout.BRIGHTNESS_RULE,
                'compiler_host': {'machine': platform.machine(), 'python': platform.python_version(),
                                  'toolkit_root': str(args.toolkit_root.resolve())},
                'gym_source_sha256': {str(path.relative_to(args.gym_root.resolve())): host.sha256(path)
                                      for path in sources},
                'files': {path.relative_to(release).as_posix(): host.sha256(path)
                          for path in sorted(release.rglob('*')) if path.is_file()}}
    (release / 'chain.json').write_text(json.dumps(manifest, indent=2) + '\n')
    board_chain.verify_release(release, layout)
    archive = output / 'isp_chain.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(release, arcname='isp_chain')
    (output / 'isp_chain.SHA256SUMS').write_text(host.sha256(archive) + '  ' + archive.name + '\n')
    print('Board package: ' + str(archive))
    print('Checksum: ' + str(output / 'isp_chain.SHA256SUMS'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gym-root', type=Path, required=True)
    parser.add_argument('--toolkit-root', type=Path, required=True)
    parser.add_argument('--pto-root', type=Path, required=True)
    parser.add_argument('--target-soc', default='MC62CM12AA')
    parser.add_argument('--seed', type=int, default=20261123)
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    args = parser.parse_args(argv)
    try:
        assets = args.gym_root.resolve(strict=True) / VALIDATION
        _, layout = load_provider(assets / 'chain_layout.py')
        output = args.output.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=False)
        _compile(args, output, assets, layout)
        _package(args, output, assets, layout)
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print('ISP chain build failed: ' + str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
