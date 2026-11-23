"""Compile one static 1952 OM for a selected range of the six FP16 ISP operators."""

import argparse
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile

from . import board_chain, host
from .isp_1952_validation import GEMM_KERNEL, ROOT, TRANSFORMER_ROOT, chain_layout as layout
from .provider import resolve_gym_root

HERE = Path(__file__).resolve().parent


def _compile(args, output, work, stages):
    config = work / 'span.json'
    config.write_text(json.dumps({'start': stages[0], 'end': stages[-1], 'seed': args.seed}) + '\n')
    _, _, height, width = layout.IMAGE_SHAPE
    build = work / 'build'
    command = [sys.executable, '-B', '-m', 'pypto_1952.run_profile', 'build',
               '--op', 'ISPChain', '--provider', str(ROOT / 'chain_provider.py'), '--case-config', str(config),
               '--gym-root', str(args.gym_root),
               '--toolkit-root', str(args.toolkit_root), '--pto-root', str(args.pto_root),
               '--target-soc', args.target_soc, '--height', str(height), '--width', str(width),
               '--dtype', 'fp16', '--block-dim', '8', '--output', str(build)]
    log = output / 'build.log'
    print('Building one OM for ' + ' -> '.join(stages) + ' (log: ' + str(log) + ')', flush=True)
    try:
        with log.open('w') as stream:
            subprocess.run(command, cwd=str(HERE.parent), stdout=stream, stderr=subprocess.STDOUT, check=True)
    finally:
        # The compiler build stages additional copies of model.om for the
        # generic runner. Retain diagnostics from that workspace.
        evidence = output / 'build_evidence'
        for item in (list(build.glob('atc/*.log')) + list(build.glob('native/*/*.log'))
                     + list(build.glob('atc/om_audit.json')) + list(build.glob('atc/adapter_manifest.json'))):
            destination = evidence / item.relative_to(build)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, destination)
    return build


def _package(args, output, work, build, stages):
    build, case, _, _ = host.validate_build(build)
    release = work / 'isp_chain'
    release.mkdir()
    for source, name in ((HERE.parent / 'scripts/board/test_isp_chain.sh', 'run.sh'),
                         (HERE / 'board_chain.py', 'board_chain.py'),
                         (HERE / 'host.py', 'acceptance.py'),
                         (ROOT / 'chain_layout.py', 'chain_layout.py')):
        shutil.copy2(source, release / name)
    for name in ('model.om', 'runner.py', 'execution.json'):
        shutil.copy2(build / 'bundle' / name, release / name)
    for name in ['case.json', 'binding.json'] + [r['file'] for r in case['inputs'] + case['outputs']]:
        shutil.copy2(host.relative_file(build / 'case', name), release / name)

    sources = [TRANSFORMER_ROOT / op / (op + '.py') for op in layout.STAGES if op != 'GEMMFusedKernel']
    sources += [TRANSFORMER_ROOT / 'dec_enc_fused/brightness.py', GEMM_KERNEL]
    manifest = {'schema_version': 2, 'kind': 'pypto_1952_isp_chain', 'stages': list(stages),
                'start': stages[0], 'end': stages[-1], 'image_shape': list(layout.IMAGE_SHAPE),
                'seed': args.seed, 'target_soc': args.target_soc, 'brightness_rule': layout.BRIGHTNESS_RULE,
                'compiler_host': {'machine': platform.machine(), 'python': platform.python_version(),
                                  'toolkit_root': str(args.toolkit_root.resolve())},
                'gym_source_sha256': {str(path): host.sha256(args.gym_root / path) for path in sources},
                'deployment_source_sha256': {str(path.relative_to(HERE)): host.sha256(path)
                                             for path in sorted(ROOT.rglob('*.py'))},
                'files': {path.relative_to(release).as_posix(): host.sha256(path)
                          for path in sorted(release.iterdir()) if path.is_file()}}
    (release / 'chain.json').write_text(json.dumps(manifest, indent=2) + '\n')
    board_chain.verify_release(release, layout)

    archive = work / 'isp_chain.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(release, arcname='isp_chain')
    checksum = work / 'isp_chain.SHA256SUMS'
    checksum.write_text(host.sha256(archive) + '  ' + archive.name + '\n')
    for artifact in (release, archive, checksum):
        artifact.replace(output / artifact.name)
    print('OM: ' + str(output / 'isp_chain/model.om'))
    print('Board package: ' + str(output / archive.name))
    print('Checksum: ' + str(output / checksum.name))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gym-root', type=Path, required=True)
    parser.add_argument('--toolkit-root', type=Path, required=True)
    parser.add_argument('--pto-root', type=Path, required=True)
    parser.add_argument('--target-soc', default='MC62CM12AA')
    parser.add_argument('--seed', type=int, default=20261123)
    parser.add_argument('--start', help='First operator to compile, inclusive (default: preprocess)')
    parser.add_argument('--end', help='Last operator to compile, inclusive (default: postprocess)')
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    args = parser.parse_args(argv)
    try:
        args.gym_root = resolve_gym_root(args.gym_root)
        stages = layout.selected_stages(args.start, args.end)
        output = args.output.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=False)
        with tempfile.TemporaryDirectory(prefix='.compiler-', dir=output) as temporary:
            work = Path(temporary)
            build = _compile(args, output, work, stages)
            _package(args, output, work, build, stages)
        return 0
    except subprocess.CalledProcessError:
        print('ISP chain build failed. See ' + str(output / 'build.log'), file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print('ISP chain build failed: ' + str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
