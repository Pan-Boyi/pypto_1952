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
from .provider import load_provider

HERE = Path(__file__).resolve().parent
VALIDATION = Path('ops/pypto_pro/experimental/ops_transformer/isp_1952_validation')


def _compile(args, output, work, assets, layout, stages):
    config = work / 'span.json'
    config.write_text(json.dumps({'start': stages[0], 'end': stages[-1], 'seed': args.seed}) + '\n')
    _, _, height, width = layout.IMAGE_SHAPE
    build = work / 'build'
    command = [sys.executable, '-B', '-m', 'pypto_1952.run_profile', 'build',
               '--op', 'ISPChain', '--provider', str(assets / 'chain_provider.py'), '--case-config', str(config),
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


def _package(args, output, work, build, assets, layout, stages):
    build, case, _, _ = host.validate_build(build)
    release = work / 'isp_chain'
    release.mkdir()
    for source, name in ((HERE.parent / 'scripts/board/test_isp_chain.sh', 'run.sh'),
                         (HERE / 'board_chain.py', 'board_chain.py'),
                         (HERE / 'host.py', 'acceptance.py'),
                         (assets / 'chain_layout.py', 'chain_layout.py')):
        shutil.copy2(source, release / name)
    for name in ('model.om', 'runner.py', 'execution.json'):
        shutil.copy2(build / 'bundle' / name, release / name)
    for name in ['case.json', 'binding.json'] + [r['file'] for r in case['inputs'] + case['outputs']]:
        shutil.copy2(host.relative_file(build / 'case', name), release / name)

    transformer = assets.parent
    sources = [transformer / op / (op + '.py') for op in layout.STAGES if op != 'GEMMFusedKernel']
    sources += [transformer / 'dec_enc_fused/brightness.py', transformer / 'postprocess/reference_nv12.py']
    sources += sorted(assets.glob('*.py'))
    gym = args.gym_root.resolve()
    gemm = gym / 'ops/pro/experimental/gemm_fused_kernel'
    sources += [gemm / name for name in ('kernel.py', 'packing.py', 'reference.py', 'provider.py')]
    manifest = {'schema_version': 2, 'kind': 'pypto_1952_isp_chain', 'stages': list(stages),
                'start': stages[0], 'end': stages[-1], 'image_shape': list(layout.IMAGE_SHAPE),
                'seed': args.seed, 'target_soc': args.target_soc, 'brightness_rule': layout.BRIGHTNESS_RULE,
                'compiler_host': {'machine': platform.machine(), 'python': platform.python_version(),
                                  'toolkit_root': str(args.toolkit_root.resolve())},
                'gym_source_sha256': {str(path.relative_to(gym)): host.sha256(path) for path in sources},
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
        assets = args.gym_root.resolve(strict=True) / VALIDATION
        _, layout = load_provider(assets / 'chain_layout.py')
        stages = layout.selected_stages(args.start, args.end)
        output = args.output.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=False)
        with tempfile.TemporaryDirectory(prefix='.compiler-', dir=output) as temporary:
            work = Path(temporary)
            build = _compile(args, output, work, assets, layout, stages)
            _package(args, output, work, build, assets, layout, stages)
        return 0
    except subprocess.CalledProcessError:
        print('ISP chain build failed. See ' + str(output / 'build.log'), file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print('ISP chain build failed: ' + str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
