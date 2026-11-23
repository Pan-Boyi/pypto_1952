"""Static OM providers for the FP16 ISP operators."""

import hashlib
import json
from pathlib import Path

from pypto_1952.isp_1952_validation import ROOT, TRANSFORMER_ROOT
from pypto_1952.provider import load_provider, resolve_gym_root

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identities(adapter):
    result = {}
    for name, item in adapter['kernels'].items():
        manifest, obj = Path(item['frozen_manifest']), Path(item['frozen_object'])
        if sha256(manifest) != item['native_manifest_sha256'] or sha256(obj) != item['native_object_sha256']:
            raise ValueError('Frozen kernel changed: ' + name)
        result[name] = {'manifest_sha256': sha256(manifest), 'object_sha256': sha256(obj)}
    return result


def write_case(output, inputs, outputs, tolerances, config):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)

    def write_tensors(values):
        records = []
        for name, value in values.items():
            value = value.detach().cpu().contiguous()
            dtype = {'torch.float16': ('fp16', '<f2'), 'torch.float32': ('fp32', '<f4'),
                     'torch.int32': ('int32', '<i4')}[str(value.dtype)]
            file = output / (name + '.bin')
            value.numpy().astype(dtype[1], copy=False).tofile(file)
            records.append({'name': name, 'shape': list(value.shape), 'dtype': dtype[0], 'file': file.name,
                            'bytes': file.stat().st_size, 'sha256': sha256(file)})
        return records

    case = {'schema_version': 1, 'operator': config['operator'], 'inputs': write_tensors(inputs),
            'outputs': write_tensors(outputs),
            'atol': tolerances[0], 'rtol': tolerances[1], 'configuration': config,
            'golden': 'Independent CPU PyTorch reference'}
    path = output / 'case.json'
    path.write_text(json.dumps(case, indent=2) + '\n')
    return path


def make_case(adapter_path, output):
    from pypto_1952.isp_1952_validation import golden

    adapter = json.loads(Path(adapter_path).read_text())
    manifest = json.loads(Path(next(iter(adapter['kernels'].values()))['frozen_manifest']).read_text())
    config = manifest['isp_test_configuration']
    inputs, outputs, tolerances = golden.make_case(**config)
    path = write_case(output, {name: inputs[name] for name in adapter['graph_spec']['inputs']},
                      {name: outputs[name] for name in adapter['graph_spec']['outputs']}, tolerances, config)
    case = json.loads(path.read_text())
    case.update(bias_encoding='round(FP32_bias * 65536), INT32 Q16', native_kernels=identities(adapter),
                golden_source_sha256=sha256(ROOT / 'golden.py'))
    path.write_text(json.dumps(case, indent=2) + '\n')
    return path


def bind_case(case_file, om_file, output, adapter_path):
    from pypto_pro_atc.audit import audit_om_binary
    from pypto_pro_atc.network import validate_network_launch_audit, validate_network_spec

    case_file, om_file, adapter_path = map(Path, (case_file, om_file, adapter_path))
    case, adapter = json.loads(case_file.read_text()), json.loads(adapter_path.read_text())
    audit_path = adapter_path.parent / 'om_audit.json'
    audit = json.loads(audit_path.read_text())
    portability = audit_om_binary(om_file)
    if (adapter.get('kind') != 'native-network' or adapter.get('status') != 'om-compiled'
            or adapter['om']['sha256'] != sha256(om_file) or case['native_kernels'] != identities(adapter)
            or adapter['audit']['sha256'] != sha256(audit_path) or audit['om_sha256'] != sha256(om_file)
            or audit['host_portability'] != portability):
        raise ValueError('Case, frozen kernels, audit and OM identities differ')
    manifests = {name: json.loads(Path(item['frozen_manifest']).read_text())
                 for name, item in adapter['kernels'].items()}
    validate_network_launch_audit(audit, manifests, adapter['graph_spec'], adapter['kernels'], adapter['target_soc'])
    tensors = validate_network_spec(manifests, adapter['graph_spec'])
    for record in case['inputs'] + case['outputs']:
        if tensors[record['name']] != {key: record[key] for key in ('shape', 'dtype')}:
            raise ValueError('Case tensor differs from frozen ABI: ' + record['name'])
    data = om_file.read_bytes()
    if any(Path(item['frozen_object']).read_bytes() not in data for item in adapter['kernels'].values()):
        raise ValueError('OM lacks a complete frozen device ELF')
    bindings = []
    for record, declared in zip(case['outputs'], audit['graph']['outputs']):
        names = list(dict.fromkeys(v for v in (declared.get('logical_name'), declared.get('producer_ref'),
                                               declared.get('model_declared_name'), record['name']) if v))
        bindings.append({'logical_name': record['name'], 'om_declared_names': names})
    binding = {'schema_version': 2, 'case_sha256': sha256(case_file), 'om_sha256': sha256(om_file),
               'native_kernels': case['native_kernels'], 'onnx_sha256': adapter['onnx']['sha256'],
               'target_soc': adapter['target_soc'], 'runtime_dependencies': 'device_kernel_only',
               'host_portability': portability, 'om_audit_sha256': sha256(audit_path),
               'output_name_bindings': bindings}
    Path(output).write_text(json.dumps(binding, indent=2) + '\n')


def build_spec(config):
    operator = config['op']
    directory = resolve_gym_root(config.get('gym_root')) / TRANSFORMER_ROOT / operator
    height, width = config.get('height', 1080), config.get('width', 1920)
    blocks = config.get('block_dim', 8)
    if blocks != 8:
        raise ValueError('The canonical 1952 validation uses eight blocks')
    if operator in ('input_fused', 'ffm_fused') and (height, width) != (1080, 1920):
        raise ValueError(operator + ' is fixed at 1080x1920')
    _, module = load_provider(directory / (operator + '.py'))
    kernel_options = {}
    if operator == 'preprocess':
        if height < 1 or width != 1920:
            raise ValueError('Preprocess requires positive height and width 1920')
        inputs, outputs = ['x', 'lut'], ['out']
        kernels = {operator: module.preprocess_kernel}
        nodes = [{'name': operator, 'kernel': operator, 'inputs': inputs, 'outputs': outputs}]
        kernel_options[operator] = {'static_shapes': {'x': [1, 4, height, width],
                                                     'out': [1, 4, height, width]}}
    elif operator == 'dec_enc_fused':
        inputs = ['x', 'channel_scale', 'global_exponent', 'global_bias', 'channel_mix',
                  'tile_scale', 'tile_exponent', 'tile_bias', 'tile_mix', 'weight', 'bias']
        outputs = ['output']
        _, brightness = load_provider(directory / 'brightness.py')
        kernels = {'gray_partials': brightness.gray_partials_kernel,
                   'image_mean': brightness.image_mean_kernel, operator: module.onchip_kernel}
        nodes = [
            {'name': 'gray_partials', 'kernel': 'gray_partials', 'inputs': ['x'], 'outputs': ['gray_sums']},
            {'name': 'image_mean', 'kernel': 'image_mean', 'inputs': ['x', 'gray_sums'], 'outputs': ['mean']},
            {'name': operator, 'kernel': operator, 'inputs': inputs + ['mean'], 'outputs': outputs},
        ]
        image_shape = [1, 4, height, width]
        kernel_options['gray_partials'] = {'static_shapes': {'x': image_shape}}
        kernel_options['image_mean'] = {'static_shapes': {'x': image_shape}}
        kernel_options[operator] = {'static_shapes': {'x': image_shape, 'output': [1, 16, height, width]}}
    elif operator == 'ffm_fused':
        inputs = ['x', 'w1', 'b1', 'w2', 'b2', 'zero', 'w3', 'b3', 'coeff', 'cb']
        outputs = ['yout', 'uvout']
        kernels = {operator: module.raw_module_kernel(nhwc=False)}
        nodes = [{'name': operator, 'kernel': operator, 'inputs': inputs, 'outputs': outputs}]
        kernel_options[operator] = {'static_shapes': {'x': [1, 16, height, width],
                                                     'yout': [1, 4, height, width],
                                                     'uvout': [1, 2, height, width]}}
    elif operator == 'input_fused':
        inputs = ['x', 'w1', 'b1', 'w2', 'b2', 'w3', 'b3', 'zeros', 'poolweights', 'ones']
        outputs = ['features']
        kernels = {'resize': module.resize_kernel, 'encoder': module.encoder_kernel, 'reduce': module.reduce_kernel}
        nodes = [{'name': 'resize', 'kernel': 'resize', 'inputs': ['x'], 'outputs': ['resized']},
                 {'name': 'encoder', 'kernel': 'encoder', 'inputs': ['resized'] + inputs[1:],
                  'outputs': ['local_features', 'partials']},
                 {'name': 'reduce', 'kernel': 'reduce', 'inputs': ['local_features', 'partials'], 'outputs': outputs}]
        kernel_options['resize'] = {'static_shapes': {'x': [1, 4, height, width]}}
    elif operator == 'postprocess':
        if min(height, width) < 1:
            raise ValueError('Postprocess requires positive height and width')
        inputs, outputs = ['y4', 'uv'], ['out']
        kernels = {operator: module.postprocess_kernel}
        nodes = [{'name': operator, 'kernel': operator, 'inputs': inputs, 'outputs': outputs}]
        kernel_options[operator] = {'static_shapes': {'y4': [1, 4, height, width], 'uv': [1, 2, height, width],
                                                     'out': [1, 6 * height * width]}}
    else:
        raise ValueError('Unknown operator: ' + operator)

    def annotate(manifest):
        manifest['isp_test_configuration'] = {'operator': operator, 'height': height, 'width': width,
                                               'seed': config.get('seed', 20261117)}

    return {'kernels': kernels, 'graph_spec': {'name': operator, 'inputs': inputs, 'outputs': outputs, 'nodes': nodes},
            'block_dim': blocks, 'kernel_options': kernel_options, 'pass_options': {'enable_slice': False},
            'annotate_manifest': annotate,
            'make_case': make_case, 'bind_case': bind_case}
