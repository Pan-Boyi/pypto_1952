"""Compose one native static OM for an inclusive range of ISP operators."""

import json
from pathlib import Path

from pypto_1952.isp_1952_validation import ROOT, chain_layout as LAYOUT, provider as assets
from pypto_1952.isp_1952_validation.gemm_fused_kernel import provider as gemm


def _stage_spec(operator, config):
    if operator == 'GEMMFusedKernel':
        return gemm.build_spec(config)
    return assets.build_spec(dict(config, op=operator))


def build_spec(config):
    from pypto_1952.isp_1952_validation import chain_bridges as bridge

    stages = LAYOUT.selected_stages(config.get('start'), config.get('end'))
    _, _, height, width = LAYOUT.IMAGE_SHAPE
    if (config.get('height', height), config.get('width', width)) != (height, width):
        raise ValueError('The static ISP graph uses ' + str((height, width)))
    config = dict(config, height=height, width=width)

    nodes, kernels, options = [], {}, {}
    external, produced, shared_boundary = {}, {}, {}

    def add_bridge(name, kernel, source, outputs):
        kernels[name] = kernel
        options[name] = {'block_dim': 1}
        nodes.append({'name': name, 'kernel': name, 'inputs': [source], 'outputs': outputs})

    for operator in stages:
        stage = _stage_spec(operator, config)
        spec = stage['graph_spec']
        prefix = operator + '_'
        links = {}

        if operator == 'GEMMFusedKernel' and ('input_fused', 'features') in produced:
            add_bridge('split_features', bridge.split_features, produced['input_fused', 'features'],
                       ['global_features', 'local_features'])
            links.update(('x' + str(i), 'global_features' if i < 4 else 'local_features') for i in range(8))
        if operator == 'dec_enc_fused' and ('GEMMFusedKernel', 'y') in produced:
            coefficients = {port: prefix + port for port in LAYOUT.DEC_COEFF_SHAPES}
            add_bridge('unpack_coefficients', bridge.unpack_coefficients, produced['GEMMFusedKernel', 'y'],
                       list(coefficients.values()))
            links.update(coefficients)

        for port in spec['inputs']:
            if port in links:
                continue
            source = LAYOUT.DIRECT_INPUTS.get(operator, {}).get(port)
            if source in produced:
                links[port] = produced[source]
                continue
            # An upstream image outside the span feeds both InputFused and DecEnc.
            name = prefix + port if source is None else shared_boundary.setdefault(source, prefix + port)
            links[port] = name
            external.setdefault(name, (operator, port))

        for name, kernel in stage['kernels'].items():
            kernels[prefix + name] = kernel
            options[prefix + name] = stage.get('kernel_options', {}).get(name, {})
        for node in spec['nodes']:
            outputs = node['outputs'] if 'outputs' in node else [node['output']]
            nodes.append({'name': prefix + node['name'], 'kernel': prefix + node['kernel'],
                          'inputs': [links.get(port, prefix + port) for port in node['inputs']],
                          'outputs': [prefix + port for port in outputs]})
        model_outputs = {prefix + port: port for port in spec['outputs']}
        produced.update(((operator, port), name) for name, port in model_outputs.items())

    graph = {'name': 'ISPChain', 'inputs': list(external), 'outputs': list(model_outputs), 'nodes': nodes}

    def make_case(adapter_path, output):
        from pypto_1952.isp_1952_validation import chain_golden as golden

        adapter = json.loads(Path(adapter_path).read_text())
        if adapter['graph_spec'] != graph:
            raise ValueError('Compiled graph differs from the selected ISP span')
        path = golden.make_span_case(output, config.get('seed', 20261123), stages, external, model_outputs)
        case = json.loads(path.read_text())
        case.update(native_kernels=assets.identities(adapter),
                    golden_source_sha256=assets.sha256(ROOT / 'chain_golden.py'))
        path.write_text(json.dumps(case, indent=2) + '\n')
        return path

    return {'kernels': kernels, 'graph_spec': graph, 'block_dim': config.get('block_dim', 8),
            'kernel_options': options, 'pass_options': {'enable_slice': False},
            'make_case': make_case, 'bind_case': assets.bind_case}
