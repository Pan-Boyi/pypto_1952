"""Run custom provider builds with synthetic compiler and ATC boundaries only."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from pypto_pro_atc import build as atc_build
from pypto_pro_atc import network as atc_network

from pypto_1952 import host, provider, run_profile
from tests.binary_fixtures import device_elf, standard_om

PROVIDER_SOURCE = '''
import hashlib
import json
from pathlib import Path
import struct
from pypto_pro_atc.audit import audit_om_binary
VERSION = 1
EVENTS = []
OPTIONS = None


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_spec(config):
    global OPTIONS
    OPTIONS = config
    EVENTS.append("build_spec")

    def annotate(manifest):
        manifest["custom_option"] = config.get("custom_option")
        EVENTS.append("annotate:" + manifest["entry"])

    def make_case(adapter_path, output):
        EVENTS.append("make_case")
        output = Path(output)
        output.mkdir()
        tensors = []
        for name, values in (("x", (1, 2, 3, 4)), ("y", (5, 6, 7, 8)), ("z", (6, 8, 10, 12))):
            path = output / (name + ".bin")
            path.write_bytes(struct.pack("<eeee", *values))
            tensors.append({"name": name, "file": path.name, "sha256": digest(path),
                            "shape": [2, 2], "dtype": "fp16", "bytes": 8})
        if config.get("bad_golden"):
            tensors[-1]["sha256"] = "0" * 64
        case = {"schema_version": 1, "operator": config["op"], "inputs": tensors[:2],
                "outputs": tensors[2:], "atol": 0, "rtol": 0}
        path = output / ("unexpected.json" if config.get("wrong_case_path") else "case.json")
        path.write_text(json.dumps(case))
        return path

    def bind_case(case_path, om_path, output, adapter_path):
        EVENTS.append("bind_case")
        binding = {"schema_version": 2, "case_sha256": digest(case_path),
                   "om_sha256": digest(om_path), "runtime_dependencies": "device_kernel_only",
                   "host_portability": audit_om_binary(om_path),
                   "output_name_bindings": [{"logical_name": "z", "om_declared_names": ["z"]}]}
        Path(output).write_text(json.dumps(binding))

    def analyze(natives, adapter_path):
        EVENTS.append("analyze")
        if config.get("change_provider"):
            Path(__file__).write_text(Path(__file__).read_text() + "\\n# changed during build\\n")
        return {"entries": {name: json.loads(Path(path).read_text())["entry"] for name, path in natives.items()}}

    kernels = {"first": "kernel_one"}
    graph = None
    if config.get("network") or config.get("missing_graph"):
        kernels["second"] = "kernel_two"
    if config.get("network"):
        graph = {"name": "custom_network", "inputs": ["x", "y"], "outputs": ["z"], "nodes": [
            {"name": "node_one", "kernel": "first", "inputs": ["x", "y"], "output": "mid"},
            {"name": "node_two", "kernel": "second", "inputs": ["mid", "y"], "output": "z"}]}
    if config.get("bad_name"):
        kernels = {"../escaped": "kernel_one"}
    if config.get("empty_graph"):
        graph = {}
    return {"kernels": kernels, "graph_spec": graph, "block_dim": 3, "annotate_manifest": annotate,
            "make_case": make_case, "bind_case": None if config.get("missing_callback") else bind_case,
            "analyze": analyze}
'''


class BuildProviderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.provider_path = self.root / "custom_provider.py"
        self.provider_path.write_text(PROVIDER_SOURCE)
        self.compiles, self.adapters = [], []
        runtime = types.ModuleType("pypto_pro.runtime.device_compile")
        runtime.compile_device_kernel = self.compile_kernel
        modules = {
            "pypto_pro": types.ModuleType("pypto_pro"),
            "pypto_pro.runtime": types.ModuleType("pypto_pro.runtime"),
            "pypto_pro.runtime.device_compile": runtime,
        }
        for context in (
            patch.dict(sys.modules, modules),
            patch.object(run_profile, "DEPENDENCIES", {}),
            patch.object(atc_build, "build_adapter", side_effect=self.single_adapter),
            patch.object(atc_network, "build_network_adapter", side_effect=self.network_adapter),
        ):
            context.start()
            self.addCleanup(context.stop)

    def compile_kernel(self, kernel, output, **options):
        output = Path(output)
        output.mkdir(parents=True)
        source, obj = output / "device.cpp", output / "device.o"
        source.write_text("// synthetic compiler boundary: " + kernel)
        obj.write_bytes(device_elf())
        manifest = {
            "entry": kernel,
            "block_dim": options["block_dim"],
            "source": {"path": source.name, "sha256": host.sha256(source)},
            "object": {"path": obj.name, "sha256": host.sha256(obj)},
        }
        path = output / "device_manifest.json"
        path.write_text(json.dumps(manifest))
        self.compiles.append((kernel, path, options))
        return path

    def write_adapter(self, natives, graph, **options):
        output = Path(options["output"])
        output.mkdir()
        objects = [Path(path).with_name("device.o").read_bytes() for path in natives.values()]
        om = output / "model.om"
        om.write_bytes(standard_om([(0, b"synthetic graph, not an ATC compilation"), (3, b"".join(objects))]))
        embedding = "all_frozen_device_elfs_found" if graph else "exact_frozen_device_elf_found"
        report = {
            "status": "om-compiled",
            "om": {"path": str(om), "sha256": host.sha256(om)},
            "validation": {
                "atc": "passed",
                "launch_abi": "passed",
                "host_portability": "passed",
                "kernel_embedding": embedding,
            },
        }
        path = output / "adapter_manifest.json"
        path.write_text(json.dumps(report))
        self.adapters.append((natives, graph, options))
        return path

    def single_adapter(self, native, **options):
        return self.write_adapter({"first": native}, None, **options)

    def network_adapter(self, natives, graph, **options):
        return self.write_adapter(natives, graph, **options)

    def arguments(self, *, output="build", **config):
        case_config = self.root / (output + "-config.json")
        case_config.write_text(json.dumps({"custom_option": "from JSON", **config}))
        return [
            "build",
            "--op",
            "custom_operator",
            "--provider",
            str(self.provider_path),
            "--gym-root",
            str(self.root / "no-gym-checkout"),
            "--case-config",
            str(case_config),
            "--toolkit-root",
            str(self.root / "sdk"),
            "--pto-root",
            str(self.root / "pto"),
            "--output",
            str(self.root / output),
        ]

    def execute(self, **config):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return run_profile.main(self.arguments(**config))

    def test_custom_single_provider_hooks_and_real_bundle_validation(self):
        with patch.object(host, "validate_build", wraps=host.validate_build) as validate:
            self.assertEqual(self.execute(), 0)
        validate.assert_called_once_with(self.root / "build")
        report = json.loads((self.root / "build/build_result.json").read_text())
        self.assertEqual(report["operator"], "custom_operator")
        self.assertEqual(report["provider_sha256"], host.sha256(self.provider_path))
        _, loaded = provider.load_provider(self.provider_path)
        self.assertEqual(loaded.OPTIONS["custom_option"], "from JSON")
        self.assertEqual(loaded.EVENTS, ["build_spec", "annotate:kernel_one", "make_case", "bind_case", "analyze"])
        self.assertEqual(self.compiles[0][1], self.root / "build/native/device_manifest.json")
        self.assertEqual(json.loads(self.compiles[0][1].read_text())["custom_option"], "from JSON")
        self.assertEqual(self.compiles[0][2]["block_dim"], 3)
        self.assertIsNone(self.adapters[0][1])
        metrics = json.loads((self.root / "build/codegen_metrics.json").read_text())
        self.assertEqual(metrics["entries"], {"first": "kernel_one"})
        self.assertFalse((self.root / "build/bundle/z.bin").exists())
        host.validate_build(self.root / "build")

    def test_custom_network_passes_all_native_manifests_to_adapter_and_analysis(self):
        self.assertEqual(self.execute(network=True), 0)
        self.assertEqual(len(self.compiles), 2)
        natives, graph, options = self.adapters[0]
        self.assertEqual(set(natives), {"first", "second"})
        self.assertEqual(natives["first"], self.root / "build/native/first/device_manifest.json")
        self.assertEqual(natives["second"], self.root / "build/native/second/device_manifest.json")
        self.assertEqual(graph["name"], "custom_network")
        self.assertTrue(options["run_atc"])
        metrics = json.loads((self.root / "build/codegen_metrics.json").read_text())
        self.assertEqual(metrics["entries"], {"first": "kernel_one", "second": "kernel_two"})
        host.validate_build(self.root / "build")

    def test_malformed_provider_contracts_fail_before_compilation(self):
        for index, key in enumerate(("bad_name", "missing_graph", "empty_graph", "missing_callback")):
            with self.subTest(key=key), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(run_profile.main(self.arguments(output="bad" + str(index), **{key: True})), 1)
                self.assertFalse((self.root / ("bad" + str(index)) / "build_result.json").exists())
        self.assertEqual(self.compiles, [])
        self.assertEqual(self.adapters, [])

    def test_invalid_golden_or_case_location_cannot_publish_a_build_result(self):
        for index, key in enumerate(("bad_golden", "wrong_case_path")):
            with self.subTest(key=key), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                output = "invalid" + str(index)
                self.assertEqual(run_profile.main(self.arguments(output=output, **{key: True})), 1)
                self.assertFalse((self.root / output / "build_result.json").exists())

    def test_provider_changed_during_compilation_cannot_publish_a_build_result(self):
        self.assertEqual(self.execute(change_provider=True), 1)
        self.assertFalse((self.root / "build/build_result.json").exists())

    def test_provider_reload_executes_the_source_matching_its_hash(self):
        with patch.object(sys, "pycache_prefix", str(self.root / "bytecode")):
            _, first = provider.load_provider(self.provider_path)
            self.assertEqual(first.VERSION, 1)
            before = self.provider_path.stat()
            self.provider_path.write_text(PROVIDER_SOURCE.replace("VERSION = 1", "VERSION = 2"))
            os.utime(self.provider_path, ns=(before.st_atime_ns, before.st_mtime_ns))
            _, second = provider.load_provider(self.provider_path)
        self.assertEqual(second.VERSION, 2)
        self.assertNotEqual(first.__provider_sha256__, second.__provider_sha256__)
        self.assertEqual(second.__provider_sha256__, hashlib.sha256(self.provider_path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
