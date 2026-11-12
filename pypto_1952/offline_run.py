"""Standalone board entry point. Python 3.7.9+ standard library; no compiler imports."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import tarfile


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root):
    root = Path(root).resolve(strict=True)
    manifest = json.loads((root / "runtime.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "pypto_1952_offline_runtime":
        raise ValueError("Expected an offline runtime package generated on the compiler host")
    files = manifest["files"]
    required = {"run.sh", "offline_run.py", "bundle/runner.py", "bundle/execution.json", "bundle/model.om"}
    if not isinstance(files, dict) or not required <= files.keys():
        raise ValueError("Offline runtime inventory is incomplete")
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Runtime package contains a symlink")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != set(files) | {"runtime.json"}:
        raise ValueError("Runtime package contains missing or unlisted files; keep results outside runtime/")
    for name, digest in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("Runtime inventory paths must be relative")
        if sha256(root / relative) != digest:
            raise ValueError("Runtime package SHA256 mismatch: " + name)
    request = json.loads((root / "bundle/execution.json").read_text())
    if (manifest["om_sha256"] != sha256(root / "bundle/model.om")
            or manifest["request_sha256"] != sha256(root / "bundle/execution.json")
            or request["om"] != {"file": "model.om", "sha256": manifest["om_sha256"]}):
        raise ValueError("Runtime release and execution request do not match")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Verify without loading ACL or touching the NPU")
    parser.add_argument("--output", type=Path, help="New result directory outside the extracted runtime directory")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=0, help="Optional synchronous whole-OM host-wall samples")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    try:
        if sys.version_info < (3, 7, 9):
            raise ValueError("Board Python must be 3.7.9 or later")
        manifest = verify(root)
        if args.verify_only:
            print(json.dumps({"status": "verified", "om_sha256": manifest["om_sha256"],
                              "device_execution": "not_run"}))
            return 0
        if args.output is None:
            parser.error("Execution requires --output; use a fresh directory outside runtime/")
        output = args.output.expanduser().resolve()
        if output == root or root in output.parents:
            raise ValueError("Place results outside runtime/ so package hashes remain valid")
        if min(args.device, args.warmup, args.repeat) < 0 or (args.warmup and not args.repeat):
            raise ValueError("device/warmup/repeat must be nonnegative; warmup requires repeat > 0")
        output.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, "-B", str(root / "bundle/runner.py"),
                   "--execution", str(root / "bundle/execution.json"), "--output", str(output / "results"),
                   "--device", str(args.device), "--warmup", str(args.warmup), "--repeat", str(args.repeat)]
        report = {"schema_version": 1, "kind": "pypto_1952_offline_result", "status": "failed",
                  "om_sha256": manifest["om_sha256"], "request_sha256": manifest["request_sha256"],
                  "runtime_manifest_sha256": sha256(root / "runtime.json"), "device": args.device,
                  "warmup": args.warmup, "repeat": args.repeat, "python_version": platform.python_version(),
                  "machine": platform.machine(), "command": command, "runner_rc": None}
        try:
            with (output / "runner.log").open("w") as log:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           universal_newlines=True, bufsize=1)
                with process.stdout as stream:
                    for line in stream:
                        log.write(line)
                        log.flush()
                        print(line, end="", flush=True)
                report["runner_rc"] = process.wait()
            if report["runner_rc"] == 0:
                report["status"] = "executed"
        except OSError as error:
            report["error"] = str(error)
        (output / "board_run.json").write_text(json.dumps(report, indent=2) + "\n")
        archive = output / "results.tar.gz"
        with tarfile.open(str(archive), "w:gz") as tar:
            for path in sorted(output.rglob("*")):
                if path.is_file() and path != archive:
                    tar.add(str(path), arcname=path.relative_to(output).as_posix(), recursive=False)
        print(json.dumps({"status": report["status"], "precision_checked": False,
                          "results_archive": str(archive), "sha256": sha256(archive)}, indent=2))
        return 0 if report["status"] == "executed" else 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        print("Offline execution failed: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
