"""Compiler-side profiling: stage, collect on 1952, retrieve, check accuracy, export and parse."""

import json
import math
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import uuid

from . import host

HERE = Path(__file__).resolve().parent


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def expected_nodes(root, binding):
    audit_path = root / "atc/om_audit.json"
    if host.sha256(audit_path) != binding["om_audit_sha256"]:
        raise ValueError("OM audit changed before profiling")
    audit = json.loads(audit_path.read_text())
    if audit.get("om_sha256") != binding["om_sha256"]:
        raise ValueError("Profiling node names belong to a different OM")
    nodes = [node["name"] for node in audit["graph"]["kernels"]]
    if not nodes or len(nodes) != len(set(nodes)) or any(not isinstance(name, str) or not name for name in nodes):
        raise ValueError("Profiling requires distinct audited native node names")
    return nodes


def stage_execution(root, destination):
    """Reuse the audited OM/inputs, but ship the current profiling-capable executor."""
    shutil.copytree(root / "bundle", destination / "bundle")
    bundle = destination / "bundle"
    shutil.copy2(HERE / "runner.py", bundle / "runner.py")
    files = {
        str(path.relative_to(bundle)): host.sha256(path)
        for path in sorted(bundle.rglob("*"))
        if path.is_file() and path != bundle / "bundle.json"
    }
    _write(bundle / "bundle.json", {"schema_version": 2, "files": files})
    return host.sha256(bundle / "runner.py")


def remote_scripts(args, run_id):
    """Generate execution/collection scripts on the compiler; no board-side golden or CSV analysis."""
    application = "\n".join(
        [
            "#!/bin/bash",
            "set +e",
            'ROOT=$(cd -- "$(dirname -- "$0")" && pwd) || exit 1',
            'cd "$ROOT/bundle" || exit 1',
            shlex.quote(args.python)
            + " runner.py --execution execution.json --output ../results --device "
            + str(args.device)
            + " --warmup "
            + str(args.warmup)
            + " --repeat "
            + str(args.repeat),
            "RC=$?",
            'printf "%s\\n" "$RC" > "$ROOT/runner.rc"',
            'exit "$RC"',
            "",
        ]
    )
    environment = "true" if not args.runtime_env else "source " + shlex.quote(args.runtime_env)
    collector = "\n".join(
        [
            "#!/bin/bash",
            "set +e",
            "set -o pipefail",
            'ROOT=$(cd -- "$(dirname -- "$0")" && pwd) || exit 1',
            'mkdir -- "$ROOT/capture" || exit 1',
            "if " + environment + "; then ENV_RC=0; else ENV_RC=$?; fi",
            "set +e",
            "PROFILER_RC=127",
            "MSPROF=" + shlex.quote(args.msprof),
            'if [ "$ENV_RC" -eq 0 ] && [ -x "$MSPROF" ]; then',
            '  cd "$ROOT/capture" || exit 1',
            '  "$MSPROF" bash "$ROOT/app.sh"',
            "  PROFILER_RC=$?",
            "else",
            "  echo 'runtime environment failed or msprof is not executable' >&2",
            "fi",
            shlex.quote(args.python)
            + ' - "$ROOT" "$PROFILER_RC" "$ENV_RC" '
            + shlex.quote(run_id)
            + " <<'PYPTO_CAPTURE'",
            "import hashlib, json, os, re, sys",
            "root, profiler_rc, env_rc, run_id = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]",
            "try:",
            "    with open(os.path.join(root, 'runner.rc')) as stream: runner_rc = int(stream.read().strip())",
            "except (OSError, ValueError): runner_rc = 127",
            "capture = os.path.join(root, 'capture')",
            "names = [name for name in os.listdir(capture) if name.startswith('PROF_')]",
            "names = sorted(name for name in names if os.path.isdir(os.path.join(capture, name)))",
            "valid = len(names) == 1 and re.fullmatch(r'PROF_[A-Za-z0-9_.-]+', names[0])",
            "valid = valid and not os.path.islink(os.path.join(capture, names[0]))",
            "collection_rc = 0 if valid else 2",
            "with open(os.path.join(root, 'bundle', 'runner.py'), 'rb') as stream:",
            "    runner_sha = hashlib.sha256(stream.read()).hexdigest()",
            "result = dict(schema_version=1, run_id=run_id, runner_rc=runner_rc, profiler_rc=profiler_rc)",
            "result.update(environment_rc=env_rc, collection_rc=collection_rc, prof=names[0] if valid else None)",
            "result.update(prof_directories=names, runner_sha256=runner_sha)",
            "with open(os.path.join(capture, 'capture.json'), 'w') as stream: json.dump(result, stream, indent=2)",
            "print('PYPTO_PROFILE_CAPTURE=' + json.dumps(result, sort_keys=True))",
            "sys.exit(1 if runner_rc or profiler_rc or env_rc or collection_rc else 0)",
            "PYPTO_CAPTURE",
            "exit $?",
            "",
        ]
    )
    return application, collector


def validate_capture(raw, state):
    capture = json.loads((raw / "capture.json").read_text())
    if not isinstance(capture, dict) or capture.get("schema_version") != 1 or capture.get("run_id") != state["run_id"]:
        raise ValueError("Profiling capture belongs to a different run")
    for key in ("runner_rc", "profiler_rc", "environment_rc", "collection_rc"):
        if type(capture.get(key)) is not int or capture[key] < 0:
            raise ValueError("Invalid profiling return-code evidence: " + key)
    if capture.get("runner_sha256") != state["runner_sha256"]:
        raise ValueError("Board used a different profiling executor")
    name = capture.get("prof")
    directories = sorted(path.name for path in raw.iterdir() if path.name.startswith("PROF_") and path.is_dir())
    if (
        not isinstance(name, str)
        or not re.fullmatch(r"PROF_[A-Za-z0-9_.-]+", name)
        or directories != [name]
        or capture.get("prof_directories") != [name]
        or (raw / name).is_symlink()
    ):
        raise ValueError("Profiling requires exactly one new PROF directory")
    return capture, raw / name


def validate_profile_execution(root, board, args):
    report = json.loads((board / "execution.json").read_text())
    execution = report.get("profiling_execution", {})
    total = 2 + args.warmup + args.repeat
    expected = {
        "enabled": True,
        "status": "completed",
        "warmup": args.warmup,
        "repeat": args.repeat,
        "completed_warmup": args.warmup,
        "completed_repeat": args.repeat,
        "correctness_calls": 2,
        "total_execution_calls": total,
        "attempted_execution_calls": total,
        "discard_first_executions": 2 + args.warmup,
        "timing_scope": "synchronous_aclmdlExecute_host_wall",
    }
    if any(execution.get(key) != value for key, value in expected.items()):
        raise ValueError("Board profiling execution counts or timing scope differ from this request")
    timings = execution.get("host_wall_seconds")
    if (
        not isinstance(timings, list)
        or len(timings) != args.repeat
        or any(type(value) not in (float, int) or not math.isfinite(value) or value < 0 for value in timings)
    ):
        raise ValueError("Board profiling host-wall samples are incomplete or invalid")
    final = execution.get("final_output", {})
    path = board / "profile_actual.bin"
    case = json.loads((root / "case/case.json").read_text())
    expected_output = case["outputs"][0]
    if (
        final.get("file") != path.name
        or final.get("sha256") != host.sha256(path)
        or final.get("bytes") != expected_output["bytes"]
        or path.stat().st_size != expected_output["bytes"]
    ):
        raise ValueError("Final profiling output is missing or its copy/hash differs")
    comparison = host.compare_files(
        path,
        host.relative_file(root / "case", expected_output["file"]),
        case["atol"],
        case["rtol"],
        expected_output["dtype"],
    )
    return execution, comparison


def remote_profile(args, output):
    root, case, binding, _ = host.validate_build(args.build)
    remote = host.remote_target(args)
    output = Path(output)
    if args.warmup < 0 or args.repeat <= 0:
        raise ValueError("Profiling requires warmup>=0 and repeat>0")
    if not args.msprof.startswith("/"):
        raise ValueError("Use the board profiler's explicit absolute path, e.g. /var/msprof")
    parser_path = Path(args.msprof_py).expanduser().resolve(strict=True)
    nodes = expected_nodes(root, binding)
    state = {
        "schema_version": 1,
        "status": "started",
        "run_id": uuid.uuid4().hex,
        "host": args.host,
        "remote_dir": str(remote),
        "device": args.device,
        "om_sha256": binding["om_sha256"],
        "request_sha256": host.sha256(root / "bundle/execution.json"),
        "warmup": args.warmup,
        "repeat": args.repeat,
        "prefix_executions": 2,
        "expected_nodes": nodes,
        "msprof": args.msprof,
        "msprof_py": str(parser_path),
        "msprof_py_sha256": host.sha256(parser_path),
        "validation": {},
        "commands": [],
    }
    state_path = output / "profile_state.json"
    _write(state_path, state)
    staging = output / "staging"
    staging.mkdir()
    state["runner_sha256"] = stage_execution(root, staging)
    application, collector = remote_scripts(args, state["run_id"])
    (staging / "app.sh").write_text(application)
    (staging / "profile.sh").write_text(collector)
    _write(state_path, state)

    def invoke(argv, log):
        state["commands"].append(argv)
        _write(state_path, state)
        try:
            result = subprocess.run(argv, capture_output=True, text=True, timeout=args.timeout, check=False)
            (output / log).write_text(result.stdout + result.stderr)
            return result.returncode
        except (OSError, subprocess.SubprocessError) as error:
            partial = []
            for name in ("stdout", "stderr"):
                value = getattr(error, name, None)
                if isinstance(value, bytes):
                    value = value.decode(errors="replace")
                if value:
                    partial.append(value)
            (output / log).write_text("".join(partial) + "\n" + str(error) + "\n")
            return 124 if isinstance(error, subprocess.TimeoutExpired) else 127

    setup = "mkdir -p -- " + shlex.quote(str(remote.parent)) + " && mkdir -- " + shlex.quote(str(remote))
    state["prepare_rc"] = invoke(["ssh", args.host, setup], "ssh_prepare.log")
    if state["prepare_rc"]:
        state.update(status="failed", errors=["Remote task directory preparation failed"])
        _write(state_path, state)
        raise RuntimeError("Cannot create a new remote profiling task directory; see ssh_prepare.log")
    state["upload_rc"] = invoke(
        ["scp", "-r", str(staging), args.host + ":" + str(remote / "payload")], "scp_upload.log"
    )
    if state["upload_rc"]:
        state.update(status="failed", errors=["Profiling bundle upload failed"])
        _write(state_path, state)
        raise RuntimeError("Profiling execution bundle upload failed; see scp_upload.log")
    state["ssh_rc"] = invoke(
        ["ssh", args.host, "bash " + shlex.quote(str(remote / "payload/profile.sh"))], "board_profiling.log"
    )
    # Always retrieve both, even if msprof or its child failed. The raw data is diagnostic evidence.
    state["results_copy_rc"] = invoke(
        ["scp", "-r", args.host + ":" + str(remote / "payload/results"), str(output / "board")], "scp_results.log"
    )
    state["capture_copy_rc"] = invoke(
        ["scp", "-r", args.host + ":" + str(remote / "payload/capture"), str(output / "raw")], "scp_profiling.log"
    )
    errors, accuracy, performance = [], None, None
    try:
        if state["capture_copy_rc"]:
            raise RuntimeError("Raw profiling data retrieval failed")
        state["capture"] = json.loads((output / "raw/capture.json").read_text())
        capture, prof = validate_capture(output / "raw", state)
        for key in ("runner_rc", "profiler_rc", "environment_rc", "collection_rc"):
            if capture[key]:
                errors.append(key + "=" + str(capture[key]))
        state["local_prof"] = str(prof)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        errors.append(str(error))
    try:
        if state["results_copy_rc"]:
            raise RuntimeError("Board output retrieval failed")
        accuracy = host.evaluate(root, output / "board")
        execution, final_comparison = validate_profile_execution(root, output / "board", args)
        accuracy["profile_final_comparison"] = final_comparison
        if accuracy["status"] != "passed" or not final_comparison["passed"]:
            accuracy["status"] = "failed"
            errors.append("Numerical comparison failed; raw PROF retained, performance summary withheld")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        errors.append(str(error))
    if state["ssh_rc"]:
        errors.append("ssh_rc=" + str(state["ssh_rc"]))
    if not errors:
        try:
            from .profiling_export import export_and_parse

            performance = export_and_parse(
                prof,
                parser_path,
                output / "analysis",
                nodes,
                args.warmup,
                args.repeat,
                prefix_executions=2,
                host_wall_seconds=execution["host_wall_seconds"],
            )
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as error:
            errors.append(str(error))
    state.update(
        status="failed" if errors else "passed",
        errors=errors,
        validation={
            "accuracy": "passed"
            if accuracy
            and accuracy.get("status") == "passed"
            and accuracy.get("profile_final_comparison", {}).get("passed")
            else "failed",
            "profiling_export": "passed" if performance is not None else "not_completed",
        },
    )
    _write(state_path, state)
    return {
        "schema_version": 1,
        "status": state["status"],
        "operator": case["operator"],
        "precision_checked_on": "compiler_host",
        "om_sha256": binding["om_sha256"],
        "run_id": state["run_id"],
        "accuracy": accuracy,
        "performance": performance,
        "profile_state": str(state_path),
        "raw_profiling": state.get("local_prof"),
        "errors": errors,
    }
