"""One compiler-host command for DecEnc build, board profiling and pipeline statistics."""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from types import SimpleNamespace
import uuid

from .host import remote_target
from .provider import load_provider

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_CONFIG = HERE / "dec_enc_profile.local.json"
DEFAULTS = {
    "schema_version": 1,
    "host": "board5102",
    "runtime_env": None,
    "toolkit_root": None,
    "pto_root": None,
    "build_dir": None,
    "pypto_root": None,
    "gym_root": None,
    "msprof_py": None,
    "output_root": None,
    "remote_root": "/tmp/pypto-dec-enc-profile",
    "python": "python3",
    "msprof": "/var/msprof",
    "device": 0,
    "warmup": 5,
    "repeat": 50,
    "timeout": 1800,
}
PIPELINES = (
    ("cube", "cube"),
    ("vec", "vector"),
    ("mte1", "mte1"),
    ("mte2", "mte2"),
    ("mte3", "mte3"),
    ("fixpipe", "fixpipe"),
    ("scalar", "scalar"),
)


def parse_shape(value):
    parts = re.split(r"[xX,×]", value)
    if len(parts) not in (2, 4) or any(not part.isdecimal() for part in parts):
        raise ValueError("shape 用 HxW 或 1x4xHxW，例如 2160x3840")
    dimensions = [int(part) for part in parts]
    if len(dimensions) == 4 and dimensions[:2] != [1, 4]:
        raise ValueError("DecEnc 输入的 N/C 必须为 1/4")
    return dimensions[-2:]


def _local_path(value, base):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def load_config(path, *, defaults=None, require_build=True, output_folder="dec-enc-profile"):
    defaults = DEFAULTS if defaults is None else defaults
    path = Path(path).expanduser().resolve(strict=True)
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or set(raw) - defaults.keys():
        raise ValueError("配置必须为 JSON 对象，且只能包含对应示例配置中的字段")
    config = {**defaults, **raw}
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("不支持的配置 schema_version")
    for key in ("device", "warmup", "repeat", "timeout"):
        if type(config[key]) is not int or config[key] < (1 if key in ("repeat", "timeout") else 0):
            raise ValueError("配置数值无效: " + key)
    for key in ("host", "remote_root", "python", "msprof"):
        if not isinstance(config[key], str) or not config[key] or "\n" in config[key] or "\0" in config[key]:
            raise ValueError("配置字符串无效: " + key)
    if not config["msprof"].startswith("/"):
        raise ValueError("msprof 必须是板端可执行文件的绝对路径")
    runtime = config["runtime_env"]
    if runtime is not None and (not isinstance(runtime, str) or not runtime.startswith("/") or "\n" in runtime):
        raise ValueError("runtime_env 填板端绝对路径，或显式 null 使用板端现有环境")
    for key in ("toolkit_root", "pto_root", "build_dir", "pypto_root", "gym_root", "msprof_py", "output_root"):
        if config[key] is not None:
            if not isinstance(config[key], str) or not config[key]:
                raise ValueError("配置路径无效: " + key)
            config[key] = _local_path(config[key], path.parent)
    selected = config["toolkit_root"] or next(
        (
            os.environ[key]
            for key in ("PYPTO_PRO_TOOLKIT_ROOT", "ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME")
            if os.environ.get(key)
        ),
        None,
    )
    if not selected:
        raise ValueError("在配置中填写 toolkit_root，或先设置 PYPTO_PRO_TOOLKIT_ROOT")
    config["toolkit_root"] = Path(selected).expanduser().resolve(strict=True)
    toolkit = config["toolkit_root"]
    if not (toolkit / "set_env.sh").is_file():
        raise ValueError("toolkit_root 中缺少 set_env.sh")
    config["pto_root"] = Path(config["pto_root"] or os.environ.get("PTO_LIB_PATH") or REPO.parent / "pto-isa").resolve()
    if not (config["pto_root"] / "include/pto/pto-inst.hpp").is_file():
        raise ValueError("找不到 PTO 头文件，请填写 pto_root")
    config["pypto_root"] = (
        config["pypto_root"] or Path(os.environ.get("PYPTO_SOURCE_ROOT", REPO.parent / "pypto")).resolve()
    )
    config["gym_root"] = (
        config["gym_root"] or Path(os.environ.get("PYPTO_GYM_ROOT", REPO.parent / "pypto-gym")).resolve()
    )
    if require_build:
        config["build_dir"] = config["build_dir"] or config["pypto_root"] / "build"
    if config["build_dir"] is not None:
        cache = config["build_dir"] / "CMakeCache.txt"
        if (
            not cache.is_file()
            or "CMAKE_HOME_DIRECTORY:INTERNAL=" + str(config["pypto_root"]) not in cache.read_text().splitlines()
        ):
            raise ValueError("build_dir 必须是 pypto_root 对应的 CMake 目录；首次安装见 README.md")
    if not config["msprof_py"]:
        candidates = [
            toolkit / suffix
            for suffix in (
                "tools/profiler/profiler_tool/analysis/msprof/msprof.py",
                "tools/profiler/analysis/msprof/msprof.py",
            )
        ]
        config["msprof_py"] = next((candidate for candidate in candidates if candidate.is_file()), None)
    if config["msprof_py"] is None or not config["msprof_py"].is_file():
        raise ValueError("找不到编译机 msprof.py，请在配置中填写 msprof_py")
    config["output_root"] = config["output_root"] or REPO / "output" / output_folder
    remote_target(
        SimpleNamespace(
            host=config["host"],
            remote_dir=config["remote_root"].rstrip("/") + "/preflight",
            device=config["device"],
            timeout=config["timeout"],
        )
    )
    return config


def _compiler_command(config, arguments):
    # Source the selected SDK in the same child that imports Pro/ATC, keeping the active venv Python.
    script = 'source "$1" >/dev/null && shift && export TORCH_DEVICE_BACKEND_AUTOLOAD=0 && '
    script += 'export PYTHONPATH="$1${PYTHONPATH:+:$PYTHONPATH}" && shift && '
    script += 'export PYPTO_SOURCE_ROOT="$1" && shift && export PATH="${1%/*}:$PATH" && exec "$@"'
    return [
        "bash",
        "-c",
        script,
        "pypto-compiler",
        str(config["toolkit_root"] / "set_env.sh"),
        str(config["pypto_root"] / "python") + os.pathsep + str(config["pypto_root"]) + os.pathsep + str(REPO),
        str(config["pypto_root"]),
        sys.executable,
        "-m",
        "pypto_1952.run_profile",
        *arguments,
    ]


def _run(command, log, *, timeout=None):
    with log.open("w") as stream:
        result = subprocess.run(
            command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout, check=False
        )
    if result.returncode:
        raise RuntimeError("阶段返回码 {}；日志: {}".format(result.returncode, log))


def _board_preflight(config, log):
    script = 'set -e; if [ -n "$1" ]; then source "$1" || exit $?; fi; set -e; test -x "$2"; '
    script += '"$3" -c "import sys; assert sys.version_info >= (3, 7, 9), sys.version; print(sys.version)"'
    command = ["bash", "-c", script, "pypto-preflight", config["runtime_env"] or "", config["msprof"], config["python"]]
    _run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", config["host"], shlex.join(command)],
        log,
        timeout=config["timeout"],
    )


def format_summary(report, dtype, shape, *, title=None):
    if (
        report.get("status") != "passed"
        or not report.get("performance")
        or report.get("accuracy", {}).get("status") != "passed"
        or report.get("accuracy", {}).get("profile_final_comparison", {}).get("passed") is not True
    ):
        raise ValueError("精度或 profiling 未通过，不打印有效性能结果")
    performance = report["performance"]
    if performance.get("device_units") != "us" or not performance.get("expected_nodes"):
        raise ValueError("缺少设备节点或明确的微秒单位")
    lines = [
        "",
        "{}  精度: PASSED  测量: {} 次（已去掉精度执行与预热）".format(
            title or "DecEnc {} {}".format(dtype, "x".join(map(str, shape))), performance["repeat"]
        ),
    ]
    for name in performance["expected_nodes"]:
        node = performance["nodes"][name]
        lines += ["", "节点: " + name, "流水/任务       median(us)       mean(us)        min(us)        max(us)"]
        metrics = [("Task Duration", node["statistics_us"])]
        metrics += [(label, node.get("pipeline_us", {}).get(key, {})) for label, key in PIPELINES]
        for label, metric in metrics:
            statistics = (
                metric
                if label == "Task Duration"
                else (metric.get("statistics") if metric.get("status") == "available" else None)
            )
            if statistics is None:
                lines.append("{:<14} 未提供 (N/A)".format(label))
            else:
                values = [float(statistics[key]) for key in ("median", "mean", "min", "max")]
                if any(not math.isfinite(value) or value < 0 for value in values):
                    raise ValueError("无效的流水耗时: " + label)
                lines.append("{:<14}".format(label) + "".join("{:15.3f}".format(value) for value in values))
    lines += [
        "",
        "单位均为 μs；各流水可重叠，不能相加作为 Task Duration。",
        "未提供表示采集/导出没有完整的对应耗时列，不能当作 0。",
    ]
    return "\n".join(lines) + "\n"


def execute(config, dtype, height, width):
    _, provider = load_provider(gym_root=config.get("gym_root"))
    case = provider.dec_enc_configuration(height, width, dtype, 8)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    label = "{}-{}x{}-{}".format(dtype, height, width, run_id)
    root = config["output_root"] / label
    root.mkdir(parents=True, exist_ok=False)
    remote = config["remote_root"].rstrip("/") + "/" + label
    state = {
        "schema_version": 1,
        "status": "started",
        "stage": "preflight",
        "configuration": case,
        "schedule": "pipelined",
        "output": str(root),
        "remote_dir": remote,
        "commands": [],
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in config.items()},
    }

    def save():
        (root / "one_click_result.json").write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")

    save()
    try:
        print("结果目录: " + str(root), flush=True)
        print("[1/4] 检查 SSH、板端 Python 和 msprof", flush=True)
        _board_preflight(config, root / "preflight.log")
        state["stage"] = "build"
        build = [
            "build",
            "--gym-root",
            str(config["gym_root"]),
            "--op",
            "dec-enc",
            "--dtype",
            dtype,
            "--height",
            str(height),
            "--width",
            str(width),
            "--block-dim",
            "8",
            "--dec-enc-schedule",
            "pipelined",
            "--build-dir",
            str(config["build_dir"]),
            "--toolkit-root",
            str(config["toolkit_root"]),
            "--pto-root",
            str(config["pto_root"]),
            "--output",
            str(root / "build"),
        ]
        command = _compiler_command(config, build)
        state["commands"].append(command)
        save()
        print("[2/4] 更新 C++ 后端，编译 Pro/PTO → ATC → OM，生成输入/golden", flush=True)
        _run(command, root / "build.log")
        build_result = json.loads((root / "build/build_result.json").read_text())
        if build_result.get("status") != "bundle-built":
            raise ValueError("编译未生成合格运行包")
        state["stage"] = "profile"
        profile = [
            "profile",
            "--build",
            str(root / "build"),
            "--host",
            config["host"],
            "--remote-dir",
            remote,
            "--python",
            config["python"],
            "--device",
            str(config["device"]),
            "--timeout",
            str(config["timeout"]),
            "--msprof",
            config["msprof"],
            "--msprof-py",
            str(config["msprof_py"]),
            "--warmup",
            str(config["warmup"]),
            "--repeat",
            str(config["repeat"]),
            "--output",
            str(root / "profile"),
        ]
        if config["runtime_env"]:
            profile += ["--runtime-env", config["runtime_env"]]
        command = _compiler_command(config, profile)
        state["commands"].append(command)
        save()
        print("[3/4] 上传、板端采集、回收、编译机验精度与 msprof.py 解析", flush=True)
        _run(command, root / "profile.log")
        report_path = root / "profile/result.json"
        report = json.loads(report_path.read_text())
        state["stage"] = "summary"
        summary = format_summary(report, dtype, [height, width])
        (root / "summary.txt").write_text(summary)
        state.update(
            status="passed",
            result=str(report_path),
            om_sha256=report["om_sha256"],
            raw_profiling=report["raw_profiling"],
            statistics_csv=report["performance"]["statistics_csv"]["path"],
            summary=str(root / "summary.txt"),
        )
        save()
        print("[4/4] 完成" + summary, flush=True)
        print("详细结果: " + str(report_path))
        print("流水 CSV: " + state["statistics_csv"])
        print("原始 PROF: " + state["raw_profiling"])
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as error:
        state.update(status="failed", error=str(error))
        save()
        print("失败阶段: {}；{}\n保留结果: {}".format(state["stage"], error, root), file=sys.stderr)
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dtype", choices=("fp16", "fp32"), nargs="?")
    parser.add_argument("shape", nargs="?", help="HxW 或 1x4xHxW，例如 2160x3840")
    parser.add_argument("--config", type=Path, default=os.environ.get("PYPTO_1952_PROFILE_CONFIG", DEFAULT_CONFIG))
    parser.add_argument("--init", action="store_true", help="首次生成本机配置，之后只传 dtype 和 shape")
    args = parser.parse_args(argv)
    args.config = args.config.expanduser().resolve()
    try:
        if args.init:
            args.config.parent.mkdir(parents=True, exist_ok=True)
            with args.config.open("x") as stream:
                json.dump(DEFAULTS, stream, indent=2)
                stream.write("\n")
            print(
                "已生成: {}\n首次填写 host、板端 runtime_env；空的本机路径会按环境/当前仓库查找。".format(args.config)
            )
            return 0
        if not args.dtype or not args.shape:
            parser.error("每次只需 dtype 和 shape，例如: fp16 2160x3840")
        height, width = parse_shape(args.shape)
        if not args.config.is_file():
            raise ValueError("首次请执行 --init，并填写一次连接配置: " + str(args.config))
        config = load_config(args.config)
        _, provider = load_provider(gym_root=config["gym_root"])
        provider.dec_enc_configuration(height, width, args.dtype, 8)
        return execute(config, args.dtype, height, width)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print("启动失败: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
