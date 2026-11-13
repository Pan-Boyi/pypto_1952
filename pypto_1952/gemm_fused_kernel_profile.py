"""One compiler-side command: FP16 GEMMFusedKernel build, upload, execute, retrieve, verify and profile."""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import uuid

from . import dec_enc_profile as common

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name("gemm_fused_kernel_profile.local.json")
DEFAULTS = {**common.DEFAULTS, "remote_root": "/tmp/pypto-gemm-fused-kernel-profile", "block_dim": 8, "provider": None}
EXPECTED_NODES = ("fc1", "activation", "fc2", "params")


def load_config(path):
    path = Path(path).expanduser().resolve(strict=True)
    config = common.load_config(path, defaults=DEFAULTS, require_build=False, output_folder="gemm-fused-kernel-profile")
    if type(config["block_dim"]) is not int or not 1 <= config["block_dim"] <= 8:
        raise ValueError("block_dim 必须是 1..8 的整数")
    if config["provider"] is None:
        config["provider"] = config["gym_root"] / "ops/pro/experimental/gemm_fused_kernel/provider.py"
    else:
        config["provider"] = common._local_path(config["provider"], path.parent)
    if not config["provider"].is_file():
        raise ValueError("找不到 GEMMFusedKernel provider，请更新 gym checkout 或填写 provider: " + str(config["provider"]))
    return config


def format_summary(report, block_dim):
    performance = report.get("performance") or {}
    if performance.get("expected_nodes") != list(EXPECTED_NODES):
        raise ValueError("profiling 节点与 GEMMFusedKernel 的 fc1/activation/fc2/params 不匹配")
    text = common.format_summary(report, "fp16", [], title="GEMMFusedKernel FP16 block_dim={}".format(block_dim))
    lines = [text, "汇总范围                         median(us)       mean(us)        min(us)        max(us)"]
    total = performance["device_task_sum_us"]
    if total.get("scope") != "sum_native_task_durations_not_end_to_end_latency":
        raise ValueError("设备任务累计耗时缺少明确范围")
    wall = performance["host_wall_seconds"]
    if wall.get("scope") != "whole_OM_API_wall_time" or wall.get("units") != "seconds":
        raise ValueError("整个 OM 主机计时缺少明确范围/秒单位")
    for label, values, factor in (("设备任务累计（非端到端延迟）", total["statistics"], 1),
                                  ("整个 OM 同步主机调用", wall["statistics"], 1e6)):
        numbers = [float(values[key]) * factor for key in ("median", "mean", "min", "max")]
        if any(not math.isfinite(value) or value < 0 for value in numbers):
            raise ValueError("无效的汇总耗时: " + label)
        lines.append(label + "  " + "".join("{:15.3f}".format(value) for value in numbers))
    comparisons = report["accuracy"]["comparisons"]
    error = max(item["max_absolute_error"] for item in comparisons)
    final_error = report["accuracy"]["profile_final_comparison"]["max_absolute_error"]
    lines += ["", "双轮精度最大绝对误差: {}；重复执行最终输出误差: {}".format(error, final_error),
              "整个 OM 同步主机调用不含模型加载及初次 H2D，也不等于单个 kernel 设备时间。"]
    return "\n".join(lines) + "\n"


def execute(config):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    label = "fp16-b{}-{}".format(config["block_dim"], run_id)
    root = config["output_root"] / label
    root.mkdir(parents=True, exist_ok=False)
    remote = config["remote_root"].rstrip("/") + "/" + label
    state = {"schema_version": 1, "status": "started", "stage": "preflight", "output": str(root),
             "remote_dir": remote, "commands": [], "config": {
                 key: str(value) if isinstance(value, Path) else value for key, value in config.items()}}

    def save():
        (root / "one_click_result.json").write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")

    save()
    try:
        print("结果目录: " + str(root), flush=True)
        print("[1/4] 检查板端 SSH、runtime、Python 和 msprof", flush=True)
        common._board_preflight(config, root / "preflight.log")
        state["stage"] = "build"
        build = ["build", "--provider", str(config["provider"]), "--op", "GEMMFusedKernel",
                 "--dtype", "fp16", "--block-dim", str(config["block_dim"]),
                 "--toolkit-root", str(config["toolkit_root"]), "--pto-root", str(config["pto_root"]),
                 "--output", str(root / "build")]
        if config["build_dir"] is not None:
            build += ["--build-dir", str(config["build_dir"])]
        command = common._compiler_command(config, build)
        state["commands"].append(command)
        save()
        print("[2/4] 编译四个 Pro kernel、ATC 静态 OM，生成输入和 golden", flush=True)
        common._run(command, root / "build.log")
        if json.loads((root / "build/build_result.json").read_text()).get("status") != "bundle-built":
            raise ValueError("编译未生成合格运行包")
        state["stage"] = "profile"
        profile = ["profile", "--build", str(root / "build"), "--host", config["host"],
                   "--remote-dir", remote, "--python", config["python"], "--device", str(config["device"]),
                   "--timeout", str(config["timeout"]), "--msprof", config["msprof"],
                   "--msprof-py", str(config["msprof_py"]), "--warmup", str(config["warmup"]),
                   "--repeat", str(config["repeat"]), "--output", str(root / "profile")]
        if config["runtime_env"]:
            profile += ["--runtime-env", config["runtime_env"]]
        command = common._compiler_command(config, profile)
        state["commands"].append(command)
        save()
        print("[3/4] 上传 OM/输入，采集 profiling，取回数据，验精度并在编译机解析", flush=True)
        common._run(command, root / "profile.log")
        state["stage"] = "summary"
        report_path = root / "profile/result.json"
        report = json.loads(report_path.read_text())
        summary = format_summary(report, config["block_dim"])
        (root / "summary.txt").write_text(summary)
        state.update(status="passed", result=str(report_path), om_sha256=report["om_sha256"],
                     raw_profiling=report["raw_profiling"],
                     statistics_csv=report["performance"]["statistics_csv"]["path"], summary=str(root / "summary.txt"))
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
    parser.add_argument(
        "--config", type=Path, default=os.environ.get("PYPTO_GEMM_FUSED_KERNEL_PROFILE_CONFIG", DEFAULT_CONFIG))
    parser.add_argument("--init", action="store_true", help="只生成一次本机配置；不会覆盖已有文件")
    parser.add_argument("--block-dim", type=int, choices=range(1, 9), help="本次覆盖配置中的 block_dim")
    args = parser.parse_args(argv)
    args.config = args.config.expanduser().resolve()
    try:
        if args.init:
            args.config.parent.mkdir(parents=True, exist_ok=True)
            with args.config.open("x") as stream:
                json.dump(DEFAULTS, stream, indent=2)
                stream.write("\n")
            print("已生成: {}\n填写编译机路径及板端 SSH、runtime、msprof；然后直接执行本脚本。".format(args.config))
            return 0
        config = load_config(args.config)
        if args.block_dim is not None:
            config["block_dim"] = args.block_dim
        return execute(config)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print("启动失败: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
