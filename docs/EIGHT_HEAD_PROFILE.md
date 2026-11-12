# 八路 FP16 MLP：在编译机一条命令完成测试

入口为 `scripts/profile_eight_head.sh`。脚本运行在**编译机 A**，通过 SSH/SCP 控制执行机 B，执行机只需兼容的 5102/1952 驱动、固件、AscendCL runtime、Python 3.7.9+ 及 msprof 采集器。

```text
编译机加载 gym 的八路 FP16 provider
  → 编译 4 个原生 Pro kernel → ONNX/ATC → 静态 OM
  → 生成输入与独立 golden
  → 只上传 OM、输入和执行器
  → 板端双哨兵精度执行 → 预热 → msprof 重复采集
  → 取回原始输出与 PROF 数据
  → 编译机验精度 → msprof.py 导出/解析
  → 终端打印 4 个节点、7 条流水及汇总统计
```

## 1. 编译机准备

使用本仓 README 的现有 PyPTO/PTO/CANN 环境，更新源码到包含 5102 适配的版本。编译机不需要 NPU，但必须能通过 SSH/SCP 访问执行机。端口和密钥放编译机 `~/.ssh/config`，例如别名 `board5102`；脚本不会保存密码。

部署仓与算子仓分开更新：

```bash
git clone https://github.com/Pan-Boyi/pypto_1952.git
git clone -b codex/1952-native-om https://gitcode.com/pan-boyi/pypto-gym.git
```

八路 kernel、FP16 参考和 provider 位于 gym 的 `ops/pro/experimental/eight_head_mlp_5102/`。部署仓只提供流程脚本，不携带另一份 kernel/golden。现有 gym checkout 若没有该目录，需要更新上述分支，或在配置中填写包含该 provider 的绝对路径。PyPTO 与 PTO-ISA 继续使用已适配 5102 的版本，准备方法见 README。

激活编译机已有 conda/venv 后运行脚本。也可通过 `PYTHON=/所选环境/bin/python` 指定编译机解释器；脚本不自动创建或激活 venv。`build_dir=null` 使用已安装的框架，只有需要增量重编源码 C++ 扩展时才填写对应 CMake build 目录。

## 2. 配置一次

从部署仓根目录运行：

```bash
bash scripts/profile_eight_head.sh --init
```

编辑生成的 `pypto_1952/eight_head_profile.local.json`。该文件被 gitignore 排除；示例在 `pypto_1952/example_eight_head_profile.json`。

| 字段 | 位置与含义 |
|---|---|
| `host` | 编译机能连接的执行机 SSH 别名或 user@host |
| `runtime_env` | **执行机** runtime 环境脚本绝对路径；已有环境则用 `null` |
| `python` | **执行机** Python 命令或绝对路径，默认 `python3` |
| `msprof` | **执行机**采集器绝对路径，默认 `/var/msprof` |
| `toolkit_root` | **编译机** CANN 根目录，内含 `set_env.sh` |
| `msprof_py` | **编译机** `analysis/msprof/msprof.py`，也可用 SDK 标准布局自动查找 |
| `pypto_root` / `pto_root` / `gym_root` | 编译机对应源码路径，可按同级目录或现有环境变量查找 |
| `provider` | 默认 gym 八路算子 provider，通常保持 `null` |
| `build_dir` | 可选的编译机 PyPTO CMake 目录；不要求每次重编框架 |
| `output_root` | 编译机结果根目录；默认 `output/eight-head-profile` |
| `remote_root` | 执行机任务根目录；默认 `/tmp/pypto-eight-head-profile` |
| `block_dim` | 默认 `8`，也可每次命令行覆盖为 `4` 等 1..8 的值 |
| `warmup` / `repeat` | 默认预热 5 次、测量 50 次，精度的两次执行单独排除 |
| `timeout` | 每次 SSH/SCP 操作超时秒数，默认 1800 |

相对路径配置按 JSON 文件所在目录解析；编译机和执行机的路径独立，不能把编译机 CANN 或 Python 路径直接作为板端路径。脚本在编译子进程中 source 所选 SDK，并保留调用者选择的 Python。

## 3. 每次只执行一个脚本

```bash
bash scripts/profile_eight_head.sh

# 比较 4-block 版本；每次自动创建新的本地/远端目录。
bash scripts/profile_eight_head.sh --block-dim 4

# 使用指定配置文件。
bash scripts/profile_eight_head.sh --config /实际路径/eight-head.json
```

输入、参数、中间张量及输出为 FP16，GEMM 使用 5102 原生定点累加器，输入和参考数据由 gym provider 生成。该脚本不接收 dtype/shape，避免把固定算子的合同误当动态配置。

终端打印四个节点 `fc1`、`activation`、`fc2`、`params` 的：

- Task Duration 的 median / mean / min / max，单位 μs；
- cube、vec、mte1、mte2、mte3、fixpipe、scalar 相同统计，缺失项显示 N/A；
- 四个设备任务时长累计，明确标为“非端到端延迟”；
- 同步 `aclmdlExecute` 的整个 OM 主机调用时间；
- 两轮精度和 profiling 最后一次输出的最大绝对误差。

各流水可能重叠，不能相加作为 Task Duration。设备任务累计也不等于端到端延迟。主机调用计时不含模型加载与初次 H2D，不等于单个 kernel 的设备时间。

## 4. 看结果与失败日志

```text
output/eight-head-profile/fp16-b8-<时间与唯一ID>/
  one_click_result.json      整个流程状态、失败阶段、命令和配置
  preflight.log             SSH/runtime/Python/msprof 检查
  build.log                 编译入口日志
  build/native/             4 个设备对象、manifest、详细编译日志
  build/atc/                ATC 日志、OM 和 ABI/ELF 审计
  build/case/               输入、golden、模型绑定，留在编译机
  profile.log               上传、采集、回收、验精度和解析日志
  profile/board/            原始输出与执行报告
  profile/raw/PROF_*/       原始采集数据
  profile/analysis/         导出日志、op_summary CSV、profiling_summary.json、node_statistics.csv
  profile/result.json       精度与 profiling 最终结论
  summary.txt               与终端一致的汇总
```

只有命令退出码为 0、精度通过且 profiling 采集和解析都通过时，才打印有效性能表。失败返回非零，保留原始数据和日志；编译失败不会使用旧 OM。每次运行都建立新目录，便于比较同一输入/工具链/设备上的不同调度。

本地已验证真实设备编译/OM 构建和合成数据驱动的完整四节点采集/解析链路；本环境无法连接内网 5102，实板精度及性能结果仍需在内网执行后获得。

编译机与执行机不能直连时，继续使用 [离线运行包流程](OFFLINE_TESTING.md)。
