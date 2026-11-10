# 5102 双方适配合并与验证（2026-11-10）

本轮合并以 PTO-ISA `6ad4b5cf4`、PyPTO `822cded8183`、gym `eeb36209` 为基线，结合 Josh 的 PTO-ISA `256a599b`、PyPTO `0da8d1be` 及其 SDK 选择改动。框架仓保留指令、IR、代码生成、ABI 和框架回归；实际算子及正确性资产在 gym，双机运行与 profiling 在本仓库。

## 五部分提交

| 部分 | 提交 |
|---|---|
| PTO-ISA 指令与 CPU/PV | `03056184236a5d389e9c362e147b80340930f774` |
| PyPTO 编译、CPU、静态设备导出 | `e1f83a3901ffadc7786ec86a40fcee94d23b80e2` |
| 通用 ONNX/GE/TBE/ATC | `c5d128aaa34674fec53dac33bb03756dc380cb31` |
| gym 算子与正确性 | `d97a8f1525362e4b69b9eaec8ab0dcb1b0c5097a` |
| 双机部署、运行与 profiling | 本文所在提交 |

每部分相对既定基线只有一个提交。原发布版本保存在前三仓本地 `backup/1952-before-peer-merge-20261110` 分支。

## 合并后的行为

- 按指令组织 DAV5102 头文件，抽取公共寄存器封装；具体 dtype 映射不继承 A5 的全部类型。
- 使用对方的 5102 资源值：UB 248 KiB、L1 1 MiB、L0A/B 各 64 KiB、L0C 256 KiB、Bias 4 KiB、FBuffer 6 KiB、ScaleLeft/Right 各 4 KiB。其他架构原值保留。
- 通过公共加载框架支持单矩阵和多矩阵 ND→NZ，保留 Vec 精确字节、短行及多维搬运，提供大 stride 软件回退。
- TMOV 使用展开参数；MAD 默认 shift=42，透传 shift/phase 和 valid M/K/N，兼容整数与 phase-first 模板，支持原地累加和 INT32 BT Bias。CPU 通过原生 PV 指令对照数值。
- raw INT32 写回和 DEQF16 的已验证 scale `2^-16` 保留；partial Acc 写回按真实 MAD 分形 pitch 处理。
- 指令两端的 `PIPE_ALL` 已移除，同步复用 A5；optimized DecEnc 使用定向 flag。内核出口保留 A5 风格的完成同步。
- 移除 SDK/PV 哈希白名单，保留实际文件指纹和独立模型目录。模型目录仍需提供其运行所需库及配置。
- 保留已有 CPU views、scalar、VF、flag、packed mask 能力；合入对方普通 JIT 配置范围，静态导出仍为独立入口。

## 验证结果

环境为 Ubuntu 24.04 AArch64、Python 3.12.3 venv、CANN 9.2；PV 使用现有完整的 9.1 beta1 模型目录。没有引入 Python 3.9 兼容改动。

| 检查 | 通过数量 |
|---|---:|
| PTO 原生编译、CPU 和 PV 回归 | 104 |
| CPU target GTest（含其他架构接口合同） | 19 |
| PyPTO IR、代码生成、CPU/PV、真实设备导出 | 181 |
| ONNX/GE/TBE/ATC 框架 | 63 |
| gym 算子、同步、正确性和设备导出 | 48 |
| 双机编排、provider、runner、profiling 解析 | 95 |
| 合计 | 510 |

以上用例无 skip。另确认 PyPTO C++ 扩展构建和 A5 共享寄存器头的原生编译通过。新 DMA 回归包含 FP16/FP32 多维、短列、超过 40 bit 的 row/loop stride，以及 multi-ND 尾矩阵的真实 PV 数值与 GM guard 检查。Cube 回归包含 shift 41/42/43、phase、Bias、累加、非整块 valid、partial 写回和非 16 行起点。

五组运行包由最终源码重新经过 Bisheng 和 ATC 构建，设备 ELF、实际任务 ABI、主机可移植性及 PTO 依赖指纹检查通过。

| 用例 | OM SHA256 |
|---|---|
| dec-enc-fp16-4k | `967662f8f591af56b863421bdad084439a79058124d1ec83edd193913f3a47f6` |
| dec-enc-fp32 | `17ef5448c12b3e757eb97cdec4b72e5543c176d3bac7f279c6e00033d9f2931d` |
| dec-enc-optimized | `5420e07f2001b5b86405700c5ee5a5274e4a271ce73b41f1a8fa51509dceedc6` |
| gemm-final | `8a752b53726111b9a0e8adbc2b5d618c85cc1c3f881284b93a65884a41b7467b` |
| swiglu | `b61c3a29f9c524d66e90883e34beba7dce04e4bff588173977b54bf14cb3324c` |

机器可读结果见 [merged_5102_validation.json](merged_5102_validation.json)。本次原始日志位于验证工作区 `adaptation/evidence/merged-5102-20261110`，完整运行包位于 `adaptation/merged-builds-final-20261110`。

## 使用边界与上板步骤

静态导出仍要求固定 shape、tensor-only 参数、单个 Cube 或 Vector section、零额外 workspace/tiling；普通 NPU JIT 启动范围没有自行扩大。现有算子按显式 Tile 地址验证，未宣称 Vec AUTO 路径完整可用。TEXTRACT 的列窗口按 16 元素分形边界，任意行起点已覆盖数值回归。

所有新包仍为 `device_execution=not_run`。本轮没有连接执行机，没有新增实板精度或 profiling 结果。CPU/PV 数值、设备编译和 OM 构建结果分别记录，不替代实板验收。

2026-11-11 确认的验证分工：当前开发环境没有 5102 设备，也无法通过 SSH 访问内网。这里的验收范围为本地 CPU/PV 仿真、设备代码编译和 OM 构建；实板精度、profiling 采集及解析由用户在内网完成。实板结果待用户执行后补充，不要求为本地开发提供设备或远端 SSH。

用户在内网编译机按 [README](../README.md) 配好执行机 SSH、runtime 环境及 msprof 路径后，用新输出目录构建，再执行 `run` 或 `profile`。不要复用合并前 OM。profiling 命令会回收输出验精度，并在编译机解析原始性能数据。
