# 2026-11-10 拆仓与验证

后续双方适配合并及当前验证结果见 [MERGED_5102.md](MERGED_5102.md)。本文下方测试数字记录首次拆仓时的验证。

## 五部分与提交

本轮只整理原 `codex/5102-cube-swiglu` 之后的新增适配，不改该基线之前的历史。
框架指令、IR、代码生成与 ABI 回归保留在框架仓；实际算子与其数值测试移到 gym。

| 部分 | 仓库及分支 | 本轮提交 |
|---|---|---|
| PTO-ISA 5102 指令与 CPU/PV 支持 | `pan-boyi/pto-isa`，`codex/1952-native-om` | `03056184236a5d389e9c362e147b80340930f774` |
| PyPTO 5102 编译、模拟与静态设备导出 | `pan-boyi/pypto`，`codex/1952-native-om` | `e1f83a3901ffadc7786ec86a40fcee94d23b80e2` |
| 通用 ONNX/GE/TBE/ATC 接入与 OM 审计 | 同上 | `c5d128aaa34674fec53dac33bb03756dc380cb31` |
| Pro 算子、合同、输入/golden 与正确性资产 | `pan-boyi/pypto-gym`，`codex/1952-native-om` | `d97a8f1525362e4b69b9eaec8ab0dcb1b0c5097a` |
| 双机部署、运行与 profiling | `Pan-Boyi/pypto_1952`，`main` | 本文所在的首次提交 |

对应基线为 PyPTO `822cded8183`、PTO-ISA `6ad4b5cf4`、gym `eeb36209`。本地旧历史保留在各仓库的 `backup/1952-before-repository-split-20261110` 分支。

框架通用接入现在通过可安装的 `pypto_pro_atc` 包调用。它不在 import 时加载 PyPTO 原生扩展，保证部署入口可以先增量构建 C++，再加载 Pro。
本仓库从 gym provider 读取已有 kernel 和正确性回调，不包含另一份算子或参考计算。六份迁移的 Pro 定义与原版本逐一进行 AST 对比，仅归一化合同的相对导入后，当时 6/6 一致；后续合并中 optimized DecEnc 的同步已按新方案调整，见合并记录。

## 验证环境与结果

保留原 Ubuntu 24.04 AArch64 虚拟机和 venv：Python 3.12.3、CANN 9.2、Torch 2.12.0+cpu、NumPy 2.2.6、pytest 8.4.2、ONNX 1.23.2。未引入 conda 或 Python 3.9 兼容修补。

| 检查 | 结果 |
|---|---|
| PyPTO C++ `tile_fwk_interface` 增量构建 | 通过 |
| PyPTO CPU、代码生成、静态设备导出回归 | 141 项通过，无 skip |
| PyPTO 通用 ATC 包回归 | 63 项通过，无 skip |
| PTO-ISA 五组指令编译/CPU 回归 | 75 项通过，无 skip |
| gym 算子与正确性资产全目录 | 41 项通过，无 skip |
| 本仓库部署、runner、profiling、provider 回归 | 95 项通过，无 skip |
| 本仓库 Ruff import/错误检查及格式化 | 通过 |
| 本仓库 venv editable 安装与 CLI | 通过 |

框架集合中一条旧断言仍期待 `bar_all` 被拒绝，但当前模拟器已支持它；将断言改为仍未支持的 `bar_mte2` 后，单独补跑通过，其余 140 项此前通过。
gym 回归包含 Add/GEMM CPU、DecEnc 完整八路与 4K FP16 多核数值，以及真实设备对象导出。
部署回归使用合成传输、OM/ELF 结构和性能数据；新增 provider 测试保留真实编排、打包和哈希检查，只替换编译器/ATC 边界。

## 真实 CANN 编译

以下五组均由新部署入口从 gym 读取算子，通过 PyPTO 静态导出和 `pypto_pro_atc` 生成 OM。
均为 `bundle-built`，已检查原生设备 ELF 嵌入、参数 ABI、主机可移植性，以及运行包输入/OM/绑定的哈希。

| 用例 | OM SHA256 |
|---|---|
| Add | `b98b0fb26a6d3d08dcdcb236ed7b593ff4880ae64b474a869d415cb02e4e2027` |
| GEMM | `4684fc8b509781c0ad8d681f097b6a58d3ea1b31fd608f9422fb3470cbc888d4` |
| SwiGLU 四节点网络 | `a6ec546b85a323e4c9f851c72d4b362ccb674fc03fe72ca1c7754f2bdee89f00` |
| DecEnc 540×960 FP32 | `35afede6901b15d3fc7d745ae8d6b5339d08d17c87a0baddfac877a40140b17f` |
| DecEnc 2160×3840 FP16 | `f0a319c79aefb4f7539ccbe35bf80abf5384d8607f3a9b87f1a5d8a64f01791d` |

所有新包的 `device_execution` 仍为 `not_run`。本次未连接真实 5102 板卡，因此没有新增实板精度或性能结论，也没有将模拟 profiling 数据当作实测值。

## 本地证据位置

验证虚拟机工作目录：`/home/ubuntu/tmp/pypto-pro-5102`。

- 日志：`adaptation/evidence/repository-split-*.log`，包括 framework 首轮与同步断言修复后的单项补跑。
- 真实构建汇总：`adaptation/evidence/repository-split-real-builds.json`；Add 单独记在 `repository-split-real-add.log`。
- 五组 OM、冻结设备对象、ATC 日志、审计与运行包：`adaptation/repository-split-builds/`。
- 拆仓前的工具快照：`adaptation/repository-split-backup-20261110/`。

这些本地产物不进入 Git。复现命令以根目录 README 为准；旧历史记录中的 tools 路径已停止使用。
