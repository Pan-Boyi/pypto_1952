# PyPTO-Pro 5102 双机测试

这个仓库把已经写好的 Pro 算子编译成静态 OM，上传到 5102 执行，再把输出和性能数据取回编译机。

算子只维护一份，放在 **pypto-gym**；输入生成、golden 和误差标准也在那里。本仓库负责调用、打包、传输、执行和性能统计。框架的设备导出、ONNX/ATC 接入在 **PyPTO**，底层指令支持在 **PTO-ISA**。

双方适配的合并内容、五部分提交和验证记录见 [5102 合并与验证](docs/MERGED_5102.md)。新运行包尚未实板执行。

本地验证范围已于 2026-11-11 确认：当前开发环境没有 5102 设备，也不能通过 SSH 访问内网执行机，因此在本地完成 CPU/PV 仿真、设备代码编译和 OM 构建；实板精度验证、profiling 采集及解析由用户在内网环境执行。提供设备或远端 SSH 不作为本地验证的前提。

## 先分清两台机器

| | 编译机 A | 执行机 B（5102） |
|---|---|---|
| 工作 | 编译 OM、生成输入和 golden、发起测试、验精度、解析性能 | 加载 OM、执行、保存输出、采集原始性能数据 |
| 需要 | CANN toolkit/ATC/Bisheng、C++ 工具链、源码版 PyPTO、PTO-ISA、gym、本仓库、Python venv | 匹配的驱动和 AscendCL runtime、Python 3.7.9+ |
| 采集性能时另需 | SDK 的 `analysis/msprof/msprof.py` 及其依赖 | 可执行的 `msprof` 采集器，例如 `/var/msprof` |

编译机可以没有 NPU。执行机不安装 PyPTO、PTO、gym、ONNX、OPP 或编译工具，也不接收 golden。使用下文双机自动化命令时，内网编译机需能通过 SSH/SCP 访问执行机；本地仿真和构建不需要连接执行机。

编译机和执行机无法直连时，使用 [静态出包与离线测试](docs/OFFLINE_TESTING.md)：`scripts/static_1952.sh build` 在编译机完成构建并生成 `runtime.tar.gz`；板端解压后用 `runtime/run.sh` 执行，生成 `results.tar.gz`；搬回编译机后用 `scripts/static_1952.sh accept` 验精度。全流程不要求 SSH，也不自动切换 Python 环境。

## 1. 准备源码和 venv

四个仓库可放在同一级目录。前三个仓库使用整理后的 `codex/1952-native-om` 分支；本仓库使用 `main`。

```bash
git clone -b codex/1952-native-om https://gitcode.com/pan-boyi/pto-isa.git
git clone -b codex/1952-native-om https://gitcode.com/pan-boyi/pypto.git
git clone -b codex/1952-native-om https://gitcode.com/pan-boyi/pypto-gym.git
git clone https://github.com/Pan-Boyi/pypto_1952.git
```

已有 checkout 可继续使用，先更新对应分支。历史经过整理，旧分支有本地改动时应先保存；不要直接用 `git pull` 合并新旧历史。

继续使用 venv 隔离方案。本次复测环境是 Ubuntu 24.04 AArch64、Python 3.12、CANN 9.2；这次没有修改 Python 3.9 的兼容性。

下面记录本次复测使用的源码开发安装。PyPTO 官方安装采用 `python3 build_ci.py --clean --no_isolation` 构建，再安装生成的 `.run` 包；安装参数和 CANN 安装前缀按[官方编译安装文档](https://gitcode.com/cann/pypto/blob/master/docs/zh/install/build_and_install.md)填写。官方安装后的环境可使用第 3 节分步入口，并省略 `--build-dir`；第 2 节一键入口目前依赖源码 CMake 构建目录。

```bash
# 从四个仓库的共同父目录执行；CANN 路径换成编译机实际安装位置。
source /实际路径/cann/set_env.sh
python3.12 -m venv .venv
source .venv/bin/activate
export PYTHON="$VIRTUAL_ENV/bin/python"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYPTO_SOURCE_ROOT="$PWD/pypto"
export PYPTO_GYM_ROOT="$PWD/pypto-gym"
export PTO_LIB_PATH="$PWD/pto-isa"
export PYPTO_PRO_TOOLKIT_ROOT=/实际路径/cann

python -m pip install setuptools pybind11 wheel torch numpy decorator scipy attrs psutil sympy onnx pytest
cd "$PYPTO_SOURCE_ROOT"
PYPTO_ALLOW_WHL_BUILD=1 \
PYPTO_BUILD_EXT_ARGS='--cmake-options="-DBUILD_WITH_CANN=OFF"' \
  python -m pip install -e . --no-build-isolation
cd ../pypto_1952
python -m pip install -e . --no-build-isolation
```

首次源码构建还需要系统 C++/CMake/Python 开发工具及可用的第三方依赖缓存或下载源。`BUILD_WITH_CANN=OFF` 关闭的是 PyPTO 主机真机运行时构建；设备代码仍由 CANN 编译。

检查是否导入了正确的源码版：

```bash
python -c 'import sys, pypto, pypto_pro_atc; print(sys.executable); print(pypto.__file__); print(pypto_pro_atc.__file__)'
```

gym 的测试资产不随 wheel 打包，因此要保留 gym **源码 checkout**，不能只安装一个 gym wheel。算子修改后重新生成 OM；框架 C++ 或 Python 版本变化后，先更新对应的 PyPTO 扩展和 CMake 配置。

## 2. 直接测试已有 DecEnc

GEMMFusedKernel 使用独立的一键入口，见 [GEMMFusedKernel 编译、执行与 profiling](docs/GEMM_FUSED_KERNEL_PROFILE.md)。在编译机配置一次 SSH 和两端工具路径后，运行 `bash scripts/profile_gemm_fused_kernel.sh`，即可完成四个 kernel 编译、OM 构建、上传、执行、数据回收、精度验证、profiling 解析及终端统计打印。算子及 golden 仍来自 gym，不复制进部署仓。

在本仓库根目录执行：

```bash
bash scripts/profile_dec_enc.sh --init
```

编辑生成的 `pypto_1952/dec_enc_profile.local.json`。本地配置已被 gitignore 排除，按实际机器填写：

- `host`：编译机能访问的 SSH 别名或 `user@host`。端口、密钥写在 `~/.ssh/config`，配置里不放密码。
- `runtime_env`：执行机的 runtime 环境脚本绝对路径；非交互 SSH 已能找到 runtime 时可用 `null`。
- `toolkit_root`：编译机 CANN 路径；留空时使用 `PYPTO_PRO_TOOLKIT_ROOT` 等环境变量。
- `pypto_root`、`gym_root`、`pto_root`：三个源码目录。同级布局或上述环境变量已设置时可留空。
- `build_dir`：当前 PyPTO checkout 的 CMake 构建目录，默认 `<pypto_root>/build`；首次安装不能省略。
- `msprof_py`：编译机的 `analysis/msprof/msprof.py`；标准 SDK 布局能找到时可留空。
- `python` 和 `msprof`：**执行机**的 Python 命令和采集器路径。它们不用于选择编译机的 venv。

以后每次只填 dtype 和高宽：

```bash
bash scripts/profile_dec_enc.sh fp16 2160x3840
bash scripts/profile_dec_enc.sh fp32 1080x1920
```

入口会检查 SSH 和板端工具，增量构建 PyPTO C++ 后端，从 gym 加载算子和测试数据生成器，生成新 OM，上传、采集、回收、验精度，然后打印 Task Duration 及 cube、vec、mte1、mte2、mte3、fixpipe、scalar 的 median/mean/min/max，单位为 μs。

缺失的性能列显示 `N/A`，不会补成 0。各流水可能重叠，不能相加作为总耗时。默认预热 5 次、测量 50 次；两次精度执行和预热不会计入统计。

DecEnc 当前支持的形状、dtype、调度版本和精度标准由 gym 的 `ops/pro/experimental/native_5102` 定义。默认使用 `pipelined`；当前形状为 H=540 的 1..4 倍、W=960 的 1..4 倍，N=1、C=4。

## 3. 分步构建、验精度、采集性能

调试时可以拆开执行。下列命令均在编译机、本仓库根目录运行；每次使用新的输出和远端目录。

```bash
# 编译并打包；首次准备好源码版 PyPTO 后才可用 --build-dir 增量更新。
bash scripts/test_1952.sh build \
  --gym-root "$PYPTO_GYM_ROOT" --op dec-enc \
  --height 540 --width 960 --dtype fp16 --block-dim 8 \
  --toolkit-root "$PYPTO_PRO_TOOLKIT_ROOT" --pto-root "$PTO_LIB_PATH" \
  --build-dir "$PYPTO_SOURCE_ROOT/build" --output out/dec-enc-01

# 只执行并验精度。
bash scripts/test_1952.sh run \
  --build out/dec-enc-01 --host board5102 \
  --remote-dir /tmp/pypto-tests/dec-enc-run-01 \
  --runtime-env /板端实际路径/runtime/set_env.sh --python python3 \
  --output out/dec-enc-run-01

# 复用同一份 OM 采集性能，并再次验精度。
bash scripts/test_1952.sh profile \
  --build out/dec-enc-01 --host board5102 \
  --remote-dir /tmp/pypto-tests/dec-enc-prof-01 \
  --runtime-env /板端实际路径/runtime/set_env.sh --python python3 \
  --msprof /var/msprof --msprof-py /编译机实际路径/analysis/msprof/msprof.py \
  --warmup 5 --repeat 50 --output out/dec-enc-prof-01
```

已有 provider 也支持 `--op add`、`gemm`、`swiglu`。SwiGLU 是三个 GEMM 和一个激活节点组成的 OM，不是单个混核 kernel。DecEnc 可用 `--dec-enc-schedule baseline|optimized|pipelined` 生成独立版本对比。

矩阵测试由部署工具逐项调用，具体形状清单来自 gym：

```bash
python -m pypto_1952.dec_enc_matrix --phase build --matrix smoke \
  --gym-root "$PYPTO_GYM_ROOT" --root out/dec-enc-matrix-01 \
  --toolkit-root "$PYPTO_PRO_TOOLKIT_ROOT" --pto-root "$PTO_LIB_PATH"
```

`smoke` 是四个边界配置，`full` 是 32 个配置。后续 `--phase run` 或 `--phase profile` 使用同一个 `--root`，并补上 SSH、runtime 和 profiling 参数；完整字段见 `--help`。

## 4. 已完成的新 Pro 算子怎么接入

算子实现和正确性资产继续放在 gym。给它提供一个测试 provider 文件即可，本仓库不需要加入新算子的计算代码或 golden。

provider 暴露 `build_spec(config)`，返回：

| 字段 | 内容 |
|---|---|
| `kernels` | 名称到已有 Pro `@pl.jit` 函数的映射；不重新实现计算 |
| `graph_spec` | 单 kernel 为 `None`；多节点网络给出连线，格式见已有 gym provider |
| `block_dim` | 本次编译的核数 |
| `make_case(adapter_path, output)` | 生成输入、独立 golden 和 `case.json`，返回 case 文件路径 |
| `bind_case(case, om, binding, adapter)` | 核对算子合同、框架 OM 审计和哈希，写出绑定文件 |
| `annotate_manifest(manifest)` | 可选，记录算子测试配置；在 ATC 冻结对象前调用 |
| `analyze(natives, adapter)` | 可选，接收名称到原生 manifest 路径的映射，生成代码诊断数据 |

参考文件为 gym 的 `ops/pro/experimental/native_5102/tests/provider.py` 及其 `tests/assets`。`case.json` 按输入/输出名称记录文件、shape、dtype、字节数和 SHA256，并包含 `atol/rtol`；不要把参考输出写进 kernel。

用文件路径调用新 provider，配置项由它自己解释：

```bash
python -m pypto_1952 build \
  --provider /实际路径/pypto-gym/你的算子/tests/provider.py \
  --op 你的算子名 --case-config /实际路径/case-options.json \
  --toolkit-root "$PYPTO_PRO_TOOLKIT_ROOT" --pto-root "$PTO_LIB_PATH" \
  --output out/new-op-01
```

生成运行包后，`run/profile/compare` 的用法与 DecEnc 相同。当前执行器支持一个输出，输入/输出为 FP16 或 FP32；框架导出路径要求固定 shape、静态 tensor 指针参数、单个 Cube 或 Vector section。动态形状、额外 workspace/tiling、跨核同步或单 kernel 混核不在本次支持范围内。

## 5. 看什么才算成功

- `build/build_result.json` 的 `status=bundle-built`：OM 和运行包已生成，尚未执行。
- 板端 `execution.json` 的 `status=executed`：执行结束，尚未判断精度。
- 编译机结果目录 `result.json` 的 `status=passed` 且命令退出码为 0：精度通过；profiling 命令还要求采集和解析通过。

一键入口默认写到 `output/dec-enc-profile/<本次任务>/`。保留 `build.log`、`profile.log`、`summary.txt`、OM、设备对象、哈希、原始输出及 `profile/raw/PROF_*`。性能明细在 `profile/analysis/node_statistics.csv` 和 `profiling_summary.json`。失败时先看对应日志；不要用旧 OM 替代失败构建的产物。

原始版本的历史验证记录见 [docs/VALIDATION.md](docs/VALIDATION.md)。CPU 数值正确和 ATC 编译通过不等于新版已在真实 5102 上通过精度或达到某个性能值。

## 开发回归

本仓库的传输、runner、编排和解析测试使用合成数据，不要求 SSH、CANN 或 NPU。需能导入拆分后的 `pypto_pro_atc`；部分 DecEnc 编排用例还读取相邻 gym checkout 的轻量合同。

```bash
python -m unittest discover -s tests -v
```

框架回归留在 PyPTO/PTO-ISA；算子数值和 golden 测试位于 gym。当前迁移验证记录见 [docs/REPOSITORY_SPLIT.md](docs/REPOSITORY_SPLIT.md)。
