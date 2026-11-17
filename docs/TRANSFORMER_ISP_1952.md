# DecEncFused、FFMFused、InputFused：静态 OM 与板端验精度

这三个算子使用独立的板端验证包。每个包包含一个静态 OM、确定性输入、独立 PyTorch golden、执行器和对应的 `test_<算子>.sh`。板端只需要兼容的 1952/5102 驱动、固件、AscendCL runtime 和 Python 3.7.9+，不需要 PyPTO、PTO-ISA、ATC、PyTorch 或 NumPy。

## 源码与固定合同

编译使用以下 Pan-Boyi 仓库版本，部署工具使用 `Pan-Boyi/pypto_1952` 的 `main`：

| 仓库 | 分支 | 已验证提交 |
|---|---|---|
| [Pan-Boyi/pypto](https://gitcode.com/pan-boyi/pypto) | `codex/1952-native-om` | `5e37784051650994604e05bffba582906b0901f7` |
| [Pan-Boyi/pto-isa](https://gitcode.com/pan-boyi/pto-isa) | `codex/1952-native-om` | `62e9893255a8389e6a6fcba1f4d0def20d5217ea` |
| [Pan-Boyi/pypto-gym](https://gitcode.com/pan-boyi/pypto-gym) | `codex/1952-transformer-om` | `30a9c4188585e2f75c1ab75df3b3b88009f22d66` |

PyPTO 和 PTO-ISA 的已验证卷积支持已合并到上表所列 Pan-Boyi 仓库的 1952 分支。编译时统一从这些仓库取源码，并固定到上表提交。gym 使用三个算子的 OM 适配修复版本，provider 位于 `ops/pypto_pro/experimental/ops_transformer/<算子>/provider.py`。这里的 gym 分支与旧 DecEnc/GEMMFusedKernel 流程的 `codex/1952-native-om` 分支不同，应使用独立 checkout。

| 算子 | 默认输入 | OM 输出 | 原始容差 atol / rtol |
|---|---|---|---|
| `dec_enc_fused` | FP16 `[1,4,1080,1920]` + ISP 参数/权重 | `output`: FP16 `[1,16,1080,1920]` | `0.0005 / 0.001` |
| `ffm_fused` | FP16 `[1,16,1080,1920]` + 权重/色彩系数 | `yout`: `[1,4,1080,1920]`; `uvout`: `[1,2,1080,1920]`，均 FP16 | `0.003 / 0.003` |
| `input_fused` | FP16 `[1,4,1080,1920]` + global/local 两套编码器参数 | `features`: FP16 `[17,128]` | `0.003 / 0.005` |

卷积 bias 按 `round(bias * 65536)` 编码为 **INT32 Q16**，由 provider 自动生成；不能把 FP32 bias 的原始字节传给 DAV5102 OM。ISP、色彩投影等系数仍按源代码使用 FP32。InputFused 的 feature 第 0 行是 global，后 16 行是按行排列的 local views。

## 修复内容

- DecEncFused 为输出增加 `pl.Output`，图像维度改为 `pl.STATIC`，由 `static_shapes` 固定到 OM 的形状。原来的 `pl.DYNAMIC` 会保留四个运行时维度标量，无法进入 tensor-only 静态 ABI。
- FFMFused 为 Y、UV 增加输出标记，保留两个独立输出。
- InputFused 将原来的三个 tiling-key 阶段拆成输出契约明确的 resize、encoder、reduce 节点，连接为一个 OM。GE 为 reduce 分配独立输出，因此最后一阶段显式复制 16 行 local features，再保留原 global reduction 及 INT32 partials 的位传输方式。
- 执行器支持 INT32 输入和多输出；每个输出分别保存两轮原始数据、核对描述符、检查有限性和误差，并检查两轮结果一致。

## 编译机出包

以下操作均在编译机进行。首次准备时，使用一个新的工作目录，把四个仓库放在同级：

```bash
mkdir -p ~/isp-1952
cd ~/isp-1952
git clone -b codex/1952-native-om https://gitcode.com/pan-boyi/pypto.git
git clone -b codex/1952-native-om https://gitcode.com/pan-boyi/pto-isa.git
git clone -b codex/1952-transformer-om https://gitcode.com/pan-boyi/pypto-gym.git
git clone https://github.com/Pan-Boyi/pypto_1952.git

# 固定到上表已验证版本；创建本地分支以便后续开发。
git -C pypto switch -c isp-pr2-validated 5e37784051650994604e05bffba582906b0901f7
git -C pto-isa switch -c isp-pr2-validated 62e9893255a8389e6a6fcba1f4d0def20d5217ea
git -C pypto-gym switch -c isp-om-validated 30a9c4188585e2f75c1ab75df3b3b88009f22d66
```

已有 checkout 时先保存本地改动，再 `git fetch origin`、切换并快进到对应分支；不要在旧源码目录中直接覆盖有改动的文件。PyPTO 更新后必须同时重建 C++ 接口与 Python 绑定，不能只替换 Python 文件。

使用 Python 3.10+ 编译环境，本次验证为 Python 3.12、CANN 9.2.0。需要系统 C++/CMake/Python 开发工具和可用的第三方依赖缓存或下载源。将 CANN 路径换成编译机实际安装位置：

```bash
cd ~/isp-1952
source /实际路径/cann/set_env.sh
python3.12 -m venv .venv
source .venv/bin/activate
export PYTHON="$VIRTUAL_ENV/bin/python"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYPTO_SOURCE_ROOT="$PWD/pypto"
export PYPTO_GYM_ROOT="$PWD/pypto-gym"
export PTO_LIB_PATH="$PWD/pto-isa"
export CANN_SDK_ROOT=/实际路径/cann

python -m pip install 'setuptools>=70.3' pybind11 wheel torch numpy decorator scipy attrs psutil sympy onnx pytest
cd "$PYPTO_SOURCE_ROOT"
PYPTO_ALLOW_WHL_BUILD=1 \
PYPTO_BUILD_EXT_ARGS='--cmake-options="-DBUILD_WITH_CANN=OFF"' \
  python -m pip install -v -e . --no-build-isolation
cd ../pypto_1952
python -m pip install -e . --no-build-isolation
export PYTHONPATH="$PYPTO_SOURCE_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import sys, pypto, pypto_pro_atc; from pypto import pypto_impl; print(sys.executable); print(pypto.__file__); print(pypto_impl.__file__); print(pypto_pro_atc.__file__)'
```

`BUILD_WITH_CANN=OFF` 仅关闭 PyPTO 主机真机运行时构建，设备代码仍使用 CANN 编译。打印的模块路径应指向当前 PyPTO checkout；不要混用旧的 `pypto_impl`。

一条命令依次编译三个算子，并生成统一板端测试包：

```bash
# 在 pypto_1952 仓库根目录；每次使用新的输出目录。
bash scripts/build_transformer_suite.sh "$PWD/out/isp-01" \
  --gym-root "$PYPTO_GYM_ROOT" \
  --toolkit-root "$CANN_SDK_ROOT" --pto-root "$PTO_LIB_PATH" \
  --target-soc MC62CM12AA
```

每个算子在独立进程中生成输入和完整形状的 golden、编译 native kernel、调用 ATC 生成新 OM、校验 ABI 并打包。任一阶段失败即停止，不复用旧 OM，也不会生成统一成功包。保留至少 3 GB 可用磁盘空间。默认八个 block；FFM 与 InputFused 固定 1080×1920，FFM 比较全部六个输出通道，InputFused 比较全部 17×128 个 features。

产物：

```text
out/isp-01/
  dec_enc_fused/           native、ATC 日志、OM、输入、golden 和独立板端包
  ffm_fused/
  input_fused/
  <算子>.log               各算子的编译终端日志
  board_suite.tar.gz      搬到板端的统一包
  board_suite.SHA256SUMS   统一包校验和
```

仍可单独编译一个算子；DecEncFused 还可追加 `--height 31 --width 113` 等静态形状用于尾块检查：

```bash
bash scripts/build_transformer_ops.sh build --op dec_enc_fused \
  --gym-root "$PYPTO_GYM_ROOT" \
  --toolkit-root "$CANN_SDK_ROOT" --pto-root "$PTO_LIB_PATH" \
  --target-soc MC62CM12AA --output out/dec_enc_fused-01
```

`PYTHON=/所选环境/bin/python` 可指定编译解释器，脚本不自动切换环境。已有构建只更新板端脚本时，可运行 `bash scripts/build_transformer_ops.sh package --build out/ffm_fused-01 --output out/ffm_fused-board-02`，不会重新编译 OM。

## 板端一次验证三个算子

把 `board_suite.tar.gz` 和 `board_suite.SHA256SUMS` 搬到板端同一全新目录，在该目录执行：

```bash
sha256sum -c board_suite.SHA256SUMS
tar -xzf board_suite.tar.gz
bash board_suite/test_transformer_suite.sh /板端结果/isp-01 \
  --runtime-env /板端/runtime/set_env.sh --device 0
```

脚本按 DecEnc、FFM、InputFused 顺序核对各包哈希，解压并调用对应独立验证入口，任一失败立即停止。`--runtime-env` 使用板端自己的环境脚本，环境已配置时可省略。输出根目录必须全新；每个算子的报告在 `/板端结果/isp-01/<算子>/result/validation.json`，日志和原始结果也在该目录下。只有三者均通过，终端才打印 `All three operator tests passed`。板端使用 Python 3.7.9+，不安装编译端 venv。

先做离线包校验时可追加 `--verify-only`；它不会访问 NPU，也不表示实板精度通过。

## 板端分别执行

每个算子的包放入独立目录，避免三个同名 `runtime.tar.gz` 相互覆盖。在对应目录中：

```bash
sha256sum -c SHA256SUMS
tar -xzf runtime.tar.gz
```

先只检查包和 golden，不加载 ACL、不访问 NPU：

```bash
bash runtime/test_dec_enc_fused.sh --verify-only
# 另两个包分别使用 test_ffm_fused.sh、test_input_fused.sh。
```

三个包各自的运行命令：

```bash
# 在 DecEncFused 包所在目录
bash runtime/test_dec_enc_fused.sh --runtime-env /板端/runtime/set_env.sh \
  --device 0 --output /板端结果/dec_enc_fused-01

# 在 FFMFused 包所在目录
bash runtime/test_ffm_fused.sh --runtime-env /板端/runtime/set_env.sh \
  --device 0 --output /板端结果/ffm_fused-01

# 在 InputFused 包所在目录
bash runtime/test_input_fused.sh --runtime-env /板端/runtime/set_env.sh \
  --device 0 --output /板端结果/input_fused-01
```

板端环境已配置时可省略 `--runtime-env`；指定时须放在第一个参数。板端可通过自己的 `PYTHON` 选择解释器。结果目录必须全新且位于 `runtime/` 外部。

只有脚本退出码为 0、打印 `<算子>: PASSED`，且 `validation.json` 的 `status=passed` 才表示精度通过。`status=executed` 只表示 ACL 执行完成。脚本不会调整容差来通过验证。

```text
<结果目录>/
  validation.json          板端精度结论，各输出最大绝对误差/非有限值/不一致元素数
  board_run.json           执行过程和运行器返回码
  runner.log               ACL 日志；失败时保留诊断
  results/execution.json   模型实际张量描述符和两轮输出哈希
  results/actual*.bin      第一轮输出；FFM 的 UV 为 actual_1.bin
  results/actual_second*.bin
  results.tar.gz           原始执行材料，可搬回编译机再次验收
```

可在编译机对原始结果再次验收：

```bash
bash scripts/static_1952.sh accept --build out/ffm_fused-01 \
  --results-archive /回收/ffm_fused/results.tar.gz --output out/ffm_fused-check-01
```

## 验证范围

PR #2 同步前，使用 Ubuntu 24.04 AArch64 编译机、Python 3.12、CANN 9.2.0、目标 `MC62CM12AA`，实际重新执行三个完整形状的原生设备编译和 ATC，并校验所有板端包。PyPTO PR 回归 45 项、PTO-ISA PR 回归 38 项通过。原始分支失败证据、修复后的 native manifest、ATC 日志、OM/运行包 SHA256 随产物保存。

部署框架 122 项回归通过，使用合成 ACL 覆盖真实打包、隔离 Python 校验、INT32 输入、双输出执行/比较和未写回输出的失败判定；新批量入口还覆盖编译失败停止、板端失败停止、损坏包拒绝和仅校验时不报精度通过。旧命令的回归使用其原有 gym 测试资产。独立 CPU oracle 回归覆盖权重打包、Q16 bias、DecEnc 区域边界/尾块/零指数以及 FFM sigmoid/green 舍入/独立 Y 通道。CPU 模拟器尚不支持 INT32 位传输/寄存器，reducer 的整段数值回归明确跳过；其 native 编译和 OM 构建已通过。

本环境没有执行板端 NPU 验证，也没有测得三个算子在 1952 上的性能；实板精度以以上脚本生成的 `validation.json` 为准。
