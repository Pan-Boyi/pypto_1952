# DecEncFused、FFMFused、InputFused：静态 OM 与板端验精度

这三个算子使用独立的板端验证包。每个包包含一个静态 OM、确定性输入、独立 PyTorch golden、执行器和对应的 `test_<算子>.sh`。板端只需要兼容的 1952/5102 驱动、固件、AscendCL runtime 和 Python 3.7.9+，不需要 PyPTO、PTO-ISA、ATC、PyTorch 或 NumPy。

## 源码与固定合同

基于用户提供的源码版本：

| 仓库 | 分支 | 基线提交 |
|---|---|---|
| joshjms/pypto-gym | `joshjms/dec-enc-fused` | `8e918608ae77e5ad8bb9bc5295a743c66dc0d874` |
| joshjms/pypto | `codex/pypto-pro-5101-swiglu` | `5e37784051650994604e05bffba582906b0901f7` |
| joshjms/pto-isa | `codex/pypto-pro-5101-swiglu` | `62e9893255a8389e6a6fcba1f4d0def20d5217ea` |

编译时使用包含本次修复的 gym checkout，provider 位于 `ops/pypto_pro/experimental/ops_transformer/<算子>/provider.py`。PyPTO 和 PTO-ISA 使用上表版本；这两个依赖的源码未做额外修改。更新 PyPTO 后需同时重建、安装 C++ 接口和 Python 绑定，避免混用旧的 `pypto_impl`。

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

先激活自己的编译 Python 环境并 source CANN。三个源码仓可与部署仓放在同级目录。

```bash
export PYPTO_SOURCE_ROOT=/源码/pypto
export PYPTO_GYM_ROOT=/源码/pypto-gym
export PTO_LIB_PATH=/源码/pto-isa
export CANN_SDK_ROOT=/编译机/cann
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
source "$CANN_SDK_ROOT/set_env.sh"

cd /源码/pypto_1952
for op in dec_enc_fused ffm_fused input_fused; do
  bash scripts/build_transformer_ops.sh build --op "$op" \
    --gym-root "$PYPTO_GYM_ROOT" \
    --toolkit-root "$CANN_SDK_ROOT" --pto-root "$PTO_LIB_PATH" \
    --target-soc MC62CM12AA --output "out/$op-01"
done
```

每个 `--output` 必须是全新目录；失败不会回退到旧 OM。默认使用八个 block，FFM 与 InputFused 固定 1080×1920。DecEncFused 可在构建时追加 `--height 31 --width 113` 等形状，生成独立静态 OM，用于边界与尾块检查。`PYTHON=/所选环境/bin/python` 可指定编译解释器；脚本不自动切换环境。

新出包目录为 `out/<算子>-01/board/`，其中 `runtime.tar.gz` 和 `SHA256SUMS` 搬到板端。输入/golden 由完整形状生成，FFM 比较全部六个输出通道，InputFused 比较全部 17×128 个 features。

已有构建只更新板端脚本时，可重新打包：

```bash
bash scripts/build_transformer_ops.sh package \
  --build out/ffm_fused-01 --output out/ffm_fused-board-02
```

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

本地使用 Ubuntu 24.04 AArch64 编译机、CANN 9.2.0、目标 `MC62CM12AA`，实际执行原生设备编译和 ATC。原始分支失败证据、修复后的 native manifest、ATC 日志、OM/运行包 SHA256 随产物保存。

部署框架回归使用合成 ACL，覆盖真实打包、隔离 Python 校验、INT32 输入、双输出执行/比较和未写回输出的失败判定；旧命令的回归使用其原有 gym 测试资产。独立 CPU oracle 回归覆盖权重打包、Q16 bias、DecEnc 区域边界/尾块/零指数以及 FFM sigmoid/green 舍入/独立 Y 通道。CPU 模拟器尚不支持 INT32 位传输/寄存器，reducer 的整段数值回归明确跳过；其 native 编译和 OM 构建仍需通过。

本环境没有执行板端 NPU 验证，也没有测得三个算子在 1952 上的性能；实板精度以以上脚本生成的 `validation.json` 为准。
