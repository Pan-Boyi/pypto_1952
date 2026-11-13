# 静态出包与离线测试：编译、执行环境分离

本流程不要求编译机访问板卡，也不调用 SSH/SCP。通用脚本在 `pypto_1952`；算子、输入生成、golden 和误差阈值仍由 gym 的 provider 提供。新算子无需改 PyPTO 或 PTO-ISA，前提是其操作和静态导出 ABI 已支持。

| 环境 | 工作和依赖 |
|---|---|
| 编译机 A，可以无 NPU | 当前 Python 环境、PyPTO/Pro、PTO-ISA 头文件、gym、pypto_1952、CANN toolkit/Bisheng/ATC。生成设备代码、OM、输入/golden，回收结果后验精度。 |
| 执行机 B，5102/1952 | 兼容驱动/固件、AscendCL runtime、Python 3.7.9+。只接收运行包，加载 OM、保存原始输出。 |

脚本使用 `PYTHON` 指定的编译机解释器，未设置时用当前 PATH 中的 `python3`。不会自动创建、激活或切换 venv/conda。可使用已有 conda 或 venv，但所选环境必须已具备当前分支的编译依赖；本仓项目声明的编译侧 Python 版本是 3.10+，没有因此扩大到 Python 3.9 的完整构建兼容承诺。板端标准库脚本单独保持 Python 3.7.9+。

## 1. 编译机静态构建并出包

先激活自己选择的 Python 环境，再在同一个 shell 中 source 编译机 CANN。路径按实际安装填写；不由脚本安装依赖或下载 SDK。

```bash
source /编译机/cann/set_env.sh
export PYTHON=/所选Python环境/bin/python
export PYPTO_SOURCE_ROOT=/源码/pypto
export PYPTO_GYM_ROOT=/源码/pypto-gym
export PTO_LIB_PATH=/源码/pto-isa
export PYPTO_PRO_TOOLKIT_ROOT=/编译机/cann
export PYPTO_1952_ROOT=/源码/pypto_1952
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

cd "$PYPTO_1952_ROOT"
bash scripts/static_1952.sh build \
  --gym-root "$PYPTO_GYM_ROOT" --op dec-enc \
  --height 540 --width 960 --dtype fp16 --block-dim 8 \
  --toolkit-root "$PYPTO_PRO_TOOLKIT_ROOT" --pto-root "$PTO_LIB_PATH" \
  --output out/dec-enc-static-01
```

`build` 完整复用原 `test_1952.sh build` 的参数和编译实现，再自动执行离线打包。首次源码配置按仓库 README 安装。只有需要增量更新当前源码 checkout 的 C++ 扩展时才传 `--build-dir "$PYPTO_SOURCE_ROOT/build"`，不强制它作为已安装环境的前置条件。

GEMMFusedKernel 在 gym 有专用入口，自动选择 provider、op、dtype 和默认 8-block 调度：

```bash
bash "$PYPTO_GYM_ROOT/ops/pro/experimental/gemm_fused_kernel/scripts/test_static_1952.sh" \
  build --output /产物/gemm-fused-kernel-fp16-b8-01
```

该入口从上述 `PYPTO_PRO_TOOLKIT_ROOT` / `PTO_LIB_PATH` 取工具链路径，或者显式传 `--toolkit-root` / `--pto-root`。用 `--block-dim 4` 和另一个全新 `--output` 构建 4-block 版本。默认同级布局中可不设置 `PYPTO_1952_ROOT`。可从任意工作目录调用，`--output` 的相对路径相对于部署仓根目录。

任意其他 provider 直接使用通用入口：

```bash
bash scripts/static_1952.sh build \
  --provider /源码/pypto-gym/某算子/provider.py --op 某算子 \
  --case-config /实际配置.json \
  --toolkit-root "$PYPTO_PRO_TOOLKIT_ROOT" --pto-root "$PTO_LIB_PATH" \
  --output out/new-static-01
```

已有经过审计的构建可只出包：

```bash
bash scripts/static_1952.sh package \
  --build out/dec-enc-static-01 --output out/dec-enc-repackage-01
```

所有输出目录均要求全新，已有目录不会被覆盖。编译或 ATC 失败不会转而使用旧 OM，也不会输出成功运行包。

## 2. 产物和传输边界

```text
out/<任务>/
  native/                   设备 C++、ELF、编译日志、设备 manifest
  atc/                      ONNX、ATC 命令/日志、frozen ELF、OM/ABI/可移植性审计
  case/                     输入、golden、误差阈值、case 和模型绑定，留编译机
  bundle/                   原有输入与 OM 运行材料
  build_result.json         status=bundle-built，device_execution=not_run
  offline/
    runtime.tar.gz          只将这个包搬到执行机
    SHA256SUMS              运行包 SHA256
    package_result.json     编译环境、构建记录、OM/case/request/archive 哈希
    runtime/
      runtime.json          完整运行材料清单和哈希
      run.sh                板端 shell 入口
      offline_run.py        标准库检查、执行和结果打包
      bundle/               只有 OM、输入、运行器和 execution request
```

golden、case/binding、源码、主机 ATC 插件、OPP、ONNX 和编译依赖均不进入 `runtime.tar.gz`。运行器从 OM 获取张量描述，不在板端生成输入或判定精度。可用人工复制、移动介质或内网传输，不要求建立双机连接。

## 3. 执行机离线检查与运行

搬运 `runtime.tar.gz` 和对应 `SHA256SUMS` 后，在一个新目录解压：

```bash
sha256sum -c SHA256SUMS
tar -xzf runtime.tar.gz

# 不加载 ACL、不访问设备，只核对完整包清单和哈希。
bash runtime/run.sh --verify-only

# runtime 环境已经配置时：
bash runtime/run.sh --device 0 --output /板端任务/gemm-fused-kernel-run-01

# 需要显式 source 板端环境时，将它放在第一个参数：
bash runtime/run.sh --runtime-env /板端/runtime/set_env.sh \
  --device 0 --output /板端任务/gemm-fused-kernel-run-02
```

板端默认用其当前 `python3`；如需指定，使用板端自己的 `PYTHON=/板端/Python/bin/python`，不要沿用编译机 Python 的绝对路径。板端 runtime 路径独立配置，不能照抄编译机 CANN 路径。

`--output` 必须在解压后的 `runtime/` 目录之外，且目录未存在。运行前核对清单、OM 和输入哈希；两轮执行分别预填不同哨兵，保存输出供编译侧验收。不论模型执行成功还是失败，已启动的运行流程均尽量保存日志并生成结果包：

```text
/板端任务/<运行>/
  board_run.json
  runner.log
  results/execution.json
  results/actual.bin
  results/actual_second.bin
  results.tar.gz            搬回编译机的文件；失败时可能只有部分结果
```

退出码 0、`status=executed` 仅表示执行完成，**不表示精度通过**。失败返回非零；保留结果包中的诊断，不以手工修改 execution.json 的方式绕过失败。

可选 `--warmup 10 --repeat 100`：两轮精度执行之后预热和采样，同时保存最后一轮输出。计时范围是同步 `aclmdlExecute` 整个 OM 的主机墙钟时间，不包括模型加载和初次 H2D；它不等于设备 kernel 时间，也不替代 msprof 流水指标。需要设备级 profiling 时继续使用原有 `profile` 入口。

## 4. 编译机回收并验精度

只将板端生成的 `results.tar.gz` 搬回编译机，使用原构建目录验收：

```bash
cd "$PYPTO_1952_ROOT"
bash scripts/static_1952.sh accept \
  --build /产物/gemm-fused-kernel-fp16-b8-01 \
  --results-archive /回收/results.tar.gz \
  --output /产物/gemm-fused-kernel-check-01
```

也可用 gym 算子脚本的 `accept` 子命令，参数相同。验收复用原 `host.evaluate`，检查实际运行的 OM/request、输入/输出合同、双哨兵一致性及有限性，最后和编译侧 golden 按阈值比较。若有重复计时，还检查调用计数和最终输出精度；通过后才给出 whole-OM 主机时间分布。

结果保存在新目录的 `result.json`，解压原始材料放 `returned/`。**命令退出码为 0 且 `status=passed` 才表示精度通过**。失败、错误模型、漏写输出及错误结果文件都会返回非零。原始日志和输出保留用于定位。

更改 kernel、调度或 ABI 之后，必须重新编译 OM 并出包；更换输入/参数需重新生成对应的 case/golden。不要在解压后修改哈希清单里的文件。

## 开发验证

```bash
python -m unittest tests.test_offline -v
python -m unittest discover -s tests -v
```

离线流程测试使用合成 ACL，覆盖实际打包、隔离 Python 校验、真实运行器逻辑与精度比较，以及失败回传和包损坏；合成用例不代表真实 NPU 执行。真实设备编译/ATC 出包验证记录随本次产物单独保存。实板精度和性能仍由内网环境执行并补充。
