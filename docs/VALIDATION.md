> 历史记录：以下内容对应拆仓前的提交和目录。当前命令以仓库根 README 为准；旧提交号仅用于追溯原验证，不能用于组合当前四仓版本。

# 验证记录与当前边界

更新：2026-11-10。编译环境：Ubuntu 24.04 AArch64 Multipass，实际安装的 CANN 9.2。
9.3 尚未安装验证；工具按能力检查，不用版本号白名单。

## DecEnc 一键编译与 profiling（2026-11-10）

新增 `profile_dec_enc.sh <dtype> <HxW>`，首次保存一次 SSH、板端 runtime 和本机路径配置，
之后只指定 dtype/shape。自动做连接预检查、C++ 增量构建、原生 Pro/PTO/ATC 编译、上传采集、回收、
编译机精度验收和 `msprof.py` 解析；终端打印 Task Duration 与七类流水的 median/mean/min/max。
精度、采集或解析失败不发布合格性能表。缺失硬件列显示 N/A，真实 0 保留。

- 部署工具完整回归 123 项通过，新增一键参数、配置、阶段顺序、失败中止和报告输出测试。
- 三项集成测试使用模拟编译与 SSH/SCP，实际经过现有 capture/精度门禁、运行 exporter 子进程、
  解析已知 CSV，再由一键入口打印七类统计；同时验证最终精度失败和 exporter 失败时保留原始数据、拒绝性能表。
  合成数据不作为真实性能结果。
- CSV 兼容大小写、空格和下划线拼写，并补齐 `fixpipe_exe_time(us)`；保留原始列名供核对。
  cycles、比例、ns/ms、无单位列以及另一类核的计数不作为本表微秒耗时；歧义列明确拒绝。
- ARM 虚拟机实际跑过一键入口的编译阶段：540×960 FP16、默认 12 行，自动更新 interface、调用 CANN 9.2
  生成 OM、case 和运行包，编译机完整文件校验通过。修正了 ATC 子 Python 必须使用当前虚拟环境 PATH 的问题。
- 本轮未访问内网板端。真实构建验证明确隔离 SSH/profiling 阶段；不能据此声称一键脚本已完成实板采集。

证据为验证环境 `adaptation/evidence/dec-enc-oneclick-*` 与 `adaptation/native-dec-enc-oneclick/`。
用法见 [DEC_ENC.md](DEC_ENC.md#一键编译并采集性能)，本轮没有修改设备计算实现。

## 参考 AscendC 的 DecEnc 流水版（2026-11-09）

用户补充：AscendC 在 2160×3840 FP16 上约 4000μs。已核对远端参考分支，HEAD 仍为
`ac63bc9f289b67684d5a7372c0708a63e8f8a351`。本地没有该次原始 profiling，未独立复测这份时延。

新增默认 `pipelined`，保留下面两轮的 `baseline`、`optimized`。使用 Pro 已有的定向 set/wait，
实现输入两组×四通道预取、输出两个单通道槽交替写回。FP16 的升降精度放进 VF 寄存器，
默认每块 12 行；FP32 每块 6 行。热循环没有 ALL 屏障，设备入口只在退出时统一等待一次。
大输入/输出经新可选 `bypass_l2=True` 对应 PTO L2 hint，参数搬运保留默认策略。
这迁移了 AscendC 的排布方式，计算和 OM 生成仍全部走原生 Pro/PTO 框架。
对应 PTO 提交为 `27108d6bb33fb28c93b01f9570a3483cfda1db19`，需要与本轮 PyPTO 一起更新。

本轮验证结果：

- 最终全部 16 形状×2 精度的 32 份流水版 OM，通过真实 CANN 9.2 编译、ATC、主机依赖审计和实际任务 ABI 核对。
  FP16 默认 12 行、FP32 默认 6 行，全部 blockDim=8、80B 参数。
- 完整 Pro 数值用例中，两种精度各三个八核 CPU 对照通过；另增用户实际 2160×3840 FP16、12 行分块的
  完整 CPU 对照，通过全部 265420800 个输出元素。输入行、通道和空间参数组有不同值，可检出错取缓冲或错组。
  两种代码生成和两种真实设备导出也通过，该模块合计 11 项。CPU 不模拟真实异步时序或性能。
- 2160×3840 FP16 从统一 build 入口完成最终 OM、case、绑定和运行包，编译机完整文件校验通过。
  九份输入与 530841600 字节 golden 的描述/哈希，均与原始版本生成的数据一致。
- 最终 4K FP16 对象重编与冻结对象逐位相同，SHA256 为
  `0b13937811c7059d70d5c0aedbe503c092a48fd0471704cfcd04ae392b0f10a2`。
  LLVM 中只有设备退出的一次 ALL；五个方向的定向事件、两组输入地址选择和寄存器转换均保留。
  输入/输出 cache 字段为 4，八张参数表为 0；没有 GM 参数标量读取，也没有整块 TCVT。
- CPU/VF/事件/L2/Cube/固定点 IR 合并回归 124 项通过，PTO 新旧编译/LLVM/CPU 回归 73 项通过，部署工具 105 项通过。
  各集合有重叠，不相加计数。事件测试拒绝漏等、重复 set、非法 id 和退出遗留，并检查失败后重启隔离。
- 前端旧 block_ops 测试另有 131 项通过；1 项 `test_make_tile_missing_addr_rejected` 仍因 Python 3.12
  错误文案含 `keyword-only` 而与旧正则不符。该 make_tile 实现未修改，保留为已知测试问题。
  新增文件 Ruff、新增 C++ 代码格式和 diff 检查通过；两个旧文件的长行及 C++ 既有格式问题未改。

证据位于验证环境 `adaptation/native-dec-enc-pipeline/` 和 `adaptation/evidence/dec-enc-pipeline-*`；
最终对象的源文件、LLVM、重编日志和哈希在 `fp16-max-l2/native-audit/`。
git pull 后须重建 C++ interface（build 加 `--build-dir`），再生成新 OM。板端 runner 未改动，仍支持 Python 3.7.9。

该版没有沿用上一轮的两路 pow/mix 合并：每块 32 个单输出 VF，便于尽早写回。
分块增大减少整图启动次数，但 log 与 FP16 升精度会重复计算，因此不从代码计数推算加速比。
实际实板精度、时延和硬件流水重叠均待内网复测。完整命令见 [DEC_ENC.md](DEC_ENC.md)。

## DecEnc scalar 开销优化（2026-11-09）

用户反馈：上一版已在 1952 实板验精度通过，但 2160×3840 的 Task Duration 约 13000μs，
scalar 耗时偏高。尚未收到 dtype 和原始 profiling，因此该数字只作为用户反馈保留。

本次默认改为 `optimized`，同时保留 `baseline`。八张运行时参数表先 DMA 到每核 UB，
全局参数在入口读取，空间组参数在换组时读取；小块循环不再逐个读 GM 参数。
四通道静态展开，两路 pow 与两路 mix 分别合入原生 Pro VF；此算子按实际缓冲依赖放置手工屏障。
九输入、一输出、80 字节启动参数、八核分工、数学公式、FP32 累加与最终 FP16 舍入均保持原合同。
PTO 新增严格按有效字节搬运的短 TLOAD 和 5102 VF 寄存器包装，CPU 路径补齐同一 Pro 定义的仿真支持。
本轮 PTO 对应提交为 `b62069d9e3bb625a176f94c3838b369cf194ae4e`，PyPTO/PTO 均使用
Pan-Boyi 仓库的 `codex/1952-native-om` 分支，需要一起更新。

本轮实际检查：

- 优化版全部 16 形状×2 dtype 的 32 份 OM，通过 CANN 9.2 Bisheng、ATC、设备代码审计和实际 TASK_INFO ABI 核对。
- FP16、FP32 各三个完整八核 CPU 数值对照通过，覆盖 540×960、540×1920、1080×2880；
  包含跨空间组分工，使用未改动的独立参考与容差。
- 两版最小 FP32 从统一 build 入口完整出包；编译机逐文件校验通过，九份输入及 golden 描述/哈希完全一致。
  两版对象均与 LLVM 诊断对应的冻结对象逐位一致。
- CPU/集成回归 51 项、固定点 IR 回归 31 项通过；Pro 用例另有 10 项通过，集合有重叠，不相加计数。
- PTO 短搬运、VF 寄存器与原向量回归 48 项通过。FP16 单元素参数的真实 LLVM 搬运宽度为 2 字节，
  另有保护页和 UB guard 测试，未通过多读 GM padding 绕过短参数问题。
- 部署工具 103 项通过；包括 A/B 选择、独立目录、代码证据绑定和报告篡改拒绝。Ruff 与 diff 检查通过。
  板端 runner 未改动，继续使用上一轮 Python 3.7.9 执行协议。

对 540×960 FP32、rows=3、8 核的真实优化后 LLVM，按实际循环次数加权：

| 每个 3×960 小块的热循环操作 | baseline | optimized |
| --- | ---: | ---: |
| VF 调用 | 88 | 24 |
| ALL 屏障 | 424 | 65 |
| GM 参数标量读取 | 56 | 0 |

新版仍在每核入口做 8 次参数 DMA，并在入口和换组处读 UB 参数；上表不表示完全没有参数访问。
诊断重编对象 SHA256：baseline `209ca58a30d207d48a2b419304d79285614fb4b3943fdec798d101460d3c8aeb`；
optimized `61572879b2276e590a9ad05d9ae0e6674ef8ba2bc391bf800aec9a180d9e0fb3`。
证据保存在验证环境 `adaptation/native-dec-enc-perf/`、`adaptation/evidence/dec-enc-perf-*`。

这是编译结构和 CPU 数值证据，**尚无优化版实板精度或加速比结论**。默认 tile 未增大，未做双缓冲，
不能从减少的指令次数直接推算 13000μs 会降到多少。两版 4K profiling 命令见 [DEC_ENC.md](DEC_ENC.md)。

## 多核 DecEnc（2026-11-09）

原合同取自 ops-nn `dec-enc-fused-kernel-back` 的 `ac63bc9f289b67684d5a7372c0708a63e8f8a351`。
PTO 对应提交为 `70a5e9651f5052892b850e9550799510c5a5a69a`。
当前由一份原生 Pro 定义生成一个融合 Vector kernel；九个运行时输入和一个输出，
十个 GM 指针共 80 字节，无额外 workspace/tiling 参数。默认 blockDim=8，
使用 coupled AiCore block 索引；四通道输入在 UB 中供八路复用，直接写入最终输出。

- 全部 16 形状×2 dtype 的 32 份八核 OM 已通过真实 Bisheng 编译、ATC 和设备代码独立性审计。
  实际解码 TASK_INFO，逐项核对入口、blockDim=8、80 字节参数、输入/输出指针描述；
  未把图属性或 ELF 文件头当作实际启动 ABI 的替代。
- 最小 540×960、最大 2160×3840 各两种精度，共四个边界用例，已从构建入口完成 golden、
  绑定与运行包生成，并通过编译机完整文件校验；最大 FP32 输出为 1,061,683,200 字节。
- 同一完整 Pro 定义的 FP16、FP32 CPU 八核分工对照通过，覆盖全部八路输出。
  CPU 逐核重置私有仿真内存并顺序执行，验证分工和数学，不模拟硬件并发或性能。
- 新增 5102 原生 TLOG、FP16↔FP32 TCVT、packed TCMPS/TSELS 和 FP32 TAXPY。
  真实编译/LLVM 指令、CPU packed mask、负例和旧 Vector 回归共 23 项通过。
- 修复 CPU scalar/tensor TCMPS 对带 padding mask 的行排布问题，并验证比较→选择、
  六种比较、尾段、padding 和保护区域。
- 板端 runner 已用真实 Python 3.7.9 加 `-S` 跑过 25 项测试。
  H2D、不同哨兵预填、D2H 落盘和 SHA 均分块执行，golden 仍只在编译机。
- 编译机实际比较过两份各 1,061,683,200 字节的 FP32 文件，最后一个元素的错误能准确检出；
  进程最大常驻内存约 45.3 MiB。这是主机 IO/比较验证，不是算子性能数据。
- 部署工具 90 项、ATC 适配层 54 项在依赖齐全的 Ubuntu 环境全部通过，无跳过；
  CPU 回归 50 项通过，包含完整 DecEnc 两种精度；Pro/设备导出新旧用例另有 11 项通过。
  这些测试集合有重叠，不相加作为算子数量或独立硬件验证数量。

本地证据位于验证环境 `adaptation/native-dec-enc/`、`adaptation/evidence/dec-enc-*`
和 `adaptation/dec-enc-vector-ut-final3/`。板端位于内网，本轮没有实板执行或性能结论；
32 份 OM 仍保留 `device_execution=not_run`。验收命令见 [DEC_ENC.md](DEC_ENC.md)。

## 已完成的 Add/GEMM 验证

用户内网测试报告：旧 OM 在 x86 编译后，AArch64 执行机尝试加载 x86 主机插件而失败。
此前仅检查设备 ELF 和任务属性，不能据此保证 OM 没有主机依赖。

接入层现改为标准 AI Core 路径：`REG_OP` 推导形状，ONNX 注册使用
`.ImplyType(TVM)`，标准 TBE 编译回调校验 shape、dtype、ND 格式和哈希，
只复制冻结的原生 PTO 对象和启动 JSON，再由 `context.add_build_res` 交给 CANN。
入口名保持不变，JSON 不含会触发自动 tiling 的 `compileInfo`。
主机插件不再嵌设备 ELF，也不注册 `CustomOpFactory`。没有引入 ops-nn、
AscendC 算子工程或手写 Host tiling。

新工具实际解析 OM 分区与 ELF，拒绝主机插件、自定义运行时扩展和不支持的格式。
制作绑定时再次检查，只有主机依赖审计通过才生成 `schema_version=2` 的 binding，
其中记录 `runtime_dependencies=device_kernel_only` 和审计证据。
旧 OM、binding 和 bundle 均不再交付，必须重新构建。

上一轮正式 `test_1952.sh build` 已分别从 Pro 导出开始，完成 ATC、审计、数据绑定和出包：

| 算子 | OM 大小 | OM SHA256 |
| --- | ---: | --- |
| Vector Add | 12,112 字节 | `6e1857969df82cb6cc695b36e013025decaf914b8a5bd873f13ca83338cf7624` |
| Cube GEMM | 11,460 字节 | `0ac7b08e7727734b78f60d106ee9ab690fa673702fda1159e2927b3de5e189ac` |

两份都返回 `bundle-built`，`host_portability=passed`。只有标准模型、权重、
TBE 设备 kernel 和任务分区，没有 `SO_BINS`、`CUSTOM_OPS` 或主机 ELF。
各有一个完整的 4137 设备 ELF，哈希与原生导出对象一致。
独立解码实际 `TASK_INFO` 也确认：各一个原生 kernel，入口分别为 `vector_add`、
`gemm`，blockDim 为 1；参数是输入 0、输入 1、输出 0 三个 GM 指针，共 24 字节；
没有 `.so`、workspace 或 tiling 参数。证据位于本地验证目录的
`portable-add/`、`portable-gemm/`，各自保存 `task_abi_evidence.json`。

当时的部署工具 51 项测试通过，包含 x86/ARM 主机 ELF、插件分区、旧绑定、
哈希错配和错误输入的拒绝用例；Ruff 检查通过。设备计算源码未修改，
此前 80 项计算回归作为下面的历史证据保留，本轮没有重复算作新测试。
用户随后确认：上一版 Add、GEMM 均已在内网实板测试通过。
这是用户反馈的单算子上板结论；不代表下面新增 FFN 或新执行协议已经完成实板验收。

## 本次新增 FFN 与执行流程

新增完整 SwiGLU FFN：三个原生 GEMM、一个原生向量激活节点，共四个节点放入同一 OM。
GEMM 对象被复用三次，激活由 Pro 的 `muls → exp → adds → div → mul` 生成。
网络有 `x/wgate/wup/wdown` 四个 FP16 输入，输出为 `y`；当前示例均为 16×16。
这不是单 kernel 混核。

真实 CANN 9.2 四节点 ATC 和审计已通过：每个节点均为标准
`AIcoreEngine + ELF_AICUBE`、blockDim 1，两份完整设备 ELF 均在 OM 中，主机依赖审计通过。
正式 `test_1952.sh build --op swiglu` 已完成导出、ATC、审计、golden 生成、绑定和出包。
OM 为 25,137 字节，SHA256 为
`661b4706fce56fad3b012464bb438ca4df97af8bf33447c40fc2904574b76679`。
256 个输出 golden 与独立 Torch 分阶段 FP16 参考全部一致，最大差为 0；
这验证的是编译端参考数据，不是 NPU 输出。Add、GEMM 的新格式运行包也重新构建通过。

增加性能采集前，部署与 ATC 接入层共 80 项测试通过，无跳过；其中用 mock SSH/SCP 验证了上传、远程调用、
结果回收及编译端验收，并覆盖 NaN、漏写、错包及把额外 golden/主机库放入运行包的拒绝用例。
五个新增 PTO 向量运算的真实 CANN 编译、LLVM 指令和不支持模式拒绝测试共 5 项通过。
PTO 对应提交为 `d0b13ebc381cc2fc3c9d397dfa0a3eb50bbd3ae2`。

本次板端改为 Python 3.7.9 可运行的纯执行器。运行包只携带 OM、输入、执行请求和 runner，
case、golden、binding 留在编译机。板端只输出 `status=executed` 与原始数据；
编译机负责 SSH/SCP、结果回收、两轮完整写回检查与精度比较。
只有编译机 `result.json` 的 `status=passed` 才算最终验收通过。
运行包和板端 CLI 已变化，旧包应重新生成，不能混用旧 runner 或 binding。
已用真实 Python 3.7.9 加 `-S` 跑过执行器的 19 项测试，并成功读取正式 FFN 运行包的四个输入。
这说明执行器不需要 NumPy、Torch 或额外 Python 包；FFN 实板数值仍待用户在内网验收。

## 本次新增性能采集流程

`test_1952.sh profile` 已串起上传、板端采集、原始数据回收、编译端精度比较与统计。
复用已审计的 OM 和输入，不重新 ATC；只在 staging 换入最新 runner，记录新 runner 哈希。
golden、case 和 binding 始终留在编译机。
采集使用 Fused 同款 `msprof bash app.sh`；本地使用 SDK 或 TbeWorkTestSuit 的
`msprof.py export summary -dir` 解析回收数据的副本。

真实 Python 3.7.9 的上述 19 项执行器测试覆盖预热、重复测量、调用次数、最终输出拷回和失败保留。
另有 9 个模拟 CLI 场景验证流程及失败处理；这些使用合成采集数据，**没有完成真实板端性能采集**，
不能据此给出时延或吞吐数字。完整命令见 [性能采集](TESTING.md#性能采集)。
加入 profiling 后，部署与接入层共 108 项测试通过；其中包含 20 项导出/CSV 检查，
以及实际运行生成的采集 shell、模拟传输和失败场景的回归。

每次模型执行先有两轮正确性检查，再预热、测量；每个原生节点分别去掉前 `2 + warmup` 条记录。
测量后只额外拷回一次输出做精度核对，不增加 kernel 调用。
主机整网 API 墙钟时间与设备任务微秒数分开报告，FFN 节点之和不当作端到端时延；
缺失的硬件流水列标为不可用，不补 0。结果保存 JSON、CSV、原始 `PROF_*` 和日志。
采集器、应用返回码分别检查，旧数据、错次数、精度失败或导出失败均不得产出合格的性能结论。

## 历史原生计算和编译检查（2026-11-05）

- Pro/IR/CPU/实际设备编译回归：80 项通过。
- 部署工具和 ATC 接入层测试：26 项通过。
- 新增 Python 文件通过 Ruff 0.14.14 检查；C++/Shell 检查通过。
- PTO Vector：18 项编译、边界和 LLVM 检查通过。
- PTO Cube：8 个官方 DAV5102 PV 冻结样本，每例检查 5 处布局/结果，共 40 项逐位一致。
  包括初始化、累加、非方阵、舍入和饱和；另有 14 项编译/LLVM 检查通过。

计算使用原生 Pro/PTO，没有 ops-nn 或 AscendC 计算实现。Pro 算子只有一份定义，
设备导出不要求用户另写 Host tiling 函数。

## 历史统一脚本记录（旧 OM，不再交付）

旧版 `test_1952.sh build` 曾从头走设备导出、自定义 GE/ONNX 接入、ATC、审计、case 与运行包：

| 算子 | OM 大小 | OM SHA256 |
| --- | ---: | --- |
| Vector Add | 10,632 字节 | `c4344a53744b1858c8a9f66c6fdc4830e1f5e791efa26a62e35f1855b4303457` |
| Cube GEMM | 9,964 字节 | `c37d08e4d54317508bdb1f42991599c38934d7e7902b2f12fb1c428964b6db4e` |

两项均返回 0、状态 `bundle-built`，并通过：

- OM 完整包含对应的冻结设备 ELF；不是只检查一个 ELF 文件头。
- SoC、入口、`AIcoreEngine + ELF_AICUBE` 任务描述符合本次目标。
- case、OM、设备对象、绑定和运行包文件的哈希一致。
- 每组 256 元素的独立 golden 重算一致。
- 运行包的 `runner.py` 可以在 `python -S` 下启动，不需要 PyPTO/ONNX。

这些哈希仅用于识别旧产物，**不是当前交付物或验收目标**。
当时没有主机依赖审计，以上检查不能证明跨主机部署可用。
重新构建后应核对新产物自己的绑定和审计，无需追求与历史 OM 相同的哈希。

## 尚未验证

补充的原生 Vector raw-text PV 实验没有通过：DMA identity 搬运两例逐位通过，
但包含 `TLOAD → TADD → TSTORE` 的两例完整程序及显式 flags 对照均失败。
FP16 的两个 UB 输入各 256 个元素正确，计算结果只有 128/256 个正确，GM 输出异常。
目前 raw-text PV 的 VF 启动和调度方式也未验证，原因尚未确定，不能把这项失败
直接解释为真实板端必然失败。用户反馈的 Add 实板通过，也不意味着这个 raw-text
PV 实验已经修好；两项结果分开记录。

1952 板端位于内网，本轮 FFN 未由本地直接连接板端执行。
构建结果保留 `device_execution=not_run`，不把生成 OM 当成 FFN 上板通过。
请按 [TESTING.md](TESTING.md) 执行、回收和比较，保留板端原始结果及编译机 `result.json`。

Cube 路径仍为静态二维、FP16 输入、固定 shift 42 和对应的 FP16 写回。
本轮 DecEnc 已扩展静态多维参数、FP32 Vector 和多核离线导出；
单 kernel Cube/Vector 混核、动态 shape 和硬件流水性能调优仍未完成。
FFN 实板结果待内网验收；DecEnc 优化前版本已有用户实板精度通过反馈，本次优化版精度和性能待复测。
