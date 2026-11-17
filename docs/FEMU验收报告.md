# FEMU 在线闭环验收报告

**已在本机无 GPU、无 KVM 的条件下，完成混合 MLA＋KDA → 原生 vLLM 调度 → 原生 Mooncake 磁盘 offload → FEMU BlackBox NVMe 的在线闭环。** 实际设备读完成会影响 Mooncake GET 和正在运行的 vLLM 请求恢复。计算仍为固定预算的占位执行。

本报告依据最终通过的两组实验。之前的 `fast-01` 至 `fast-04` 是启动诊断或失败运行，不纳入结果。

## 环境和边界

| 项目 | 设置 |
|---|---|
| FEMU | 提交 `a898d22b2f894042afe0f4ef21867393c5c78a70`，项目内构建 |
| 虚拟化 | QEMU TCG，2 vCPU，4 GiB guest RAM；未使用 KVM |
| 客体内核 | Alpine virt `6.18.60-0-virt` |
| 设备 / 文件系统 | `FEMU BlackBox-SSD Controller`，`/dev/nvme0n1` → `/mnt/femu`，ext4 / noatime |
| NAND | 512 MiB 物理容量、384 MiB 可见容量；2 channels × 4 LUNs |
| 几何 | 512 B sector，8 sectors/page，64 pages/block，256 blocks/plane，1 plane/LUN |
| 固定介质参数 | 页写 200 μs、擦除 2 ms、channel transfer 参数 0 |
| vLLM | `0.34.0rc1.dev302+gbe47189f3.cpu`，原生 Scheduler / KVCacheManager / MooncakeStoreConnector |
| Mooncake | non-CUDA `0.3.14`；master、embedded real client 均在客体内 |
| 混合配置 | `configs/hybrid-tiny.json`，1 MLA 层＋1 KDA 层，TP=1 |
| 计算预算 | prefill step 10 ms，decode step 2 ms；不执行模型 forward |

虚拟机仅连接本次新建的 RAM-backed FEMU SSD，没有连接宿主物理磁盘，没有开放客体网络。宿主 `/usr` 与 conda 库只读导出，项目通过 9p 共享。Mooncake 实际写入的是解析后的 `/mnt/femu/replay`，不是 9p 目录；原生 Store 不接受 offload 路径中的符号链接，适配器已据此修正。

## 验收方法

使用 `examples/two_requests.weka.json` 的小型合成 WEKA 格式验收夹具，并非公开 Claude Code 长 trace 的全量回放。两条请求分别输入 256 / 320 tokens，输出均为 4 tokens，共享前 256 tokens。第一条请求完成后：

1. 等待 native Store 确认磁盘副本，驱逐 Store 内存副本。
2. 重置原生 vLLM 本地 prefix cache。
3. 在一次性客体内执行 sync 和 page-cache eviction。
4. 第二条请求经原生 hybrid lookup 命中并恢复 MLA＋KDA 状态。
5. 采集同一 monotonic 时钟下的块请求 issue/complete、Store GET、`load_complete`、计算和首 token 事件。

`--read-delay-ms=0`，没有人为 receiver sleep。两组只改变 FEMU 的 `pg_rd_lat`。比较脚本检查输入 SHA256、harness 源码 SHA256、包版本、混合配置、调度参数、计算预算、QEMU 命令和启动资产 SHA256，验证除 trial 名和 NAND 读参数外一致；10 个写出对象的键、尺寸和完整内容摘要也一致。

## 结果

| 指标 | 快读组 | 慢读组 |
|---|---:|---:|
| NAND 页读参数 | 40 μs | 100 ms |
| 恢复的共享前缀 | 256 tokens | 256 tokens |
| 恢复对象 | 4 MLA 块＋1 KDA 快照 | 相同 |
| 逻辑恢复字节 | 35,392 B | 35,392 B |
| GET 窗口内实际 NVMe 读取 | 5 次，49,152 B | 相同 |
| 块读 issue → complete 平均时延 | 0.107 ms | 100.259 ms |
| Mooncake GET 窗口 | 177.640 ms | 566.307 ms |
| 第二条请求 TTFT | 218.000 ms | 608.241 ms |
| 第二条请求计算预算总和 | 16.000 ms | 16.000 ms |
| 第二条请求实际计算窗口总和 | 25.349 ms | 26.221 ms |
| 对象 SHA256 / 输出长度校验 | 通过 | 通过 |

增量：块读平均完成时延增加 **100.152 ms**，GET 增加 **388.667 ms**，TTFT 增加 **390.241 ms**。完成时序验证均通过：

```text
native_get_begin
  → FEMU namespace 的真实块读 issue / complete
  → store_get_end（MLA 与 KDA 校验通过）
  → load_complete
  → compute_begin
  → first_token
```

两组应用事件序号均为 `50 → 52 → 53 → 54 → 59`；对应块完成时间落在 GET 窗口内。块 trace 没有 buffer overflow，所有用于统计的读完成均有匹配的 issue。真实块读取量大于逻辑对象量，包含文件系统对齐 / readahead 等影响，不等同于对象净字节数。

这说明 SSD 模拟器的完成时延确实反馈到在线推理调度链。不能把块时延、GET 增量和 TTFT 增量简单按次数线性相加。

## 证据与复现

- [快读组验证](../outputs/femu-tcg-fast-05/verification.json)
- [慢读组验证](../outputs/femu-tcg-slow-01/verification.json)
- [严格配对比较](../outputs/femu-comparison.json)
- [快读块 trace](../outputs/femu-tcg-fast-05/block-trace.txt)、[慢读块 trace](../outputs/femu-tcg-slow-01/block-trace.txt)
- 每组 `launch.json`、`guest-mounts.txt`、`replay/run_manifest.json`、`replay/events.jsonl`、`replay/summary.json` 和 `disk-export/` 保留原始证据。客体关闭后，原 `replay/disk` 符号链接不可在宿主直接访问，导出文件位于 `disk-export/`。

```bash
.venv/bin/python scripts/run_femu.py --trial femu-my-fast --nand-read-ns 40000
.venv/bin/python scripts/run_femu.py --trial femu-my-slow --nand-read-ns 100000000
.venv/bin/python scripts/compare_femu.py outputs/femu-my-fast outputs/femu-my-slow \
  --output outputs/femu-my-comparison.json
```

构建和客体准备方法见 [FEMU 接入与复现](FEMU-next.md)。当前每组启动仍需数分钟，启动时间不计入请求 TTFT。已在宿主预计算同一 venv 的包映射，客体核对分发元数据后使用；模型信息预热采用 vLLM 原生、按源码 hash 校验的缓存。两者只减少启动发现工作，不代替调度、分配或存储操作。`Timeout (0:02:00)!` 是主动启用的周期栈诊断输出，最终成功与否由 guest exit code、summary 和 verification 判断。

## 限制与下一步

最终回归：**30 项测试通过**，Ruff 和 shell 语法检查通过，doctor 的原生库导入检查通过并识别本地 FEMU 二进制。证据见 [回归日志](../outputs/final-regression.log) 和 [环境检查](../outputs/final-doctor.json)。

这是 **每组一次的功能对照**，不是统计意义的性能评估。100 ms 是用于放大因果信号的干预参数，不是对真实 NAND 的时延标定。TCG 执行开销明显：16 ms 预算实际约 25–26 ms；FEMU 在本机受 memlock 限额影响不能锁住后端内存，启动日志已记录可能的 page-fault 抖动。GET 的软件开销远高于快读组的介质时间，不能将绝对 TTFT 当成真实 GPU 系统预测。

当前使用 tiny 状态、少量请求和低 SSD 占用，没有覆盖长期 GC、写放大、耐久性、生产并发、跨 TP、GPU/DMA/RDMA 或实际模型精度。尚未实现逐对象到 bucket 文件偏移再到 LBA 的一一映射；当前证据关联到 GET 批次。混合状态的其他验证及同键删除后重落盘的 Mooncake 限制见 [混合注意力验收报告](混合注意力验收报告.md)。

下一步可在支持 KVM 的环境保持这套验收路径，增加重复实验、实际目标模型状态尺寸和工作集压力，再比较 MLA/KDA 对象布局。AIPerf 回放服务接入仍按用户要求保留为 [待改进项](待改进项.md)。
