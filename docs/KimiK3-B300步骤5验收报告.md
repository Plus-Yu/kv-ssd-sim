# 步骤 5 · 自然容量压力与混合缓存生命周期

2026-11-18，**通过**。在关闭人工清缓存的 FEMU 客体中，完整前缀从仅磁盘副本恢复，缺少 KDA 的前缀触发全量 prefill。两个兄弟分支的调度和保存均由原生 vLLM、Mooncake 驱动。结论覆盖单进程 TP1 混合功能夹具。AIPerf 真实切片、K3 真实尺寸 TP8 在线运行及 B300 性能估计仍属后续工作。[可离线复核的验收结果](../evidence/kimi-k3-step5/femu-step5-natural03/verification.json)

| FEMU 观测项 | 结果 |
|---|---|
| 请求与保存 | 22/22 请求完成，46 个原生保存作业；9 对共同调度的子请求存在实际写入的保存作业重叠 |
| 自然容量压力 | 4 个越过 4.5% 水位且驱逐数大于零的周期；触发日志中最高内存占用比例为 12.75% |
| 完整前缀 | 提交时有内存副本，恢复时 2 个 MLA 对象及 1 个 KDA 对象均只有 LOCAL_DISK；128 tokens 命中，逐对象摘要及长度匹配 |
| 设备读 | GET 读取有效内容 264,768 字节，同窗口的 2 次 FEMU 块读完成共 278,528 字节（272 KiB）。块跟踪的扇区数与来宾 NVMe 计数一致 |
| 缺片 | 一份 KDA 尚未提交，同 hash 的 MLA 仍存在；外部加载 0、调度全部 128 个输入 tokens，随后成功首次写入该 KDA |
| 最终层级 | 142 个不同键中，121 个仅磁盘，16 个磁盘和内存均有副本，5 个仅内存。要求保留的 4 个种子对象均有磁盘副本 |

## 配置与对照

本次通过原生 replay 直接驱动 vLLM Scheduler、KVCacheManager、MooncakeStoreConnector 和 embedded Mooncake。合成 WEKA 夹具包含 22 个请求。两个子 agent 各自连续运行 9 轮，后续父轮次等待两支完成。首个生产请求保存 128-token 共享前缀，同一前缀随后出现在一个 192-token 请求中。另一个生产请求提供 64-token 前缀，用于独立的缺片对照。[工作负载](../examples/pressure.weka.json)

采用 TP1 的一层 MLA、一层 KDA，原生块池 24 块，固定 prefill/decode 每 step 10/2 ms。Mooncake 内存段 32 MiB、staging 16 MiB、磁盘预算 128 MiB，FEMU 客体可见单盘 384 MiB。4.5% 内存高水位对应约 1.44 MiB，单周期目标驱逐比例设为 85%。这个低水位使小工作集也能触发原生驱逐，不能据此推算生产服务器的命中率。[混合规格](../configs/pressure-hybrid.json) · [容量与实验角色](../configs/pressure-store.json)

- **自然淘汰。** 只靠请求分配 vLLM 块和提交 Store 对象制造压力。没有主动清除上层副本、调用前缀重置、清客体页缓存或注入接收等待。FEMU 内启用原生 io_uring 直接读，验收同时核对命名空间扇区计数及 GET 窗口内配对的块读完成。
- **并发保存。** 按请求和原生 store job ID 关联入队、处理、结束和真实 PUT。在共同调度的两个子分支中，核验两份实际写入过对象的保存作业生命周期重叠。TP1 只有一个原生发送线程，队列允许多个作业未完成。
- **缺片重算。** 仅对指定生产者，在发送前省去其 KDA checkpoint 提交，正常写入对应的 MLA。对后续探针核对同一前缀哈希的 MLA 存在、KDA 无副本、零外部加载、全部 prompt tokens 被调度。重算后第一次成功发布该 KDA；此项只验证重算和发布，未对新 checkpoint 发起第三次请求。该路径没有删除旧键；旧键删除再创建仍按[既有排查](Mooncake同键重落盘排查.md)排除。
- **结束屏障。** 等待指定生产者需要保留的键有磁盘副本，另存所有历史成功 PUT 对象的最终层级。允许普通压力对象处于内存、磁盘或已经淘汰的状态。

## 宿主机同负载对照

高、低水位均采用 500 ms 读取租约，输入、计算、缓存规格、软件和写出的 142 个对象逐键及摘要一致。90% 水位组没有成功驱逐周期，恢复时三个目标对象均有内存副本；4.5% 水位组发生 5 次成功驱逐，三个目标对象均仅有 LOCAL_DISK。两组 GET 内容校验均通过。宿主数据用于验证水位与副本状态的关系，设备读的判据取自独立 FEMU 客体。[宿主对照记录](../evidence/kimi-k3-step5/host-comparison.json)

开发时第一版 FEMU 夹具观察到设备读与重算，但子分支 PUT 的日志不足以判定保存作业重叠。随后加入原生 job 记录，并让两支的首轮 64 tokens 同时容纳在一个 128-token step 中。另一轮宿主高水位对照发现，旧冷读配置的零毫秒租约会使有内存副本的 GET 返回 -707，日志记录了数据传输完成前租约过期。最终对照与 FEMU 验收均使用显式的 500 ms 租约。[原失败日志](../evidence/kimi-k3-step5/historical-zero-lease-host.log)

## 运行与离线复核

从项目根目录运行。FEMU 使用新的 trial 名；此实验配置将四个实验角色绑定到夹具中的指定请求。

```bash
.venv/bin/python scripts/run_femu.py --trial femu-pressure-repeat \
  --input examples/pressure.weka.json --hybrid-config configs/pressure-hybrid.json \
  --pressure-config configs/pressure-store.json --num-blocks 24 --max-model-len 1024 \
  --nand-read-ns 40000
.venv/bin/python scripts/verify_femu.py outputs/femu-pressure-repeat
```

仅复核归档时无需启动客体。验证器从日志重建调度依赖、缺片处置、保存作业并发、驱逐与设备读因果顺序。

```bash
(cd evidence/kimi-k3-step5 && sha256sum -c --quiet SHA256SUMS)
.venv/bin/python scripts/verify_femu.py evidence/kimi-k3-step5/femu-step5-natural03
.venv/bin/python scripts/check_pressure.py evidence/kimi-k3-step5/step5-host-control02 \
  evidence/kimi-k3-step5/step5-host-low02
```

完整原生回归测试为 80 项通过，包含 5 项针对错误证据的反例测试。步骤 4 归档重算与原配对 JSON 逐字节一致，步骤 3 的 36 个分片对象也已重新通过离线核对。三次 FEMU 开发运行均保留在本机 `outputs/femu-step5-natural01` 至 `natural03`；最终归档对应第三次。
