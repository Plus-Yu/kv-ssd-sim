# 步骤 4 · FEMU 输入透传与冷读计时

**通过。** 在第二个请求的计算预算同为 16 ms 时，将 NAND 页读从 40 µs 改为 100 ms，AIPerf 原始首 token 时延增加 402.75 ms。两组均从 FEMU 恢复 256 个前缀 tokens，MLA/KDA 摘要全部匹配。两组写出的 10 个不同对象逐一核对了键、字节数及内容摘要。[配对结果](../../evidence/kimi-k3-step4/comparison.json)

| 第二个请求的测量项 | 快读（ms） | 慢读（ms） |
|---|---:|---:|
| 冷准备，独立计时 | 451.61 | 928.53 |
| 冷准备结束至客户端开始计时 | 10.10 | 10.62 |
| FEMU 块读平均完成时延 | 0.235 | 100.206 |
| Mooncake GET | 162.12 | 565.79 |
| vLLM 原生请求 TTFT | 203.01 | 606.71 |
| AIPerf 原始客户端 TTFT | 252.13 | 654.88 |

每个 GET 窗口均记录到 5 次块读完成，共 49,152 字节。两组计算实际耗时分别为 23.92 和 26.06 ms。冷准备本身相差约 477 ms，但两组都在准备完成后才开始请求计时；这次客户端观测到的增量与 GET 增量（403.67 ms）相差不到 1 ms。

## 实验与判据

2026-11-18。使用 AIPerf 0.13.0 依次发送两个 WEKA 合成请求，输入分别为 256、320 tokens，各输出 4 tokens，共享前 256 tokens。每个 trial 独立启动一个 TCG 客体（2 vCPU、4 GiB 内存、384 MiB FEMU NVMe），运行 TP1 的 1 层 MLA＋1 层 KDA。块池特意设为 96，以便在客体运行清单里检验非默认参数。prefill/decode 每 step 的预算分别为 10/2 ms，接收端延迟注入为 0。

宿主将 trace、混合几何和运行参数写成 trial 私有快照。客体核验三份 SHA256，记录实际命令。验收再核对 AIPerf 消费的输入摘要、服务端生效的几何、块数、上下文及计算预算。快慢组复核使用同一套内核、initramfs、FEMU、软件和输入，对照变量为 NAND 页读时延。

第二轮的冷读按以下顺序测量。所有时间戳使用同一客体的单调时钟，客户端与服务端通过请求 ID 关联。

1. 第一轮 HTTP 响应完成。AIPerf worker 在发送第二轮前，通过独立的 HTTP 请求等待服务端准备。
2. 服务端等待待处理传输，确认磁盘副本，驱逐 Mooncake 内存副本，重置 vLLM 前缀缓存，清除客体页缓存，再回送确认。
3. 客户端收到确认，记录发送标记，调用 AIPerf 原生发送方法。以原始记录中首个含 token 的响应时间减去发送起点，独立核对导出的 TTFT。
4. 以块层 issue/complete 配对验证 GET 窗口中的 FEMU 读。原生 Store 完成并校验恢复内容后，调度器继续执行并发送首 token。

人工准备会延长两轮之间的间隔。第二轮 HTTP 传输、服务端构造占位数据、排队和调度的耗时都计入客户端 TTFT。此前把冷准备放在收到请求之后，客户端 TTFT 的慢减快为 −88.02 ms。原文件 `outputs/femu-aiperf-comparison.json` 的 SHA256 为 `0acb36034b0ab7e13ca2ac8c736247045544089f7387e54924c70397877b225f`。[历史联合验收报告](../01-原型与闭环接入/AIPerf-FEMU联合验收报告.md)

## 复现

在项目根目录运行，每次使用新的 trial 名。两组顺序启动，避免两个客体争用同一宿主。

```bash
args=(--workload aiperf --input examples/aiperf-two-requests.weka.json
      --hybrid-config configs/hybrid-tiny.json --num-blocks 96
      --max-model-len 8192 --prefill-ms 10 --decode-ms 2)
.venv/bin/python scripts/run_femu.py --trial femu-replica-fast --nand-read-ns 40000 "${args[@]}"
.venv/bin/python scripts/run_femu.py --trial femu-replica-slow --nand-read-ns 100000000 "${args[@]}"
.venv/bin/python scripts/compare_femu.py outputs/femu-replica-fast outputs/femu-replica-slow \
  --output outputs/femu-replica-comparison.json
```

本轮验收覆盖串行、有界、强制冷读的功能对照。步骤 3 的 K3 单 checkpoint 唯一对象共 470,605,824 字节，已超过本客体 384 MiB（402,653,184 字节）的可见盘容量。真实 trace 切片的 71,168-token 共享前缀也超过本轮 8,192 的上下文配置。开展对应工作集的实验还需要适配客体资源与执行器。TCG、宿主调度以及 FEMU backend 未锁页均可能影响绝对时延；单次配对只能检验变化方向。

## 离线复核

归档中保留了输入快照、客体回执、启动与运行清单、客户端原始记录、服务端事件和未裁剪的块日志。使用现有虚拟环境可直接重做配对，无需启动虚拟机。

```bash
(cd evidence/kimi-k3-step4 && sha256sum -c SHA256SUMS)
.venv/bin/python scripts/compare_femu.py evidence/kimi-k3-step4/femu-step4-fast01 \
  evidence/kimi-k3-step4/femu-step4-slow01 --output outputs/step4-offline-check.json
cmp outputs/step4-offline-check.json evidence/kimi-k3-step4/comparison.json
```

步骤 1 的输入审计（含 AIPerf 离线核对）、步骤 2 的原生配置对照、步骤 3 的 36 对象分片证据均已重新通过校验。启用原生集成的完整测试结果为 75 项通过。下一步按计划验证自然容量压力下的淘汰、SSD 回读和缺片重算。
