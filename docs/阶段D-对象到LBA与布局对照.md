# 阶段 D · 单盘对象归因与 MLA/KDA 布局对照

## 实测结果

两组各回放一次，独立开机、格式化，并通过 HTTP、原生 Store、设备计数与分配轨迹交叉校验。两组发布和恢复的 208 个键逐项匹配，包括 88 个 1,769,472 B MLA 对象和 120 个 18,723,840 B KDA 对象，载荷合计 2,402,574,336 B。

| 指标 | 原生混放 | 按类分放 |
|---|---:|---:|
| 最终 bucket 数（混合 / 仅 MLA / 仅 KDA） | 9 / 0 / 3 | 0 / 10 / 13 |
| 自然读（笔 / B） | 0 / 0 | 0 / 0 |
| 自然写（笔 / B） | 656 / 2,403,635,200 | 685 / 2,404,085,760 |
| 相邻写地址首尾连续（对数 / 全部相邻写对） | 583 / 655（89.01%） | 591 / 684（86.40%） |
| 最热 1 MiB 区域占写入字节 | 0.0436% | 0.0436% |
| 峰值写 MiB/s（1 秒桶）/ 数据 I/O 峰值在途 | 250.00 / 62 | 250.01 / 54 |
| 两种载荷共享的物理单元数（512 B 扇区 / 4 KiB 块） | 47 / 69 | 0 / 0 |
| 包含两种有效载荷的写请求（笔） | 65 | 0 |
| 写字节归因覆盖（仅有效载荷 / 全部已解释类别） | 99.95586% / 99.99966% | 99.93713% / 99.99966% |
| 窗外成功 GET 的键数 / 同期设备读取 B | 208 / 2,603,053,056 | 208 / 2,568,294,400 |

这对样本中，分放相对混放的设备写入差为 **+450,560 B**（+0.0187%）。两种布局写入的对象内容完全一致。按逐字节归因，增加的 440 KiB 中，ext4 日志占 404 KiB，文件 extent 尾部填充增加约 36 KiB，索引文件少写 4 B。分放直接改变了文件和块内的共置；提交次序、请求合并和峰值还受后台回写与 TCG 调度影响。每种布局各一次，只报告这对运行的观测值。

| 自然写入的字节组成（B） | 混放 | 分放 |
|---|---:|---:|
| MLA 载荷 | 155,713,536 | 155,713,536 |
| KDA 载荷 | 2,246,860,800 | 2,246,860,800 |
| bucket 键头 | 43,848 | 43,848 |
| Mooncake 索引文件 | 47,498 | 47,494 |
| extent 内超出 EOF 的字节 | 56,110 | 92,978 |
| ext4 journal | 704,512 | 1,118,208 |
| 已识别静态 ext4 元数据块 | 200,704 | 200,704 |
| 文件所有权时间未证实 | 0 | 0 |
| 其余未映射地址 | 8,192 | 8,192 |

读写覆盖率的分母分别为窗口内的设备读、写字节，零分母在 JSON 中记为 null。热点使用从 LBA 0 起的固定 1 MiB 区域，跨区 I/O 按字节拆分。窗外设备读包含读取放大和预读等可能开销；每键恢复正确性按 SHA256 判定。FEMU 报告内存后端未能锁页，绝对时间与峰值只代表这台 TCG 客体。本阶段统计止于主机块接口，NAND 编程、GC 搬迁和写放大留给阶段 E。

[逐项结果](../evidence/ssd-d/results.json) · [按载荷着色的写地址图](../evidence/ssd-d/layout.png) · [带宽、在途与地址时间线](../evidence/ssd-d/traffic.png) · [I/O 大小分布](../evidence/ssd-d/sizes.png) · [逐笔字节归因、对象映射和原始证据](../evidence/ssd-d/)

## 实验设计

在两次独立启动的 FEMU BlackBox 空盘上，各重放一次 C2 的 `trace_0504` 前两轮。输入为 11,133/139 和 11,300/96 个输入/输出 token，保留 4 秒 think time。K3 采用真实尺寸的缓存对象、八个逻辑 rank、固定 10/2 ms 的 prefill/decode 预算。客体 8 GiB，namespace 3 GiB，Store 段 2,520 MiB、offload 配额 2,840 MiB。两组使用同一 ext4 UUID、目录哈希种子、格式化参数和采集程序。每组各自经历真实的分配、请求合并与后台回写。[输入选择](../datasets/k3-c2/selection.json) · [K3 在线配置](../configs/k3-c1-online.json)

两组均加载同一实验适配器，使用一个 Mooncake Store 和一块 FEMU 盘。混放组原样传递原生分桶方案。分放组保留原生准入、延迟处理和 bucket 大小上限，在每个获准混合 bucket 内按 MLA、KDA 稳定分组，继续使用原生文件写入、元数据发布和 GET。两组均调用原生规划器 7 次，进入处理的混合候选 bucket 分别有 9 和 10 个。真实后台批次随运行时调度变化，配对冻结输入与预算并核对最终键集。分放发生在文件层；同一 SSD 上不同文件的 extent 仍可能相邻。[原生分桶适配器](../native/bucket_layout.cpp)

## 怎样将块请求归到对象

1. 首次 HTTP 发送至设备采集结束构成自然负载窗口。两轮之间保留所有缓存。发送队列排空且磁盘副本全部发布后，继续观察，直到设备至少连续 65 秒无活动。NVMe issue/complete 流与分段 diskstats 对账。
2. 采集结束后在客体读取 bucket 的原始键头和有效载荷。逐对象验证长度和 SHA256，用 FIEMAP 获取文件逻辑地址到设备字节地址的映射。随后通过原生 Store 清理内存副本，确认所有键仅有磁盘副本，清理客体页缓存，再 GET 到预填哨兵值的缓冲区，重新核对摘要及设备读计数。这些审计读单列。
3. 在独立 ftrace 实例记录 ext4 分配与释放，检查缓冲未覆盖。对最终 extent 的每个逻辑块，要求最后一次分配同时符合 inode、文件逻辑块号、物理块号，且早于该笔 I/O，才计入文件类别。对缺少所有权证明的字节保留未知类别。已知 journal、超级块、组描述符、块和 inode 位图及 inode 表分别根据客体的 ext4 布局识别；其余文件系统结构保守保留。
4. 对每笔设备 issue 按实际字节交集拆分。一个 512 B 扇区可部分是键头、部分是载荷。报告中“共享块”指同一物理单元同时容纳 MLA 和 KDA 载荷，“跨类请求”指同一笔设备请求涉及这两种载荷。设备 I/O 的各类字节之和必须等于设备提交字节数。[客体采证](../src/kv_ssd_sim/layout.py) · [逐字节归因](../scripts/analyze_layout.py)

## 复现

从项目根目录执行。首次构建适配器需要仓库配套的 Mooncake CPU wheel、兼容的 C++ 编译器和 `tl/expected.hpp`。为每个布局选一个新的 trial 名称，两次 VM 串行执行。

```bash
.venv/bin/python scripts/build_bucket_layout.py
for layout in mixed separate; do
  .venv/bin/python scripts/run_femu.py --trial "femu-d-${layout}-repro" \
    --bucket-layout "$layout" --workload aiperf --natural-cache --block-capture stream \
    --input datasets/k3-c2/slice.json --hybrid-config configs/k3-c1-online.json \
    --guest-ram-mib 8192 --namespace-mib 3072 --blocks-per-plane 2048 \
    --max-model-len 12288 --num-blocks 22 --timeout 2400 || break
done
.venv-aiperf/bin/python scripts/analyze_layout.py outputs/femu-d-mixed-repro \
  outputs/femu-d-separate-repro --output outputs/d-reanalysis
.venv-aiperf/bin/python scripts/audit_layout_concurrency.py \
  evidence/ssd-d/trace_0004.source.json.gz --output outputs/d-concurrency.json
```

两份归档已在新目录解包复核，汇总 JSON 与四份 CSV 解压内容逐字节一致。项目测试 89 项通过、6 项跳过。要从归档复核，执行下列命令。两份压缩包内含块轨迹、分配/释放轨迹、磁盘布局、每键摘要、AIPerf 原始请求和 Store 事件；设备盘本体位于各次运行的临时 FEMU 内存后端。

```bash
(cd evidence/ssd-d && sha256sum -c SHA256SUMS)
d=$(mktemp -d)
for trial in evidence/ssd-d/femu-d-*.tar.gz; do tar -xzf "$trial" -C "$d"; done
.venv-aiperf/bin/python scripts/analyze_layout.py "$d/femu-d-mixed01" \
  "$d/femu-d-separate01" --output "$d/result"
cmp "$d/result/results.json" evidence/ssd-d/results.json
```

## 独立会话的并发证据

固定版本的真实 `trace_0004` 含 182 条外层普通请求和两个完成的 Explore 子 agent，分别发出 14 和 19 次调用。按 AIPerf 0.13.0 的时间坐标规则还原子请求区间，以原始 `api_time` 计算半开区间的交集。两个子 agent 的生命周期重叠 **55.729 秒**，其中 **17.832 秒**两个子 agent 同时有 API 调用在途。用项目的本地 tokenizer 对完整源文件做 AIPerf 离线转换，按源索引核对全部 215 次调用各出现一次，两个显式子会话均被同一个 SPAWN 分支引用。两者的首个输入分别为 18,251 和 18,259 tokens，均超出本次 12,288-token 配置。此项是源会话并发审计；设备布局对照使用串行的 `trace_0504`，其设备结果只覆盖这两轮。[固定源会话](https://github.com/callanjfox/kv-cache-tester/blob/94f6046dfd9c0c8bc0cdb24a5aa6579bac8669bf/traces/trace_0004.json) · [逐次时间与计算结果](../evidence/ssd-d/concurrency.json)
