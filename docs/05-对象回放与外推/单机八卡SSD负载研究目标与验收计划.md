# 单机八卡 SSD 负载研究目标与验收计划

2026-11-25 记录。**本阶段研究一台八卡服务器运行一个 Kimi K3 TP8 实例时，分配到这台机器的多轮 agent 会话向整机缓存卷提交哪些读写。** 主实验以一块大容量逻辑 SSD 记录整机对象负载，另保留 3 TB 单盘对照。连续会话间保留本地 KV、Mooncake 内存和已发布的磁盘副本，记录对象大小、提交时刻、并发及最后一个请求结束后的后台写入。单盘块 I/O、NAND 编程、GC 搬迁和擦除要在明确映射和前序盘态的设备实验中测量。[大容量单盘与分阶段恢复](大容量单盘与两阶段KV恢复验收.md)

现有回放器使用 vLLM 调度与 K3 分片规则生成对象键和尺寸，虚拟化 KV 数据与容量，由同版本 Mooncake 执行分配和淘汰。n15 的 kvlab 留出窗口三次均与原生文件后端对上 39 个写对象、475,914,240 B，三次自然文件读均为 0。受控清缓存用例另行对上了 4 次文件读、131,072 B。两组证据分别检验了短窗口的写入预测和读观测器的采集能力。[首轮验收报告](kvlab首轮窗口与SSD对象负载验收.md) · [受控读自检](../../evidence/kvlab-first-windows-20261124/controlled-read.json)

## 五项缺口与优先级

对象回放的 `reference` 采用 24 TB KV 配额的大容量单盘和 1M 单请求上限。此前单盘配置的资料来源与两种上下文的全库覆盖率可单独核查。[当前主实验](大容量单盘与两阶段KV恢复验收.md) · [单盘参考与上下文覆盖](单机八卡公开参考配置与上下文覆盖.md)

| 优先级 | 要回答的问题 | 现有证据与下一项验收 |
|---|---|---|
| 1．自然回读 | 旧前缀回来时，究竟哪些键要从文件恢复？ | 35 秒返回用例已有磁盘副本正例和逐键差异。先用当前观测器定位差异，再用独立会话检验误报和漏报。[历史压力用例](../../evidence/object-replay-put-timing-20261123/summary.json) |
| 2．八卡缓存拓扑 | 哪个客户端拥有对象，各客户端何时向卷提交读写？ | 主配置为 2,781 块/rank、8 个逻辑客户端，各有 128 GB Store、3 TB 独立盘配额，同一卷提供服务。n15 在线实验的八个逻辑 rank 共用 1 个客户端。实际客户端的放置、分桶、目标机目录及条带映射仍需核对。[参考设置](../../configs/kimi-k3-b300-large-ssd.json) |
| 3．长期会话与缓存前史 | 旧前缀再次使用前，这台机器还处理了多少其他数据？ | 同一份 kvlab 数据，在 131,072 和 1,048,576 两种单请求上限下，前序完整部分分别覆盖全库输入 token 的 2.83% 和 66.24%。长期样本采用后者，固定窗口对照保留前者。连续时段需记录会话路由、源端和目标端并发、预热、工具等待及竞争请求。[全库覆盖审计](../../evidence/tp8-public-20261125/audit.json) |
| 4．计算与存储节奏 | 一批对象何时可保存，何时排队，何时有可读的磁盘副本？ | 按新算 token 和批量估计 step，按新分配字节估计 PUT，单盘双向队列默认 1×，另测 4×/8×。盘读后按目标 rank 的端口和文件所属客户端的暂存预算推进至恢复完成。历史 10/2 ms 工作点供复现。目标机仍需采集批大小、上下文、实际 step、文件提交及完成时刻，以校准突发和后台排空时间。[参考时钟配置](../../configs/kimi-k3-b300-large-ssd.json) · [n15 计时拆分](../../evidence/kvlab-first-windows-20261124/calibration.json) |
| 5．持续盘态与 NAND/GC | 连续写入、替换旧对象后，主机写带来多少 NAND 编程和 GC 搬迁？ | 3 GiB FEMU 的 `0504` 窗口记录了零对象 GET 和 8,192 B 块读。56 GiB 大盘的一致快照与容量验收仍待完成。用已核实的对象负载选取持续替换段，在同一实例保留文件、填充度和 FTL 状态，按同一窗口核对块读写、NAND 编程、GC 搬迁、擦除及写放大。[设备基线与大盘待验项](../04-n15迁移与容量验收/n15-KVM基线与56GiB容量验收.md) |

## 35 秒返回用例的逐键回读验收

历史用例在 n15 宿主文件系统上运行，Store 内存段为 1,450 MiB，租约为 5 秒。两端产生相同的 120 次 rank 级 GET。原生端有 **12 次**在 GET 发起时只剩磁盘副本，对应 **89,051,136 B**。旧回放器在每 rank 保存时间设为 1 ms 时预测 **37 次、150,242,304 B**。逐次按键配对，两边只有 **1 次**同时判为只剩磁盘副本。这里的分母是 Store GET 次数。设备层的读字节占比另由同一窗口的块读字节除以块读写总字节计算。[历史参数与逐键匹配统计](../../evidence/object-replay-put-timing-20261123/summary.json) · [原生淘汰验收记录](对象回放器-Mooncake原生淘汰.md)

1. **复现已知差异。** 固定原输入、35 秒等待、初始缓存、容量、租约和版本，先保存当前回放器的逐键预测，再运行带观测器的原生 Mooncake。历史数字用作诊断起点，本轮结果单独记录。
2. **找出副本状态首次分歧的时刻。** 对同一请求、rank 和键，对齐 PUT 分配、内存就绪、bucket 写入、磁盘副本确认、内存副本回收和 GET。用文件后端的读提交与完成确认实际文件读。只读采样若只能定位“最后一次看到内存副本”和“第一次看到它消失”，保留该时间区间。[文件后端观测器](../../native/object_replay_observer.cpp)
3. **在独立会话上核验。** 用 35 秒用例定位和校准后冻结参数，另选未参与调参、前序可还原的会话。逐请求、rank、键、字节数和次数列出读盘预测命中、误报与漏报，同时保留写入对账及后台尾部。留出窗口如实测零读，记录零值，自然回读召回率继续标为待验证。

这轮的交付物是一份单机逐键回读对账报告，附预测和实测的事件路径、差异键清单及对应字节。n15 小容量实验用于验证回读机制，八卡参考容量回放用于估计指定会话的对象负载，FEMU 窗口用于测量给定盘态和 FTL 实现下的设备行为。要把写放大或 GC 频率推广到具体 SSD 型号，还需在目标盘上用相近填充度和写入历史取得设备计数作对照。

## 五项缺口可以借用哪些公开证据

截至 2026-11-25，公开资料覆盖了同版本软件的行为、同型号八卡机器的部分实测，以及真实 agent trace 的请求历史。本次检索仍缺一组把单机 B300、K3、Mooncake 逐键事件、单盘块 I/O 和 NAND 计数按同一时钟对齐的公开记录。

1. **自然回读。** 项目锁定的 Mooncake 0.3.14 源码能核对租约如何约束内存回收、文件写完后何时向 master 确认磁盘副本。FlatCake 预印本的图 7 还给出了分层计量示例。作者在单机八张 B200、八盘 RAID 0 上，用 SGLang、Mooncake 0.3.13 和 DeepSeek-V4-Flash 回放 Claude Code 流量，分别统计由显存、主存和 SSD 实际交付的前缀。可借用这一口径，在 35 秒用例里给同一次 GET 分别记录副本可用、文件读完成和恢复成功。具体命中的键仍由 n15 的原生记录检验。[固定版本的回收逻辑](https://github.com/kvcache-ai/Mooncake/blob/e1ea5d31bb5e46ff2df9f90e7c2fbaf52fcdee8c/mooncake-store/src/master_service.cpp) · [固定版本的落盘回执](https://raw.githubusercontent.com/kvcache-ai/Mooncake/e1ea5d31bb5e46ff2df9f90e7c2fbaf52fcdee8c/mooncake-store/src/file_storage.cpp) · [FlatCake 原文第 5 节](https://arxiv.org/html/2610.38355v1)
2. **八卡拓扑和容量。** vLLM 0.34.0 的连接器指南说明，`embedded` 模式下每个 rank 可贡献自己的内存段，`standalone-store` 模式下各 rank 发请求，由外部 Store 进程持有内存和 SSD。两种部署应分别记录请求进程、内存段、写盘进程和目录。AWS Samples 的单节点 K3 TP8 实测在显存利用率 0.95、FP8 KV、最大上下文 1,048,576 时，每卡实际分配 **55.0 GiB KV**。项目的 1,876 块约为 **37.1 GiB/rank**。55 GiB 可作本地 KV 容量敏感性对照，目标机的实际块数、8 × 128 GB Store 和 3 TB 单盘配额仍须用相应部署的日志核对。[vLLM 连接器部署指南](https://docs.vllm.ai/en/v0.34.0/features/mooncake_store_connector_usage/) · [AWS 八卡 B300 实测，第 1–2 节](https://github.com/aws-samples/sample-llm-inference-on-eks/blob/main/docs/KIMI-K3-B300-PERFORMANCE.md) · [项目容量账本](../02-K3参考部署与验证/KimiK3-B300步骤2验收报告.md)
3. **长期会话与前史。** WEKA 的 `replay-v5` 保留请求时序、前序引用和脱敏内容指纹。项目锁定的 40 天文件有 **3,190 个会话、370,658 个请求**。可以重建源端请求重叠、工具等待期间的竞争流量和前缀复用间隔。将哪些会话分给这台 TP8 实例仍是路由假设。在项目 131,072-token 上限下，前序完整的可回放请求只覆盖原始输入 token 的 **2.83%**，推断长期负载时要报告这个分母。Mooncake FAST ’25 的工具与 agent trace 另有一小时线上到达时刻和 512-token 块哈希，可用于独立对照前缀复用分布。[WEKA 回放格式](https://github.com/weka/kvlab/blob/main/docs/replay.md) · [40 天文件及覆盖率审计](../../evidence/kvlab-first-windows-20261124/audit.json) · [Mooncake FAST ’25 数据说明](https://github.com/kvcache-ai/Mooncake/blob/main/FAST25-release/README.md)
4. **计算和落盘节奏。** AWS 同一台八卡 B300 在 1,024 输入、1,024 输出、并发 1 时测得平均首 token 延迟 **241 ms**、平均相邻输出 token 间隔 **9.45 ms**，报告注明测量窗口含升压阶段。这组客户端指标可做同形请求的端到端对照。要校准当前每步 10/2 ms，应按实际批大小和上下文采 GPU 时间线，再与 Mooncake 的 PUT 分配、内存就绪、写盘提交和完成时刻对应。[AWS 测量方法与结果，第 4–5 节](https://github.com/aws-samples/sample-llm-inference-on-eks/blob/main/docs/KIMI-K3-B300-PERFORMANCE.md) · [vLLM 的 Nsight Systems 采集方法](https://docs.vllm.ai/en/stable/contributing/profiling/)
5. **持续盘态与 NAND/GC。** SNIA 的 SSS PTS 2.0.2 给出预处理、面向工作负载的预热及稳态测量窗口的方法。OCP Datacenter NVMe SSD 2.7 在扩展 SMART 日志 C0h 中定义了 `Physical Media Units Written`，包含用户数据和元数据。在支持这一计数器的盘上，将同窗介质写增量除以主机写字节增量，即可对照 FEMU 的写放大。GC 搬迁和擦除时序需要更细的设备遥测或 FEMU 的 FTL 事件。SSD-iq 论文给出了不同实盘在相同写入分布下的 WAF 测量，适合对照盘型与初始状态的影响。[SNIA 预处理和稳态规范](https://www.snia.org/sites/default/files/technical-work/pts/release/SNIA-SSS-PTS-2.0.2.pdf) · [OCP 2.7 扩展 SMART 定义](https://www.opencompute.org/documents/datacenter-nvme-ssd-specification-v2-7-final-pdf) · [SSD-iq 实盘实验](https://www.vldb.org/pvldb/vol18/p4295-haas.pdf)

2026-11-23 的 vLLM 实验同时使用了 **K3 和同一份 kvlab**。它用两台各八张 MI355X 的节点分别做 prefill 和 decode，采用 MoRI UMBP，每节点有两块本地 NVMe。70 个并行会话、1.8 TB NVMe 池这一组的 **91.89%** 指由 GPU 或缓存池供应的输入 token 占比。另一个 70 会话、盘前有 1.8 TB DRAM 的实验中，丢弃已被替代的 KDA 检查点后，作者报告的 NVMe 写入速率从 **2.26 降到 1.05 TB/h**。前一项适合对照前缀复用，后一项说明有必要按 KDA 检查点和 MLA 对象分别统计写字节。单块 SSD 的读字节占比仍由块层读写计数计算。[vLLM 与 MoRI 原始实验及附录](https://vllm.ai/blog/2026-11-23-skip-the-dram-tier)
