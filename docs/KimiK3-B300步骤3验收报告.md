# Kimi K3＋B300：真实尺寸分片存取验收

2026-11-18。一个位置 1,536 的 checkpoint 已通过单 rank 和全部 TP8 分片验收。只缺一份 KDA 对象时，vLLM 的 Mooncake worker 返回 0 个可复用 tokens；补齐后从纯磁盘副本返回 1,536。所有 93 层按步骤 2 的真实字节尺寸处理。

| 试验阶段 | 实际对象 | 原生 lookup | 验证 |
|---|---:|---:|---|
| rank 0 | 12 个 MLA chunk＋该 rank 的 3 个 KDA 组 | 0 | 等待落盘，清内存副本，15 次 GET 全量校验 |
| 只缺组 2、rank 7 | 35/36 个不同 key | 0 | 已写 35 个 key 均经 Store 查询存在；最后一片故意尚未提交 |
| 补齐并清内存副本 | 36/36 个不同 key | 1,536 | 8 个逻辑 rank 全部恢复，120 次 GET 全量校验 |

**如何生成数据。** 从固定的步骤 2 配置重建原生 spec、组和分配计划。向原生 KVCacheManager 申请一个 1,537-token 合成请求的块；原生 checkpoint 函数返回位置 1,536、第 0 列。分配器给三个 KDA 组的块表为 [1,2]、[3,4]、[5,6]，MLA 为 [7,8]。测试只写已分配的 checkpoint 列和首个 MLA 管理块，不把最后一个 token 算入可复用前缀。

**传输范围。** 复用一个 191,102,976 字节的已注册 CPU 缓冲区，保留原生页宽、组内层序与真实块 ID。8 个逻辑 rank 逐个建立原生 Store layout 和 key；每次 GET 前先用哨兵覆盖目标，GET 后同时核对对象 SHA256、逐层期望内容和 KDA 页尾未写区域。测试数据按层、逻辑分片和前缀 hash 独立生成；MLA 副本预期一致，KDA 各分片预期不同。

- MLA：24 层共享一个 Store 分片；12 个 128-token chunk，各 1,769,472 字节。每个 chunk 写一次，八个 rank 均实际读回。
- KDA：3 个组 × 8 个分片，共 24 个不同对象，各 18,723,840 字节。每个目标 rank 实际读回自己的三组状态。
- 合计唯一对象内容 470,605,824 字节。完整恢复阶段读回 619,241,472 字节；比唯一写入量多的原因是 MLA 被多个 rank 读取。这些是对象有效字节，不是 SSD 介质流量。

**验证边界。** 原生对象布局、块表、查找及 Mooncake TCP Store 多缓冲区 PUT/GET 均实际执行，事件中 135 次 GET 读前均只有 LOCAL_DISK。运行用一个 embedded client、一套顺序复用的 CPU 缓冲区和宿主文件系统；没有八个并发服务进程，没有执行 K3 forward、在线 WEKA 调度或 FEMU。缺片场景模拟未提交，不覆盖删除后同键重写。结果不证明步骤 1 的 71,168-token 共享前缀在实际调度时能生成对应 checkpoint。

**复核。** [逐对象和逐 rank 清单](../evidence/kimi-k3-step3/summary.json) 与[原始 Store 事件](../evidence/kimi-k3-step3/events.jsonl) 已保留。项目根目录运行：

```bash
.venv/bin/python scripts/check_k3_shard_evidence.py evidence/kimi-k3-step3
```

需重做真实存取时，使用一个尚不存在的输出目录；需要本机回环 TCP、项目原生依赖、约 1 GiB 可用磁盘及数 GiB 可用内存：

```bash
.venv/bin/python scripts/verify_k3_shards.py --output outputs/k3-step3-repeat
.venv/bin/python scripts/check_k3_shard_evidence.py outputs/k3-step3-repeat
```

本次完整运行目录为 `outputs/k3-step3-03`。前两次启动分别因布局类导入位置和沙箱禁止回环 socket 提前终止，均未计入验收。证据复核通过；部署及混合缓存相关的 11 项回归通过。下一步是步骤 4：将输入与配置传入 FEMU 客体，并分离冷缓存准备和请求计时；不直接将本次约 471 MB 对象塞入现有 384 MiB 客体盘。
