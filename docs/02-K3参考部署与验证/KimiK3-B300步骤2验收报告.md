# Kimi K3＋B300：模型与服务器配置转换验收

2026-11-18，步骤 2 通过。交付对象是供步骤 3 使用的**配置和原生元数据规划结果**。用 CPU 调用了已安装 vLLM 的 K3 层 `get_kv_cache_spec`、分组、容量、匹配与 checkpoint 规划，以及 Mooncake connector 的 Store 分片规划。没有构造权重、分配八卡 KV 池、启动 Store 或测量 B300/FEMU I/O。

## 复核与使用

在项目根目录运行（只读、离线）：

```bash
.venv/bin/python scripts/plan_k3_deployment.py --check configs/kimi-k3-b300-step2/resolved.json
```

[完整结果](../../configs/kimi-k3-b300-step2/resolved.json) 的 `geometry` 是各类缓存形状，`layers` 逐一列出层名、类型、有效/物理字节、padding 和组号；`groups` 是 Store 对象单位；`ranks` 列出每 rank 的 head 范围、分片编号、KV 池及独立 Store 参数。`source_sha256` 固定转换输入。预算均为字节，报告中 GB/TB 为十进制，GiB 为二进制。

要比较另一种预算，可将 [server.json](../../configs/kimi-k3-b300-step2/server.json)、[mooncake.json](../../configs/kimi-k3-b300-step2/mooncake.json)、[ssd.json](../../configs/kimi-k3-b300-step2/ssd.json) 复制到新目录，修改预算后运行 `.venv/bin/python scripts/plan_k3_deployment.py --config-dir 新目录 --output outputs/k3-plan-new.json`。输出必须是新文件。目标仍是步骤 1 固定的 K3/TP8；原有 tiny 运行命令不消费这些预算。

## 缓存几何与 Store 分片

| 每层、每 rank | MLA（24 层） | KDA（69 层） |
|---|---|---|
| 数据形状 | 每页 `[1,1536,576]`；FP8 per tensor，原生字节视图 uint8 | SD 卷积 `[3,4608]` BF16；递归 `[12,128,128]` FP32 |
| 有效内容 | 576 字节/token；884,736 字节/页 | 卷积 27,648＋递归 786,432＝814,080 字节/快照 |
| 分配页 | 884,736 字节，无页尾 padding | 884,736 字节，其中页尾填充 70,656 字节 |
| TP8 | DCP1，一份 latent 在 8 个 rank 上复制 | 每 rank 负责连续 12 个 heads；rank 0 为 `[0,12)`，rank 7 为 `[84,96)` |
| Store TP8 | 一个 24 层组，每 128 tokens 只有 **1 个逻辑分片**，1,769,472 字节 | 三个 23 层组；每组完整快照 **8 个不同分片**，每个 18,723,840 字节 |

NoPE 关闭旋转，不删除 K3 NVIDIA MLA 写入路径的 64 维分量。该路径将投影拆成 latent 和另一个 key 分量；其 spec 使用 `kv_lora_rank + qk_rope_head_dim = 512 + 64`，无独立 V 缓存。普通 `fp8` 没有 `fp8_ds_mla` 的 656 字节布局，也不需要给每 token 加量化 scale。KDA 的 4-tap 卷积只保留 3 个历史位置。计算按模型一基层号映射到运行时零基索引，检查层覆盖、每层 spec 和原生组归属。

在本基线的 LBHNC、非 DSA、FlashKDA 配置下，TOKENSPEED 内核支持 32/64 tokens；混合 MLA 管理块要按 128 对齐且一页能装一份 KDA 状态，故 `128 × ceil(814080 / (576 × 128)) = 1536`。分配器按最宽 24 层组为每个物理块保留 21,233,664 字节；三个 23 层组各少用一个层位。不同组复用同一物理池；不能把四个组的池大小再相加。组 padding 不进入上述对象有效字节，实际块设备写量另测。

## 哈希、命中与 checkpoint

WEKA 的 64-token hash 用于合成输入；vLLM 的 hash 和可命中粒度经原生函数分别解算，均为 128；管理块为 1,536。FlashKDA 的**内部 checkpoint 偏移**需要 16-token 对齐，单次 prefill 最多一份内部 checkpoint，且必须有可用的块表列。`align` 模式下的运行态、已保留 checkpoint 和磁盘副本是不同对象状态。

固定 `compute_mamba_prefill_checkpoints` 边界探针：`[0,1536)` 无内部 checkpoint；`[0,1665)` 可把位置 1664 写入第 0 列。对两轮输入另取“每 4096 tokens 分段”的末段示例，`[69632,71175)` 无内部 checkpoint，`[69632,72762)` 的候选是 72704（第 46 列）。这是对指定切分的函数结果，不是实际 scheduler 时间线。**输入共享 71,168 tokens 并不保证有可恢复的同位置 KDA 快照**；步骤 3 需在实际调度范围内观察运行态、块号、导出及恢复结果。

## 容量账本

权重依据是[固定 revision 的 checkpoint 索引](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/model.safetensors.index.json) 的 `metadata.total_size=1560860324864`。这里只读了索引元数据；它是整份 checkpoint 的张量字节，不是设备实测驻留。以总量除以 TP 作估算起点，为复制、切分不均和加载格式另留空间。真实 GPU 应用启动后的逐卡数据替换这些假设。

| 单 rank 项 | 字节或预算依据 |
|---|---:|
| 标称显存 × 0.95 | 288,000,000,000 × 0.95＝273,600,000,000 |
| 减 checkpoint 平均份额 | 195,107,540,608 |
| 再减实验预留 | 复制/格式 16 GiB、激活/工作区 8 GiB、通信 4 GiB、图 4 GiB、其他 4 GiB |
| 剩余 KV 预算 | **39,837,753,728** |
| 原生池规划 | 1,876 块 × 21,233,664＝39,834,353,664；余 3,400,064 字节 |

1,876 是物理池总块数，包含运行时保留块；不是每层或每个组都有 1,876 个独占块，不能直接换算并发吞吐量。8 个 rank 均在结果中保留独立记录，当前对称预算是可复算的实验工作点。

**独立的主存与 SSD 预算：**2 TB 主存预留 500 GB；8 个 embedded client 各贡献 128 GB Store segment，另各留 4 GiB 传输缓冲和 4 GiB 磁盘 staging，剩余 407,280,523,264 字节。只选一块标称 3.84 TB NVMe，不组 RAID；预留 384 GB，每个 client 的独立目录限额 375 GB，总限额 3 TB，另余 456 GB。转换会拒绝预算溢出、目录冲突或 staging 小于单个对象对齐后所需空间。实际批次仍受并发与 staging 约束。这里没有修改本机已有 384 MiB FEMU 设备；如果缩放设备与工作集，须另外记录缩放规则。

## 证据范围与检查

模型与 recipe 使用[步骤 1 固定来源](../../references/kimi-k3-b300/sources.json)。参考 vLLM `935f87de…` 的 [K3 MLA 层](https://github.com/vllm-project/vllm/blob/935f87de846980dc16c54ac1996595fc12eb3f9e/vllm/models/kimi_k3/nvidia/mla.py) 以 `get_kv_cache_config` 包装 spec；本次实际运行的是锁定 CPU wheel `be47189f…` 的 `get_kv_cache_spec`。转换器仅向层提供构造 spec 所需的元数据，不执行 CUDA 层初始化。精确组号与布局以此 wheel、本配置的规划结果为准；未验证不同版本间二进制缓存兼容性。

已验证 93 个原生 spec、4 个组、8 个 rank 的 Store 规划，生成物逐字节复现；容量拒绝、独立性、checkpoint、分片与既有 trace/hybrid 测试共 27 项通过。复跑命令：

```bash
LD_LIBRARY_PATH="$PWD/.native/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  .venv/bin/pytest -q tests/test_deployment.py tests/test_audit.py tests/test_hybrid.py
```

下一步按每 rank 元数据验证真实尺寸的缓冲区、对象键、各分片保存/恢复与缺片行为；本报告不将规划对象大小当作已观测的 SSD I/O，也不改变此前 FEMU 时延配对的验收结论。
