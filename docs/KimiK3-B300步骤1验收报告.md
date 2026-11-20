# Kimi K3＋B300：步骤 1 验收报告

日期：2026-11-18。范围：固定参考部署、完整审计一个来源文件、选取初始切片并验证 AIPerf 离线预处理。

**步骤 1 已完成。** 本次不加载模型权重，不运行目标 K3 调度或 FEMU 实验。参考部署配置尚未接入运行时；步骤 2～6 继续待实施。当前交付不能作为 B300 性能或整机容量验证。

## 固定参考部署

机器可读配置：[kimi-k3-b300-reference.json](../configs/kimi-k3-b300-reference.json)。外部来源、不可变链接及完整 SHA256 见 [sources.json](../references/kimi-k3-b300/sources.json)，环境记录见 [software-lock.json](../references/kimi-k3-b300/software-lock.json)。

| 项目 | 本次固定值 |
|---|---|
| 模型 | `moonshotai/Kimi-K3`，revision `f831ab66814297da540d832a5235f8e904f29d06` |
| 模型元数据 | 93 层；69 KDA、24 MLA；逐层编号保留在原始 config，权重量化格式 `mxfp4-pack-quantized` |
| 部署 recipe | `vllm-project/recipes@3eeceeab6265abdfed2a3aa48c5d76d9e0636e25`，`models/moonshotai/Kimi-K3.yaml` |
| recipe 对应 vLLM 源码 | `935f87de846980dc16c54ac1996595fc12eb3f9e`，来自同快照的 sources.lock.json |
| 服务器参考 | 单节点 DGX B300、8 卡；标称每卡 288 GB、主存 2 TB、8×3.84 TB 缓存 NVMe |
| 并行 | TP8、PP1、DP1、DCP1；关闭 expert parallel |
| KV/state 目标 dtype | MLA FP8；KDA 卷积 BF16、递归状态 FP32；实际 spec、scale、padding 待步骤 2 对照 |
| 调度目标 | max model len 131072；max num seqs 8；每 step 最多 4096 tokens；chunked prefill 开启 |
| 前缀匹配 | 128 tokens；trace hash 粒度为 64，两者分别记录 |
| 计算假设 | prefill step 10 ms、decode step 2 ms；混合 step 按 prefill；未经 B300 标定 |
| 请求范围 | 文本代理；不执行 forward、不加载权重、不做推测解码 |

上下文长度、batch 上限和固定计算预算是本项目选择，不是官方服务器测量值。原始 recipe 仍完整保留以便比对。没有选择或启动 `nightly/latest` 容器；镜像 digest 暂无，参考身份固定到源码及 recipe commit。

**当前环境与参考环境分开记录。** 已安装 vLLM 是 `0.34.0rc1.dev302+gbe47189f3.cpu`，commit `be47189f37f6162991366ff6b91539f6e9d1f432`；Mooncake non-CUDA 为 0.3.14，AIPerf 独立环境为 0.13.0，FEMU commit 为 `a898d22b2f894042afe0f4ef21867393c5c78a70`。本次未升级这些组件。参考 recipe 与本地 vLLM 有版本差异，后续必须验证 K3 cache spec/connector 兼容性，不能默认一致。

每 rank KV、Mooncake 内存、SSD 容量及物理块尺寸有意留空，交由步骤 2 从原生规格和资源预算推导。标称显存不能直接当 KV 池；当前 384 MiB FEMU 设备也未改成服务器标称容量。

## 输入审计与切片

来源：`callanjfox/kv-cache-tester@94f6046dfd9c0c8bc0cdb24a5aa6579bac8669bf` 的 `traces/trace_0001.json`。

- 原始文件 SHA256：`cfd45b93c58e16b1ae3cb177f91d45bb58590dc592e140e28e4155145746feb3`。
- 模型标记为 `claude-opus-4-5-20251101`；41 条父请求，0 个显式子 agent；文件中观测到的父请求峰值在途数为 1。
- 输入总量 4,873,373 tokens，输出总量 27,119 tokens；最大输入 196,860，最大输入＋输出 197,889。
- 在本次 131,072 上下文限制下，有 13 条请求超限，具体位置全部列在 [source-audit.json](../datasets/kimi-k3-b300-step1/source-audit.json)。没有缩短这些请求或将其静默丢弃。
- 按现有每 token 每层 32 字节摘要方式及 93 层估算，全文件预准备需 14,583,864,192 字节，超过现有 512 MiB 保护上限。

第一份实验输入使用连续的 `/requests/0`、`/requests/1`，从会话开头选取，保留前序关系，未选择后续 39 条。每条记录的长度、hash、时间及其他字段原样保留；仅去掉已经失效的全文件 `totals`。来源和切片原因放在 sidecar，避免向 AIPerf 严格 schema 插入额外字段。

| 原始位置 | 输入 tokens | 输出预算 tokens | 原始 t / api_time（秒） | think time（秒） |
|---|---:|---:|---|---:|
| `/requests/0` | 71,175 | 169 | 0 / 7.34 | 0 |
| `/requests/1` | 72,762 | 296 | 19 / 9.19 | 12 |

共享完整前缀为 **71,168 tokens**，128-token 对齐后仍为 71,168，占第二轮输入约 97.81%。部分尾块不当作已确认共享前缀。该切片的最大输入＋输出为 73,058；摘要估计 429,740,352 字节（约 409.83 MiB），在 512 MiB 摘要限额内。**这只通过输入/摘要预算检查，尚未证明 K3 缓冲区与全部 rank 能装入本机或 FEMU 客体。**

切片：[slice.json](../datasets/kimi-k3-b300-step1/slice.json)，SHA256 `40379c1607947b2d84c9de6a37d501fc8f0adafd59b14f0cc7a7d7585cfd3896`。范围与依赖记录见 [selection.json](../datasets/kimi-k3-b300-step1/selection.json)。

依赖完整仅指所选两轮串行父会话，未声称恢复完整工具执行 DAG。审计器可递归盘点分支，但自动切片会拒绝包含子 agent、时间重叠、缺失 think time 或无共享前缀的候选，不擅自推断复杂 agent 依赖。

## AIPerf 离线核对

实际调用安装的 AIPerf 0.13.0 `WekaTraceLoader`、`PromptGenerator`，并通过现有本地 tokenizer 的 chat template 核对最终 token IDs。未发送 HTTP 请求，未使用 GPU。

- 原文件和切片均通过 AIPerf schema 校验。
- 生成 1 个会话、2 个请求，与源索引一一对应；第二轮完整提示重建并设置 reset_context。
- 合成输入准确为 71,175 / 72,762 tokens，输出预算准确为 169 / 296，没有设置 ISL/OSL 截断。
- 实测 token IDs 的最长公共前缀为 71,168，与 trace 的完整块前缀一致。
- 第二轮 delay 为 12,000 ms。后续采用前轮实际完成＋think time 的因果回放；原 t=19 秒仅保留作来源信息，不强制按原始绝对时间发起。原 api_time 不充当 K3 计算时延。
- 固定随机种子 42，记录 tokenizer 文件及每轮合成 token IDs 哈希；复核脚本会重新生成并逐字节比较审计结果。

结果：[aiperf-audit.json](../datasets/kimi-k3-b300-step1/aiperf-audit.json)。使用的是本地合成 tokenizer，保留 trace 长度与前缀结构，**不代表 Claude 原文经 K3 tokenizer 后的长度**。此次只校验离线预处理，在线 HTTP、排队、缓存恢复及客户端 TTFT 由后续步骤验收。

## 复现与检查

在项目根目录运行完整离线复核：

```bash
cd /home/plusyu/projects/kv-ssd-sim
.venv/bin/python scripts/verify_step1.py --aiperf
```

该命令检查固定文件 SHA256、模型层覆盖，重建四个 trace 审计文件，再由 `.venv-aiperf` 重新生成合成输入审计。无须网络、GPU 或 FEMU；不会覆盖已有证据。省略 `--aiperf` 可仅检查固定文件及基础输入审计。

本次实际执行结果为 `passed`：29 个固定文件通过校验，四份输入文件逐字节重现，AIPerf 审计（含合成 token IDs 哈希）也逐字节重现。Ruff 检查通过。

独立生成新目录用于检查来源与切片：

```bash
.venv/bin/python scripts/audit_weka.py \
  --input references/kimi-k3-b300/trace_0001.source.json \
  --source-sha256 cfd45b93c58e16b1ae3cb177f91d45bb58590dc592e140e28e4155145746feb3 \
  --output outputs/k3-step1-input-audit --select-first 2
```

输出目录须不存在；源哈希不匹配、输入非法、候选超上下文或摘要预算时明确失败。原文件始终只读。

验证命令与结果：

```bash
.venv/bin/pytest -q tests/test_audit.py tests/test_trace.py tests/test_payload.py tests/test_timing.py
# 28 passed（含 16 个新增审计用例）
.venv/bin/ruff check src/kv_ssd_sim/audit.py scripts/audit_weka.py \
  scripts/audit_aiperf_input.py scripts/verify_step1.py tests/test_audit.py
```

用例覆盖篡改哈希、非法长度/时间、嵌套分支盘点与拒选、未知类型、不完整尾块、重叠请求、缺失 think time、上下文/摘要超限，以及真实源文件的确定性重建。原有 FEMU 配对 failed 证据未变更，本次不重复宣称其通过。

## 下一步

进入步骤 2：将固定 K3 原始配置转成原生 MLA/KDA cache spec，核对 FP8/状态 dtype、NoPE、padding、128-token 匹配和 TP8 分片，再建立独立的每 rank KV、Mooncake 及 SSD 容量预算。现有运行脚本仍使用小模型配置，不应直接把本步骤参考 JSON 当作已支持的运行参数。
