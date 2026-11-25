# AIPerf 接入与验收

日期：2026-11-17。状态：第一版接入已完成，两组端到端验收通过。

后续更新：已完成 [AIPerf＋FEMU 联合功能验收](AIPerf-FEMU联合验收报告.md)。包含客户端 TTFT 的严格慢读对照未通过，原因和证据单独记录；本报告以下两组数据仍仅对应宿主文件系统。

## 实现范围

链路：AIPerf 0.13.0 原生 WEKA loader / 会话与分支调度 → HTTP streaming chat → 本项目执行循环 → 原生 vLLM Scheduler / KVCacheManager / MooncakeStoreConnector → 原生 Mooncake 磁盘 offload。

新增 `kv-ssd-sim serve`，仅监听 127.0.0.1，提供 `/health`、`/v1/models`、`/v1/chat/completions`。支持文本流式和非流式响应；不支持真实模型推理、多模态、工具调用执行或多候选采样。接口由本项目实现，不是 vLLM 上游完整 HTTP serving 栈。每个输出 token 在 scheduler 更新输出之后才发送；不会提前按输出长度直接返回完整文本。

AIPerf 安装于独立 `.venv-aiperf`，锁定依赖在 `requirements-aiperf.lock.txt`，现有 `.venv` 的原生依赖未被覆盖。参考 [官方 WEKA 教程](https://docs.nvidia.com/aiperf/v0.13.0/tutorials/datasets-inputs/replay-weka-agentic-coding-traces)。运行与验证依据实际安装的 0.13.0 包和其 CLI。

## Tokenizer 和计时口径

`configs/sim-tokenizer` 是项目生成的可逆字符 tokenizer，32000 词表、无 BPE 合并、无角色边界额外 token。模板串接消息文本，保留 AIPerf 合成前缀与输入长度。它不是 Claude 或目标生产模型的 tokenizer，不验证真实文本或模型数值。

服务端接到请求后，在原生入队前准备该请求的占位状态。最多保留 32 个准备中/执行中的请求（可配置），占位摘要内存上限 512 MiB，请求完成后释放其摘要。已完成请求的指标和校验目录仍保留，因此本版不宣称内存对无限长流恒定。客户端延迟包含 HTTP、准备、入队等待及显式冷缓存操作，服务端 TTFT 从原生入队开始；不能混为一个指标。

固定的是每 step 的计算预算。实际 CPU 复制、调度及 OS 唤醒可能使墙钟时间超过预算，继续保留原有开销记录。

AIPerf 0.13.0 默认的客户端输入估算会将多条文本消息以连接符拼接。本实验第二轮曾出现客户端估算 321、实际服务端 320 的偏差。启动脚本采用 `--apply-chat-template --use-server-token-count`；验收额外读取保存的原始 HTTP payload，使用相同 tokenizer 独立分词，与 trace 长度和服务端计数三方核对。单独设置 apply-chat-template 在本次版本中没有消除该估算偏差。

因 0.13.0 的离线加载分支会把绝对 tokenizer 路径误当 Hub snapshot 名，客户端进程不继承 `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE`；仍显式使用项目本地 tokenizer。服务端继续本地加载，不下载模型权重。

## 成功实验

| 场景 | 请求数 | 输入 / 输出 tokens | 验证结果 |
|---|---:|---:|---|
| 两请求冷恢复 `aiperf-cold-05` | 2 | 576 / 8 | 第二条从 Store 恢复 256 tokens；4 个 MLA 对象和 1 个 KDA 快照校验通过 |
| 子 agent 分支 `aiperf-dag-05` | 5 | 960 / 16 | 两个子 agent 并行，原生 batch 最多包含 2 个请求；父请求等子任务完成后继续 |

两组均使用原生 Mooncake、本机文件系统、tiny MLA＋KDA 配置，没有执行 FEMU 实验。冷恢复组主动串行并驱逐内存，仅用于功能验收；分支组不启用强制冷缓存，其外部恢复数为 0，不能据此声称验证了并发 SSD 压力。

验证包括：来源请求无遗漏或重复、实际 HTTP 输入长度、输出长度、usage 数量、共享前缀块摘要、原生首 token 先于 HTTP token、恢复完成先于首 token、顺序会话及明确完成子 agent 的 JOIN。检查器对异步背景任务不强加阻塞 JOIN。

证据：

- `outputs/aiperf-cold-05/verification.json`
- `outputs/aiperf-dag-05/verification.json`
- 每组 `launch.json`、`server/run_manifest.json`、`server/events.jsonl`、`server/summary.json`
- 每组 `client/profile_export_aiperf.json`、`profile_export.jsonl` 和 `profile_export_raw.jsonl`
- 全量原生回归：`outputs/aiperf-regression.log`，**39 项通过**；Ruff 检查通过。

早期 debug、cold/dag 01～04 产物保留为诊断记录，不算通过验收：包括 tokenizer 离线加载问题、退出时过早终止 master，以及客户端 token 估算偏差。最终脚本先让 HTTP 服务排空发送并确认磁盘副本，再由服务自身关闭 master。

## 复现

在项目根目录使用新输出目录，需要允许本机 socket：

```bash
# 官方 AIPerf WEKA loader → HTTP → 原生混合缓存冷恢复
.venv/bin/python scripts/run_aiperf.py \
  --input examples/aiperf-two-requests.weka.json \
  --output outputs/my-aiperf-cold --cold-between-requests

# 原生多请求并发与子 agent JOIN；不主动清缓存
.venv/bin/python scripts/run_aiperf.py \
  --input examples/aiperf-subagents.weka.json \
  --output outputs/my-aiperf-dag --prefill-ms 30 --decode-ms 10

# 独立启动服务，供外部 AIPerf 使用
.venv/bin/python -m kv_ssd_sim serve \
  --output outputs/my-http-server --port 8000 \
  --hybrid-config configs/hybrid-tiny.json
```

`run_aiperf.py` 管理自有进程、记录完整命令，并自动调用验证脚本。默认 causal 模式对应 AIPerf 的 `--no-fixed-schedule --use-think-time-only`；`--timing fixed` 使用 AIPerf 固定时间模式，不能与原项目自写 fixed 模式的依赖语义直接画等号。

重新创建 AIPerf 环境可用：

```bash
.venv/bin/python -m venv .venv-aiperf
.venv-aiperf/bin/python -m pip install -r requirements-aiperf.lock.txt
```

若需重新生成 tokenizer，使用 `scripts/build_sim_tokenizer.py --output <新目录>`。工具不会覆盖已有目录。

## 尚未覆盖

- 这两份输入是完整补齐 AIPerf schema 的合成 WEKA 验收夹具，不是公开 Claude Code 全量 trace。真实数据审计和完整回放仍按实施计划推进。
- AIPerf→HTTP→FEMU 已在后续联合实验验证，当前 serve 支持 `--femu-mount`；其结果及未通过的端到端慢读对照见联合报告，不能把本报告中的宿主实验计为 FEMU 验收。
- 目标模型 tokenizer/template、深层分支及所有 WEKA schema 变体尚未系统验证；服务端不静默截断超长输入。
- 同键删除后重落盘的 Mooncake 问题仍未修复。此冷恢复使用仅清内存副本的路径。
- 客户端断开后，已入队的原生任务继续完成以排空缓存引用；不提供即时取消模型执行的保证。
- 关闭时检查全部已写对象的磁盘副本，适合本轮小型验收；长期自然淘汰实验需区分允许淘汰的对象与要求保留的对象，不能直接沿用此最终屏障作生产耐久性判据。
