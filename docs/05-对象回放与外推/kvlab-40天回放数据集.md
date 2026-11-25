# 下一阶段的主回放数据集

2026-11-24，项目选定 **WEKA kvlab 的 40 天原始 trace** 作为下一阶段对象回放和 FEMU 实验的主输入，用于研究多会话下单台 K3 服务器的 KV 缓存给 SSD 带来的流量及 NAND/GC 行为。选定版本为 `wekaio/replay-40d-oct14-337a03810637`，格式为 `replay-v5`，覆盖 2026 年 9 月 5 日至 10 月 14 日，共 3,190 个会话、370,658 个请求。[原始数据](https://huggingface.co/datasets/wekaio/replay-40d-oct14-337a03810637)

选择它的原因是会话开始时间 `start_ts` 和请求时间 `ts` 共享 UTC 时间轴。一个 agent 等待工具或用户时，可以查出这段时间里还有哪些会话发了请求。回放时保留这些到达间隔，并记录哪些会话被分配到同一台 K3，就能研究其他会话对旧前缀驻留和回读的影响。[kvlab 回放格式](https://github.com/weka/kvlab/blob/main/docs/replay.md)

原有 `kv-cache-tester` 的 `0504` 两轮继续用于 n15 回归对账。AgentX 可用于固定并发度的对照实验。[已有 KVM 基线](../04-n15迁移与容量验收/n15-KVM基线与56GiB容量验收.md) · [AgentX 方法](https://inferencex.semianalysis.com/infx/agentx/methodology)

已完成全库审计和首轮对象回放。保留全部前序后，n15 的 12,288-token 上限覆盖 1,962 个请求，参考配置的 131,072-token 上限覆盖 30,760 个。报告包含实际源端并发、n15 留出验证和参考窗口的对象级写入曲线。[首轮窗口与 SSD 对象负载验收](kvlab首轮窗口与SSD对象负载验收.md) · [项目容量配置](对象回放器-FEMU-真实服务器差距说明.md)
