# 步骤 1 固定参考资料

`sources.json` 给出每份外部文件的不可变 revision、原路径、链接、大小与 SHA256；GitHub 文件另记 Git blob SHA1。文件保持上游字节（`trace_0001.source.json` 原本没有末尾换行）。`software-lock.json` 记录现有 CPU 环境和参考 GPU recipe 的版本差异；`artifacts.lock.json` 固定本步骤的配置、输入、代码和报告证据。

- `model-config.json` / `model-tokenizer-config.json`：moonshotai/Kimi-K3 元数据。许可证见 `model-LICENSE`；没有下载权重或执行其远程模型代码。
- `recipe.yaml` / `recipe-sources.lock.json`：vllm-project/recipes 的固定快照。许可证见 `recipe-LICENSE`。保留原文中的滚动 tag 仅供查阅；实验身份使用 sources 中的 commit，未构建或启动 GPU 容器。
- `trace_0001.source.json`：callanjfox/kv-cache-tester 中的一份公开 WEKA 文件；许可证见 `weka-LICENSE`。这是 Claude 工作负载的结构代理，不是 K3 实测输入输出，也不代表完整 WEKA 数据集。

硬件网页只作为标称规格参考；实际可用容量和 SSD 几何/时延尚待步骤 2 建模，不把标称容量或 recipe 的预估权重开销视作测量数据。

复核与范围说明见 [步骤 1 验收报告](../../docs/KimiK3-B300步骤1验收报告.md)。后续若改变输入、模型 revision、配置或预处理实现，应生成新的基线并重新验收，不只更新哈希来掩盖差异。
