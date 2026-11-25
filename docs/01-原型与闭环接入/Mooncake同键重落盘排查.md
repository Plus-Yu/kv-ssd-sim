# Mooncake 同键删除后重落盘排查

日期：2026-11-16。范围：本项目安装的 Mooncake non-CUDA 0.3.14，master 日志报告源码提交 `e1ea5d3`，embedded client、bucket SSD offload、TCP、单机。

## 结论

已定位到 **master 的逻辑对象删除与 SSD bucket 后端的旧键索引之间的不一致**。重新 PUT 建立了内存副本；后台准备落盘时，bucket 后端仍判断旧键存在并跳过。该分支没有给新对象重新注册磁盘副本，而是按 offload 失败处理。

这不是 KDA 特有的状态尺寸或 vLLM 调度问题：去掉 vLLM、混合注意力和 ObservedStore，只使用原生 Mooncake 的普通字节对象，也复现了相同行为。没有 FEMU 参与，因此 FEMU 不是复现此故障的必要条件。

这里需要纠正原报告的用词：故障注入使用 `remove(key)` 删除 **逻辑对象**，不等于物理擦掉 SSD 上对应字节。实验中旧 bucket 文件和索引仍然使后端判断对象存在。

## 实验依据

### 原混合故障测试

新增 `disk_barrier_timeout` 事件，记录超时键及所有对象的副本层级。重新运行原来的故障场景得到：

| 对象 | 数量 | 最终副本 |
|---|---:|---|
| 未删除的旧 MLA | 4 | LOCAL_DISK |
| 删除后同键重新 PUT 的旧 KDA | 4 | MEMORY，缺少 LOCAL_DISK |
| 第二条请求产生的新 KDA | 1 | MEMORY＋LOCAL_DISK |
| 第二条请求产生的新 MLA | 1 | MEMORY＋LOCAL_DISK |

缺失磁盘副本的键集合与测试主动删除的键集合完全相等。说明后台落盘功能仍工作，失败集中在删除后复用的旧键。重新计算和指定长度的输出均完成。

证据：`outputs/mooncake-reput-hybrid-01/`；强化断言后的回归产物：`outputs/mooncake-reput-regression-01/`。

### 独立 Mooncake 最小复现

脚本：`scripts/diagnose_mooncake_reput.py`。使用普通字节串，先 PUT 并确认落盘，再驱逐内存副本，对部分键 remove，随后重写。相同进程中加入新键和保留磁盘副本的对照，观察 60 秒。

| 场景 | 大小 | 观察到的结果 |
|---|---:|---|
| 删除旧键，同键同内容重写 | 2624 B | 只有 MEMORY |
| 删除旧键，同键同内容重写 | 8192 B | 只有 MEMORY |
| 删除旧键，同键不同内容重写 | 2624 B | 只有 MEMORY |
| 保留磁盘对象，再次 PUT 同键 | 2624 B | 保持 LOCAL_DISK |
| 写入新键 | 2624 B | MEMORY＋LOCAL_DISK |

PUT 返回值均为 0。60 秒后旧键仍无磁盘副本，因此不能靠把原来的 15 秒屏障稍微延长来解决；这不等于证明任意长时间或所有配置下都永远不会恢复。

原生日志明确记录三个被删除的旧键触发 `storage_backend.cpp:2547` 的 `Key already exists in storage backend, skipping`。诊断针对副本元数据和 offload 路径，不宣称验证了同键改值后的磁盘内容或重启恢复。

正式最小复现产物：`outputs/mooncake-reput-isolated-04/diagnosis.json` 和相邻 `.log`。`isolated-01` 因沙箱禁止 socket 失败；`isolated-02` 因诊断脚本误把副本清除返回数量当成状态码而中止，已改成检查实际副本；`isolated-03` 已复现 60 秒落盘异常，但附加的普通 `get()` 在本项目零租约设置下还报告 `LEASE_EXPIRED`（新键也如此），不用于推断数据丢失。最终脚本只保留本问题所需的元数据检查。

复现命令（使用新的输出目录，需要允许本机 socket）：

```bash
.venv/bin/python scripts/diagnose_mooncake_reput.py \
  --output outputs/mooncake-reput-new --wait-seconds 60 \
  > outputs/mooncake-reput-new.log 2>&1
```

## 源码链路

检查固定提交，避免把别的版本行为混进结论。抓取副本、来源 URL 和 SHA256 在 `outputs/mooncake-reput-source/`。

1. [`RealClient::remove_internal`](https://github.com/kvcache-ai/Mooncake/blob/e1ea5d3/mooncake-store/src/real_client.cpp#L3005) 调用 master 的 Remove。
2. [`MasterService::RemoveObject`](https://github.com/kvcache-ai/Mooncake/blob/e1ea5d3/mooncake-store/src/master_service.cpp#L7384) 在本实验未开启 HA/oplog 的路径执行 `accessor.Erase()`。本次实测后端旧 bucket 的存在判断没有随逻辑删除失效。
3. [`BucketStorageBackend::IsExist`](https://github.com/kvcache-ai/Mooncake/blob/e1ea5d3/mooncake-store/src/storage_backend.cpp#L2318) 检查本地 `object_bucket_map_` 和 bucket 文件存在性；准备 offload 的[重复键分支](https://github.com/kvcache-ai/Mooncake/blob/e1ea5d3/mooncake-store/src/storage_backend.cpp#L2540) 因而直接跳过。实际二进制日志的文件名、行号及消息与该分支吻合。
4. [`FileStorage` 的未入桶任务处理](https://github.com/kvcache-ai/Mooncake/blob/e1ea5d3/mooncake-store/src/file_storage.cpp#L724) 将此类跳过任务列为失败，使用 `data_size=-1` 的元数据通知 master。
5. [`NotifyOffloadSuccess` 的负数分支](https://github.com/kvcache-ai/Mooncake/blob/e1ea5d3/mooncake-store/src/master_service.cpp#L8443) 清理 offload 任务，但不增加磁盘副本。注意这个 RPC 名字包含 Success，不代表批内所有任务落盘成功。

最终形成：master 只知道新内存副本，SSD 后端认为旧键已经存在，两边无法完成新一轮磁盘副本注册。

## 改动与验证

- 仅增强原型的超时观测，保留原有超时异常，不把失败转换成成功。
- 增加独立复现脚本，未修改安装的 Mooncake 二进制，也未替换原生存储实现。
- 强化原故障测试：精确核对缺盘键集合、仅内存层级及新 KDA 能正常落盘。
- 定向回归包含存储校验、缺失 KDA 重算、两种正常混合冷恢复；结果见 `outputs/mooncake-reput-regression-01.log`。本次未重新跑 FEMU 性能对照，已有报告为历史实验结果。

## 后续修复应落在哪里

上游修复需要协调逻辑对象删除、bucket 索引以及重写对象的代际身份。不能简单把“旧键存在”改成“直接给 master 登记成功”：同键新内容可能已经变化，会错误登记旧数据。

可考虑让逻辑删除使相应 bucket 索引项失效，或让重写携带可区分的对象代际，并安全写入新内容；均需处理共享 bucket 中的其他对象和正在进行的读取。不能删除整个 bucket 文件来修一个键，因为它可能包含其他仍有效的对象。

暂时隔离可继续使用现有“仅驱逐内存、保留磁盘副本”的冷恢复流程。若用新代际键规避，必须同步更新 lookup 的命名规则，并单独声明命中率与存储增长影响，不能暗中改键后当作原生删除恢复已修复。以上修复方向尚未实施或验收。
