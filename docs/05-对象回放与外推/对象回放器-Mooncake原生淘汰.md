# 对象回放器接入 Mooncake 原生淘汰

2026-11-23，n15。研究量是指定会话、容量和服务时间下的对象级 SSD 写入与恢复需求。回放入口默认使用与本机在线实验同一发行包的 Mooncake master 实现。vLLM 继续产生 K3 的真实键和尺寸，KV 张量使用 meta 设备，主机只分配元数据。以下记录覆盖默认单租户、每个键一个内存副本、eager offload 和默认混放。

## 实施步骤

| 步骤 | 实现与核验 |
| --- | --- |
| 1．固定执行代码 | 构建器校验 Mooncake 0.3.14 的官方源码归档、n15 已安装 wheel 的 `_store` 动态库和在线 `mooncake_master` 的 SHA-256。版本对应 `e1ea5d31bb5e46ff2df9f90e7c2fbaf52fcdee8c`。C++ 驱动直接链接 wheel 导出的 `MasterService`。每次启动复核哈希，并把执行文件、库、驱动源码和编译器写进 `launch.json`。 |
| 2．登记虚拟内存 | 每个虚拟客户端挂载一个带独立端点的 offset 内存段，向 master 登记磁盘配额。保存任务开始时调用 `PutStart`，立即计入原生 allocator。任务完成时调用 `PutEnd`，随后 vLLM 才收到保存完成通知。请求写入者的首选段及 master 实际分配的段分别保留。源地址只作为段内偏移的载体。 |
| 3．交付完整租约和落盘事件 | 连接器查存在性走 `BatchExistKey`，恢复查副本走 `GetReplicaList`。虚拟客户端按心跳领取 master 的 offload 任务，通过原生 `AllocateOffloadingBuckets` 分桶。与 FileStorage 一致，每个 bucket 写前都会调用 `BatchGetReplicaList` 查询源副本并续租。完成时按桶调用 `NotifyOffloadSuccess`，失败使用 `data_size=-1` 释放源引用。磁盘配额不足时，使用同一库的 `BucketStorageBackend::SelectEvictionCandidate` 选择整桶。 |
| 4．驱动原生内存回收 | 独立子进程的系统时钟使用回放时间，固定起点与 100 ms 原生租约索引桶对齐。使用 upstream 的测试友元接管后台内存回收时钟并停用客户端超时剔除，虚拟客户端在会话间持续在线。每个 10 ms 边界读取 master 的全池占用与分配压力，按固定源码的比例公式调用 `MasterService::BatchEvict`。候选索引、租约、引用检查、对象目标和实际释放都在同一 wheel 中执行。无压力时可跳过连续空轮次，跨越空闲期时仍执行有压力的轮次。 |
| 5．审计与时间校准 | 每次回收记录时间、回收前后原生占用、实际失去内存副本的键。观察通过 `GetReplicaListForAdmin`，避免给观察对象续租。结束时逐键核对 Python 账本和 master 的实际副本状态。运行输出每个原生 bucket 的源查询、写入开始、落盘回执时间。同一入口支持读入对齐后的时间记录。 |

构建只编译驱动。构建时给同版本头文件加一个仅用于访问虚拟 bucket 索引的友元声明，磁盘选择函数与内存选择函数均从已校验的动态库执行。项目中的 Python `ObjectStore` 负责虚拟传输、事件与账本。[构建脚本](../../scripts/build_object_master.py) · [C++ 驱动](../../native/object_replay_master.cpp) · [虚拟 Store](../../src/kv_ssd_sim/object_store.py) · [vLLM 保存与恢复入口](../../src/kv_ssd_sim/object_worker.py) · [上游 master](https://github.com/kvcache-ai/Mooncake/blob/e1ea5d31bb5e46ff2df9f90e7c2fbaf52fcdee8c/mooncake-store/src/master_service.cpp) · [上游 FileStorage](https://github.com/kvcache-ai/Mooncake/blob/e1ea5d31bb5e46ff2df9f90e7c2fbaf52fcdee8c/mooncake-store/src/file_storage.cpp)

## 验收与实测

| 条件 | 对象回放 | n15 在线实测 | 验收结果 |
| --- | --- | --- | --- |
| 受控缺片及强制回读 | 36 个对象，470,605,824 B。120 次 rank GET，全从虚拟磁盘恢复，共 619,241,472 B | 原受控探针同值。三个边界分别命中 0、0、1,536 token | 查询、分片和恢复多重集逐项通过 |
| `0504` 两轮，2,520 MiB 内存段 | 208 个键，2,402,574,336 B，10 个 bucket。零 GET。结束时 195 个键有内存副本 | 对象集合、尺寸与零 GET 一致。197 个键有内存副本 | 请求与磁盘库存通过。最终内存键集合有差异，模型独有 4 个，实测独有 6 个 |
| 短等待返回，1,450 MiB 段、5 s 租约 | 120 次 GET，其中磁盘 37 次、150,242,304 B | 120 次 GET 全部可从内存取得，磁盘 0 B | 请求、PUT/GET 键与字节、磁盘库存通过。37 次 GET 层级不同 |
| 35 s 返回，同一容量和租约 | 120 次 GET，其中磁盘 37 次、150,242,304 B | 历史记录及本次提前冻结预测的独立实跑均为磁盘 12 次、89,051,136 B | 对象需求通过。按请求、rank、键匹配，历史对照有 31 次层级不同，本次实跑有 47 次 |
| 参考容量，8 个客户端共 1,024,000,000,000 B 内存段 | 完整 `0504` 与 `0001` 前两轮，5 个请求。1,427 个对象、15,545,991,168 B，72 个 bucket。原生分配字节数与驻留有效载荷相符，零回收轮次。master 子进程峰值 RSS 60,552 KiB，整次 GNU time 记录的峰值 RSS 为 735,864 KiB | 此项使用参考预算，尚无 B300 线上逐键记录 | 验证到该容量的元数据执行及虚拟化。这组输入没有覆盖参考容量的压力区 |

每组结束时，回放器均用无续租的管理查询检查了所有对象。对账中的层级是在线 GET 前可用的副本类型。“磁盘 GET”表示当时只有磁盘副本。表内字节均按对象有效载荷计。模型校验键与尺寸。`0504` 的内容摘要来自真实写入数据的在线日志，为 `173bfdbe0ff676b4301cee05be60c30795a68772776cbd8565773d966fda0999`。[基线逐键对账](../../evidence/object-replay-mooncake-native-20261123/comparisons/baseline.json) · [短等待](../../evidence/object-replay-mooncake-native-20261123/comparisons/short.json) · [长空闲](../../evidence/object-replay-mooncake-native-20261123/comparisons/idle.json) · [冻结预测实跑](../../evidence/object-replay-mooncake-native-20261123/comparisons/frozen.json)

**差异已经出现在选择候选的时刻。** 短等待线上 master 记录了 675 个回收结果，每轮释放数均为零。长空闲线上 master 的有效轮次记录 `eviction_base=72`、释放 5 个。回放两种等待都在虚拟时间 9.830 s 触发一轮，`eviction_base=108`、释放 9 个。两端执行相同的选择代码，但读到的分配占用、已完成可见对象及租约历史仍有差别。作为一个时序参照，短等待线上 48 次 Store 批量 PUT 调用的耗时中位数为 110.0 ms，本次冻结对照为 132.3 ms。回放沿用每 rank 每个保存任务 1 ms 的设定。批量 PUT 与保存任务是不同粒度，不能把中位数直接填入该参数。[轮次、分配账与 PUT 耗时统计](../../evidence/object-replay-mooncake-native-20261123/verification.json)

**时序接口已单独验证。** 受控测试把源查询放在 5 ms、写入开始放在 8 ms、成功回执推迟至 1,000 ms，租约设为 20 ms。即使租约早已到期并持续触发压力，master 仍保留全部源副本，收到回执后的维护轮次才释放内存。另一组验证 PUT 未完成时已占用段容量且查询未命中、查询续租能保护被访问的键、NACK 后可回收最后一份内存副本、磁盘准入按原生候选回收整桶。将长空闲回放导出的 6 个 bucket 时间表重新输入后，事件、对象、请求、复用距离及时间表五个文件逐字节相同。项目非 integration 测试 104 项通过，另 6 项带有 integration 标记，独立在线对照和缺片探针另外执行。[测试](../../tests/test_object_replay.py) · [测试结果](../../evidence/object-replay-mooncake-native-20261123/regression.log) · [探针对账](../../evidence/object-replay-mooncake-native-20261123/probe.json)

**当前可外推的是给定事件和预算时的 Mooncake 对象决策。** 缺省仍用 10/2 ms 计算预算、每 rank 保存任务 1 ms、客户端心跳间隔 1 s、共享 FIFO 设备 1,024 MiB/s 和内存读端口 8,192 MiB/s。心跳间隔可用 `--offload-interval-ms` 配置，旧参数名 `--offload-delay-ms` 也指向这个间隔。实际客户端的批次并行、设备队列、文件对齐和发布回调延迟会改变这些输入。虚拟磁盘配额按有效载荷计算，客户端传输重试及超时、文件系统块 I/O、NAND 和 GC 由在线链路观测。要据此估计目标服务器的读盘比例，仍需一次对齐 PUT 分配与完成、源查询及落盘回执的真实压力窗口。已有线上结果只记录了其中部分边界。

本次原始试验位于 `.object-replay-build/trials/20261123-native-metadata/verified/`。归档保留预测、输入、时间表和真实 master 日志。`online/frozen` 对应启动在线服务前冻结的五份预测文件。历史对照使用显式 `--retrospective`，冻结对照使用默认的哈希校验。对账脚本在 GET 层级或最终驻留键不符时返回 1，JSON 保留逐键差异。[原始记录压缩包](../../evidence/object-replay-mooncake-native-20261123/run-records.tar.gz) · [校验清单](../../evidence/object-replay-mooncake-native-20261123/SHA256SUMS)

## 复现

以下命令在 n15 仓库根目录执行。归档含本次的编译后会话、逐键事件、时间表及在线对照日志。源码压缩包已缓存在 n15，其他机器可省略 `--archive` 由构建器按固定 revision 获取并校验同一份源码。离线重建需要 n15 的 Python 3.13 wheel、对应 libpython、g++-12 及已安装的 Mooncake C++ 头文件依赖。

```bash
cd /home/yzh/projects/kv-ssd-sim
root=$PWD
trial=$(mktemp -d /tmp/k3-native-object.XXXXXXXX)
/home/yzh/miniconda3/bin/python3.13 -m venv --without-pip "$trial/py"
py="$trial/py/bin/python"
pkg="$root/.venv/lib/python3.13/site-packages"
export PYTHONPATH="$root/src:$pkg" LD_LIBRARY_PATH=/home/yzh/miniconda3/lib
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 OMP_NUM_THREADS=1
export VLLM_SSM_CONV_STATE_LAYOUT=SD VLLM_CPU_KVCACHE_SPACE=0 VLLM_HOST_IP=127.0.0.1
export CUDA_VISIBLE_DEVICES='' GLOO_SOCKET_IFNAME=lo KV_SSD_USE_SYSTEM_LIBRARIES=1
export VLLM_CACHE_ROOT="$trial/cache"
unset PYTHONHASHSEED
ln -s "$pkg/mooncake/mooncake_master" "$trial/py/bin/mooncake_master"
srcpkg="$root/.object-replay-build/sources/Mooncake-e1ea5d31bb5e46ff2df9f90e7c2fbaf52fcdee8c.tar.gz"
"$py" scripts/build_object_master.py --archive "$srcpkg"
evidence="$root/evidence/object-replay-mooncake-native-20261123"
(cd "$evidence" && sha256sum -c SHA256SUMS)
tar -xzf "$evidence/run-records.tar.gz" -C "$trial" inputs
"$py" -m pytest -q tests/test_object_replay.py tests/test_storage.py tests/test_k3_online.py
"$py" scripts/check_object_probe.py --output "$trial/probe.json"
"$py" scripts/run_object_replay.py "$trial/inputs/0504.json" --output "$trial/baseline"
opts=(--num-blocks 11 --segment-mib 1450 --lease-ms 5000)
"$py" scripts/run_object_replay.py "$trial/inputs/return.json" "${opts[@]}" --output "$trial/short"
"$py" scripts/run_object_replay.py "$trial/inputs/return-idle.json" "${opts[@]}" --output "$trial/idle"
"$py" scripts/run_object_replay.py "$trial/inputs/return-idle.json" "${opts[@]}" \
  --offload-timeline "$trial/idle/bucket-timing.jsonl" --output "$trial/with-timing"
"$py" scripts/run_object_holdout.py "$trial/inputs/return-idle.json" \
  --prediction "$trial/idle" --output "$trial/online"
"$py" scripts/compare_object_replay.py --prediction "$trial/idle" \
  --server "$trial/online/server" --output "$trial/comparison.json"
"$py" scripts/run_object_replay.py "$trial/inputs/0504-full.json" "$trial/inputs/0001-two.json" \
  --profile reference-20261124 --offset-s 0 --offset-s 0.08 --output "$trial/reference"
```

时间表每行包含 `owner`、一组实际对象 `keys`、`source_query_ns`、`write_start_ns` 和 `publish_ns`，单位为相对于这次回放起点的纳秒。回放器先让 Mooncake 自行分桶，然后以客户端与完整键集合匹配时间。同一集合重复出现时按出现顺序匹配。丢行、多行、源查询早于任务可用、写入早于前序 I/O 完成等情况都会报错。把模型自己导出的时间表读回可以检验时序接口。要标定真实服务器，应填入与该输入及初始盘态对齐的实测记录。
