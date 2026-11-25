# n15 KVM 基线与 56 GiB 容量验收记录

2026-11-20，宿主项目目录 `/home/yzh/projects/kv-ssd-sim`。**第一步通过，第二步停在构建资产检查，第三步未执行。** 本次直接归档 UTC 标签 `20261120-074735` 的已完成试验。两份 KVM NAND 报告和两份历史 TCG 报告均在临时目录从原始 QMP、客体标记和块 trace 重新生成，与保存的报告逐字节相同。56 GiB 尚无新二进制、trial 或容量实测值。

## 基线对照

探针五行按 **主机读 B / 主机写 B / NAND 编程 B / GC 搬迁 B / 擦除次数 / 字节 WAF** 排列。GC 搬迁包含在 NAND 编程内。0504 各设备指标的窗口统一为 `first_send → device_end`。

| 负载和指标 | n15 · KVM | 历史 · TCG |
|---|---:|---:|
| 探针整页写，empty → page | 0 / 4,096 / 4,096 / 0 / 0 / 1 | 0 / 4,096 / 4,096 / 0 / 0 / 1 |
| 同页覆盖 512 B，page → partial | 0 / 512 / 4,096 / 0 / 0 / 8 | 0 / 512 / 4,096 / 0 / 0 / 8 |
| 只读一页，partial → read | 4,096 / 0 / 0 / 0 / 0 / `null` | 4,096 / 0 / 0 / 0 / 0 / `null` |
| 顺序填盘，read → filled | 0 / 50,331,648 / 50,331,648 / 0 / 0 / 1 | 0 / 50,331,648 / 50,331,648 / 0 / 0 / 1 |
| 交错覆写，filled → rewritten | 0 / 75,497,472 / 603,971,584 / 528,474,112 / 2,304 / 7.99989149 | 0 / 75,497,472 / 603,971,584 / 528,474,112 / 2,304 / 7.99989149 |
| 0504 唯一对象数 / 逻辑字节 | 208 / 2,402,574,336 | 208 / 2,402,574,336 |
| 0504 自然 Store GET 批次 | 0 | 0 |
| 主机读 B，块提交 / 块完成 / diskstats | 8,192 / 8,192 / 8,192 | 8,192 / 8,192 / 8,192 |
| 主机写 B，块提交 / 块完成 / diskstats | 2,403,450,880 / 2,403,450,880 / 2,403,450,880 | 2,403,700,736 / 2,403,700,736 / 2,403,700,736 |
| NAND 编程 B | 2,403,450,880 | 2,403,700,736 |
| GC 搬迁 B / 擦除次数 / WAF | 0 / 0 / 1 | 0 / 0 / 1 |
| 原始 NAND 物理页占用率，起点 → 终点 | 2.796459% → 58.756161% | 2.796459% → 58.761978% |
| 有效页 / 失效页，起点 → 终点 | 29,292 / 31 → 615,904 / 199 | 29,292 / 31 → 615,905 / 259 |
| 客体采样往返耗时 ms，起点 / 终点 | 435.802 / 423.970 | 260.268 / 230.572 |

两份 0504 的原始 PUT 事件重算出同一对象摘要 `173bfdbe0ff676b4301cee05be60c30795a68772776cbd8565773d966fda0999`。n15 的两轮均无外部恢复，第二轮本地复用 11,008 tokens。设备仍有两笔 4,096 B 读。物理页占用分母为 1,048,576 个原始页，分子包括有效页和失效页。n15 本窗口新占用 586,780 页，等于编程页增量。探针经过交错覆写后，按各 line 块磨损算出的全盘擦除下界和上界同为 2,304。以上均为本次盘态与输入的观测。

## 原始证据和运行身份

- [48 MiB 原始目录](../../outputs/femu-n15-probe-20261120-074735/) 和 [3 GiB 原始目录](../../outputs/femu-n15-0504-20261120-074735/) 保存各自的 `launch.json`、`nand-report.json`、客体计数和原始 QMP 回复。0504 的 [verification-n15.json](../../outputs/femu-n15-0504-20261120-074735/verification-n15.json) 记录通过的两轮验证，[summary.json](../../outputs/femu-n15-0504-20261120-074735-analysis/summary.json) 记录对象及 I/O 分析。旧 TCG 的两份报告及其原始包在 [evidence/ssd-e](../../evidence/ssd-e/)。
- [独立归档](../../evidence/n15-kvm-20261120-074735/) 含上述三个目录的压缩包、实验前生成的 doctor、checkout 和镜像记录、事后来源清单及 SHA256SUMS。0504 原目录中两个受权限保护的文件未收入包内：`joint/client/.aiperf_results_ready.json` 和 `model-cache/modelinfos/vllm-model_executor-models-llama-LlamaForCausalLM.json`。后者的运行时 SHA-256 在 launch 中。所需原始证据、最终验证及两套 NAND 离线复算均完整。
- **启动时记录。** 项目 HEAD 为 `775c8ce6d2ec810dcc5f0ec60cab8b07b62ee4eb`。两个 launch 都对实际传给 QEMU 命令的文件计算了 SHA-256，均为 `3672aa21e9eb08b4a41f05b8cd243c29a199f5616bf2bfcbdfb0dcec3621e7a1`。声明的 FEMU revision 为 `a898d22b2f894042afe0f4ef21867393c5c78a70`，来自启动器固定值。启动脚本在运行前查询标签 `kv-ssd-n15:baseline-v2`，写下镜像 ID `sha256:9bab62b63e7271b4dc1625e411bdafd8a2c178c6639adde11553193d5751b700`。这是共享的 `outputs/container-image-id.txt`，原文件时间为上海时间 15:47:23，同标签 checkout 记录为 15:47:35。旧容器以 `--rm` 启动，没有该容器的 inspect 存档。
- **事后核对。** 归档时 HEAD 与运行记录相同，工作树已有未提交改动。归档的 `checkout-postrun.patch` 和 `launcher-postrun.sh` 反映归档时的内容；0504 的服务端源码清单逐文件与现工作树相符。项目目录可见的旧二进制、经校验的迁移包内二进制和两份运行时二进制哈希四者一致。实际宿主 mount 源按启动脚本和迁移包 SHA 推导，完整路径在 [provenance.json](../../evidence/n15-kvm-20261120-074735/provenance.json)。当前账号无法读取 root 独占的已安装 runtime 或 Docker socket，故未取得事后容器挂载实录或实时镜像 ID。
- **输入与几何。** 0504 前两轮 trace SHA-256 为 `d123b80995bcba44e7e69b0f3795e0783882e1e4794d838b55e53091f397bdaa`，暂存的配置和 trace 均已核对 launch 及客体收据。KVM、2 vCPU，探针客体 1 GiB，回放客体 8 GiB，22 块/rank，最大上下文 12,288，prefill/decode 10/2 ms，Store 2,520 MiB，offload 2,840 MiB，自然缓存、默认混放、stream 块采集。两盘均为 2 通道 × 4 LUN × 1 plane，每块 64 页，每页 4,096 B。探针可见 48 MiB、原始 64 MiB、32 条 line；回放可见 3 GiB、原始 4 GiB、2,048 条 line。

## 大盘构建前检查

启动器把 `.runtime/n15/runtime-<迁移包 SHA>/projects/kv-ssd-sim/.femu` 挂到容器的 `/home/plusyu/projects/kv-ssd-sim/.femu`，遮盖项目根下的同名目录。launch 的可执行文件为这个挂载点下的 `build/qemu-system-x86_64`。已核对迁移包 `/home/yzh/n15-kv-baseline/baseline.tar.gz`，其 SHA-256 为 `26c7905404149c94a72350ded5f6f25d66807e4e1d723df7e434c920639d6cfd`。遍历包目录，FEMU 源码路径只有 `pc-bios/`，工具链路径只有 `lib/`。宿主工作树的可见副本也如此。

核对了固定 revision 的两个上游源文件。BBSSD 的 `query-femu(kind=summary)` 已返回几何、line 分类数量、主机写页、直接 NAND 写页、GC 写页和累计擦除等字段。`ssd_query_collect()` 在 FTL 线程相邻请求之间取数。旧 summary 未请求 line 清单时，函数在逐 line 遍历之前返回；`line-counts.total` 取自几何管理状态，不能充当遍历覆盖数。现有 `kind=lines` 还提供每条的 `vpc`、`ipc`、`erase-min`、`erase-max`。大盘需要在同一回调中返回遍历覆盖数、有效/失效页和磨损统计，包括现行擦除验证用的逐 line 加权上下界。[QAPI 定义](https://github.com/MoatLab/FEMU/blob/a898d22b2f894042afe0f4ef21867393c5c78a70/qapi/femu.json) · [FTL 回调](https://github.com/MoatLab/FEMU/blob/a898d22b2f894042afe0f4ef21867393c5c78a70/hw/femu/bbssd/ftl-query.c#L306)

**继续所需资产及访问条件**

1. 可构建的 FEMU 工作树，含上述 revision 的 Git 元数据、`configure`、QAPI 生成器、`qapi/femu.json`、`hw/femu/` 和该构建依赖的子项目。原构建器期望 `.femu/FEMU-sparse/`。可从原机同步，或按固定 revision 检出到独立构建目录；上游读取的两个文件仅用于字段审阅。
2. 原构建器期望 `.femu/toolchain/bin/x86_64-conda-linux-gnu-{cc,c++}`、`bin/python`、头文件、pkg-config 和开发库。原机的 `.femu/toolchain-explicit.txt` 可用于重建相同工具链；迁移包中也缺少这份清单。已有构建脚本会写 `.femu/build/`，恢复资产后应先把输出指向单独目录，再把新二进制通过独立路径交给 launcher 并核对新 launch 的 SHA-256。[原构建脚本](../../scripts/build_femu.sh) · [原构建环境说明](../01-原型与闭环接入/FEMU-next.md)
3. 可运行基线容器的宿主账号或执行环境。本次宿主权限检查可见 `/dev/kvm` 和 `root:docker` 的 socket；当前 `yzh` 直接访问 Docker 被拒，非交互 sudo 需要密码。启动时的 doctor 记录了原执行环境的 KVM 可用状态。

**代码与验收状态。** 本轮新增证据归档、来源清单和本报告。FEMU 补丁、新二进制及其启动哈希尚未生成。48 MiB、3 GiB 的「新 summary 与旧 lines 在同一稳定边界等价」尚待实测。56 GiB 的目标为 `namespace_mib=57344`、`blocks_per_plane=38400`，每条 2 MiB，对应 75 GiB 原始 NAND。该试跑须在前两项等价性通过并检查当时的宿主可用内存和容器限额后进行。客体可见容量、实际覆盖数、采样耗时、峰值 RSS、换页及设备计数均尚无大盘测量值。

## 复核与复现

以下命令只在临时目录解包并重算。四个 `cmp` 都应成功，归档和历史包的哈希都应为 OK。

```bash
set -eu
cd /home/yzh/projects/kv-ssd-sim
a=evidence/n15-kvm-20261120-074735
(cd "$a" && sha256sum -c SHA256SUMS)
(cd evidence/ssd-e && sha256sum -c SHA256SUMS)
d=$(mktemp -d)
for f in "$a"/*.tar.gz evidence/ssd-e/*.tar.gz; do tar -xzf "$f" -C "$d"; done
for t in femu-n15-probe-20261120-074735 femu-n15-0504-20261120-074735 femu-e-probe03 femu-e-k3-default01; do
  cp "$d/$t/nand-report.json" "$d/$t.original.json"
  python3 scripts/analyze_nand.py "$d/$t"
  cmp "$d/$t/nand-report.json" "$d/$t.original.json"
done

PYTHONPATH=scripts:src python3 - "$d" <<'PY'
import json
import sys
from pathlib import Path
from analyze_k3_ssd import transfer_ledger
from verify_aiperf import read_jsonl

root = Path(sys.argv[1])
plan = json.loads(Path("configs/kimi-k3-b300-step2/resolved.json").read_text())
trial = "femu-n15-0504-20261120-074735"
summary = json.loads((root / (trial + "-analysis") / "summary.json").read_text())
expected = (208, 2402574336, "173bfdbe0ff676b4301cee05be60c30795a68772776cbd8565773d966fda0999")
for name in [trial, "femu-e-k3-default01"]:
    events = read_jsonl(root / name / "joint/server/events.jsonl")
    obj = transfer_ledger(events, plan)
    fields = ("unique_keys", "logical_bytes", "key_content_sha256")
    assert tuple(obj[k] for k in fields) == expected
    assert all(obj[k] == summary["runs"][0]["objects"][k] for k in fields)
    print(name, *(obj[k] for k in fields))
PY
```

在具备原 Docker/KVM 权限与原运行资产的 n15 宿主上，以下命令可用新时间标签复跑**旧基线**。启动器中记录了完整的探针、0504、验证及分析参数。归档目录中的副本固定了本次检查到的启动器版本。

```bash
cd /home/yzh/projects/kv-ssd-sim
bash scripts/run-n15-checkout.sh "$PWD" /home/yzh/n15-kv-baseline
```
