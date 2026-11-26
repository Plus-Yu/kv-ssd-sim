# 阶段 E · 单盘 NAND 统计验收

## 实测结果

默认布局、真实尺寸 K3 的两轮 WEKA 回放已完成一次。首个 HTTP 实际发送至设备观测结束，主机写 **2,403,700,736 B**，NAND 编程 **2,403,700,736 B**；GC 搬迁 **0 B**，擦除 **0 次**，本窗口字节 WAF 为 **1.0000**。设备块提交、块完成、diskstats 三方写字节一致。

该客体为 8 GiB RAM，namespace 为 3 GiB，原始 NAND 为 4 GiB。ext4 格式化已在首个请求之前发生，基线快照有 29,292 个有效页、31 个失效页。原始 NAND 页占用由 **2.80%** 增至 **58.76%**，结束时有效页字节占 namespace 的 78.32%。两个边界从请求快照至收到回复的客体计时分别为 260.3 和 230.6 ms。运行使用 TCG，FEMU 提示 RAM 后端未锁页。只对这次初始盘态与两轮输入报告数值，长期填充、磨损或 GC 稳态需要另外观测。

### 受控空盘验收

校准 namespace 为 48 MiB，原始 NAND 为 64 MiB。先写单页，再覆盖同页 512 B 并读回验证；随后顺序写满 namespace，再进行三轮间隔覆盖，每隔 128 KiB 改写 64 KiB，以保留部分有效页供 GC 搬迁。下表均为相邻快照的增量。

| 窗口 | 主机写 B | NAND 编程 B | GC 搬迁 B | 擦除块数 | WAF |
|---|---:|---:|---:|---:|---:|
| 写一整页 | 4,096 | 4,096 | 0 | 0 | 1.0000 |
| 覆盖 512 B | 512 | 4,096 | 0 | 0 | 8.0000 |
| 只读一页 | 0 | 0 | 0 | 0 | 无定义 |
| 顺序填充 | 50,331,648 | 50,331,648 | 0 | 0 | 1.0000 |
| 交错覆写 | 75,497,472 | 603,971,584 | 528,474,112 | 2,304 | 7.9999 |

正控制结束时，按各 line 块磨损计算的擦除总数上下界同为 2,304，与设备擦除计数一致。探针中的交错覆写专门用来触发 GC，该 WAF 只用于验证计数链路。

[校准逐次快照](../../evidence/ssd-e/femu-e-probe03.json) · [默认 K3 的窗口与对账](../../evidence/ssd-e/femu-e-k3-default01.json) · [原始归档与校验和](../../evidence/ssd-e/)

## 采集与口径

在项目的默认单 namespace、页映射 FEMU BlackBox 上，启动时加 `--nand-stats` 可以同时记录客体块 I/O 和设备内部 NAND 计数。宿主通过私有 Unix socket 请求 `query-femu(nsid=1, kind=lines, limit=4096)`。每个边界只执行一次 FTL 查询，返回计数、几何及所有 line 的页状态。每个 line 是跨多个物理擦除块的分配单位。当前采集支持一次查询可容纳的至多 4,096 个 line。控制程序、QMP 原始回复和时间标记均写到被测盘以外。使用项目已固定的 FEMU revision `a898d22b2f894042afe0f4ef21867393c5c78a70`。[启动器](../../scripts/run_femu.py) · [宿主 QMP 采集](../../src/kv_ssd_sim/devices/femu/qmp.py) · [客体采样](../../src/kv_ssd_sim/devices/femu/nand.py)

| 统计量 | 计算方法和单位 |
|---|---|
| 主机读、写字节 | 两个采样边界之间 nvme0n1 的读、写完成扇区差分别乘 512 B。在线回放另与块层 issue 和 complete 字节核对。 |
| NAND 编程字节 | 设备报告的直接数据编程页、GC 搬迁页、映射页、填充页和校验页增量之和乘实测 NAND 页大小。当前配置每页 4,096 B。 |
| GC 搬迁字节 | `Δgc-write-pages × page-size`，为总编程量的一部分。 |
| 擦除次数 | `Δblock-erases`，单位为物理块。每个边界还保留所有 line 的最小/最大块磨损，以各 line 包含的擦除块数加权，独立检查全盘擦除总数的上下界。 |
| 字节写放大 WAF | 同一窗口的 NAND 编程字节 / 主机写入字节。主机写入量为零时输出 JSON `null`。FEMU 自报的累计 `waf` 用主机页数作分母，仅保留在原始回复中。 |
| 物理页占用率 | 所有 line 的有效页与失效但尚未擦除的页数之和 / NAND 总页数。另报有效页字节 / 主机可见 namespace 容量；两个分母在存在预留空间时不同。ext4 文件系统使用率由原有容量采样记录。 |

默认页映射几何下，额外核对窗口内的页数守恒，即 **已占用页数变化 = 编程页数 − 擦除块数 × 每块页数**。验收只接受完整 line 清单，且同一次查询前后 diskstats 无新增 I/O、无在途。自然回放在首个 HTTP 实际发送前和采集结束时采两次，核对查询边界与实际窗口之间也没有目标盘块事件。中间正常处理请求，无逐 token 的 QMP 查询。每个快照保留客体单调时钟区间，JSON 中的 `after_ns - before_ns` 包含跨客体通知和文件交换时间。[增量计算与验收代码](../../scripts/analyze_nand.py)

## 复现

在仓库根目录运行。每次启动使用新的 trial 名。校准模式在新的内存后端运行，打开专用的 48 MiB 未挂载 namespace，客体使用同步直接 I/O，输出快照保存到项目目录。

```bash
.venv/bin/python scripts/run_femu.py --workload nand-probe --trial femu-e-calibration-repro \
  --guest-ram-mib 1024 --namespace-mib 48 --blocks-per-plane 32 --timeout 240
.venv/bin/python scripts/run_femu.py --trial femu-e-weka-repro --nand-stats \
  --workload aiperf --natural-cache --block-capture stream \
  --input datasets/k3-c2/slice.json --hybrid-config configs/k3-c1-online.json \
  --guest-ram-mib 8192 --namespace-mib 3072 --blocks-per-plane 2048 \
  --max-model-len 12288 --num-blocks 22 --timeout 2400
.venv/bin/python scripts/analyze_nand.py outputs/femu-e-weka-repro
```

每次成功的 trial 生成 `nand-report.json`。其中 `windows` 是区间增量，`samples` 是各边界的累计计数、页占用和时间区间。对应的 `nand-*.qmp.json` 是原始 FTL 回复，`nand-*.guest.json` 是客体前后计数。

归档已分别解包并离线重算，两份报告与归档的 JSON 逐字节一致。K3 原始归档也通过已有的 HTTP、自然缓存、逐边界块计数验收。项目回归 92 项通过、6 项跳过。可在项目根目录复核已采集的数据。

```bash
(cd evidence/ssd-e && sha256sum -c SHA256SUMS)
d=$(mktemp -d)
for a in evidence/ssd-e/*.tar.gz; do tar -xzf "$a" -C "$d"; done
for t in femu-e-probe03 femu-e-k3-default01; do
  .venv/bin/python scripts/analyze_nand.py "$d/$t"
  cmp "$d/$t/nand-report.json" "evidence/ssd-e/$t.json"
done
```
