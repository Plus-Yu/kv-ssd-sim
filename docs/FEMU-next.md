# FEMU 接入与复现

本项目已完成实际 FEMU 在线闭环验收，结果见 [FEMU 验收报告](FEMU验收报告.md)。运行结果以 `outputs/femu-*/verification.json` 为准；没有这个通过记录时，不应把普通文件 offload 或 receiver sleep 称为 FEMU 验证。

2026-11-17 已增加 `--workload aiperf`，在客体内启动 AIPerf、HTTP 服务和 Mooncake。两组联合功能链路通过，包含客户端 TTFT 的严格对照未通过，已定位到人工冷缓存准备的计时干扰。实测结果和复现命令见 [联合验收报告](AIPerf-FEMU联合验收报告.md)。以下默认命令仍使用直接回放路径。

## 已实现的路径

```text
WEKA → vLLM 原生 Scheduler / 混合 KVCacheManager
     → 固定计算预算 + CPU MLA/KDA 占位状态
     → 原生 MooncakeStoreConnector / embedded real client
     → 客体 ext4 / FEMU BlackBox NVMe / NAND + FTL
     → 原生 GET 完成 → scheduler 恢复 → token 输出
```

vLLM、Mooncake master 和持有 offload 数据的 real client 都在同一个 Linux 客体中。`--femu-mount /mnt/femu` 把 `output/disk` 链接到 NVMe 的独立目录。传给 native Store 的是解析后的实际目录，避免其拒绝符号链接。设备从 sysfs 核对，读回期间记录块设备计数。冷恢复前依次确认磁盘副本、清除 Store 内存副本、重置 vLLM 本地前缀缓存、同步并清除一次性客体页缓存。

`femu_guest_init.sh` 捕获 `block_rq_issue` / `block_rq_complete`，与应用 monotonic 时钟对齐。`verify_femu.py` 要求在原生 GET 窗口内看到读完成，且 GET 完成先于 load_complete、请求重新计算和首 token。

## 本机环境

- 没有 GPU、没有 `/dev/kvm`；使用 TCG 软件虚拟化做功能验证。
- FEMU 固定提交：`a898d22b2f894042afe0f4ef21867393c5c78a70`。
- 私有编译器及依赖：`.femu/toolchain`，确切包版本在 `.femu/toolchain-explicit.txt`。
- 源码：`.femu/FEMU-sparse`，跳过不参与本构建的 `roms/` 源码；运行所需预编译 BIOS 在 `pc-bios/`。
- 二进制：`.femu/build/qemu-system-x86_64`。
- 客体：Alpine v3.23 virt 内核及模块；复用宿主 `/usr` 和 conda 库，只读 9p 导出；项目目录可写。镜像及依赖 SHA256 在 `.femu/assets-manifest.json`。
- launcher 目前面向本机 `/home/plusyu` 的安装路径。迁移机器需准备相同用户态运行库或改成独立客体镜像。

## 运行

```bash
# 编译完成后才运行；每次使用新 trial 名。
.venv/bin/python scripts/run_femu.py --trial femu-new-fast --nand-read-ns 40000
.venv/bin/python scripts/run_femu.py --trial femu-new-slow --nand-read-ns 100000000
.venv/bin/python scripts/compare_femu.py outputs/femu-new-fast outputs/femu-new-slow \
  --output outputs/femu-new-comparison.json
```

默认每次新建仅存在于 QEMU 内存的 FEMU NVMe，不附加任何宿主物理磁盘或磁盘映像，不开放客体网络。格式化只发生在这个一次性客体的 FEMU 设备内，脚本检查 PID 1、设备型号和不存在 `/dev/vda`。不要把客体 init 脚本作为宿主初始化脚本执行。

NAND 几何：512 B sector，8 sectors/page，64 pages/block，256 blocks/plane，1 plane/LUN，4 LUNs/channel，2 channels；物理 NAND 512 MiB，向客体暴露 384 MiB。写 200 μs，擦除 2 ms；只改变 `pg_rd_lat`。默认推理预算仍为 prefill 10 ms / decode 2 ms，receiver 延迟注入必须为 0。

输出包括 `launch.json`、客体 mount/cpu 信息、串口日志、原生回放 manifest/events/summary、块 trace、退出码及验证结果。运行会在关闭 trace 后导出 offload 文件到 `disk-export/`；客体关闭后的原始 `disk` 符号链接在宿主不可用。

## 重建编译环境

```bash
conda create -y -p "$PWD/.femu/toolchain" --override-channels -c conda-forge \
  gcc_linux-64 gxx_linux-64 make ninja pkg-config glib pixman libslirp libaio \
  python=3.12 meson flex bison zlib
# 或使用已经保存的 toolchain-explicit.txt 固定所有包。
git clone --depth 1 --filter=blob:none --no-checkout https://github.com/MoatLab/FEMU.git .femu/FEMU-sparse
git -C .femu/FEMU-sparse config fetch.negotiationAlgorithm noop
git -C .femu/FEMU-sparse fetch --depth 1 --filter=blob:none origin a898d22b2f894042afe0f4ef21867393c5c78a70
git -C .femu/FEMU-sparse sparse-checkout set --no-cone '/*' '!/roms/'
git -C .femu/FEMU-sparse checkout a898d22b2f894042afe0f4ef21867393c5c78a70
bash scripts/build_femu.sh
```

客体依赖下载自 Alpine 官方 netboot 目录的 `vmlinuz-virt`、`initramfs-virt`、`modloop-virt`，分别存为 `.femu/alpine-vmlinuz`、`alpine-initramfs`、`alpine-modloop`。Ubuntu 的 `busybox-static`、`squashfs-tools` 仅用 `apt-get download` 下载，再以 `dpkg-deb -x` 解包到 `.femu/guest-tools`；不做全局安装。使用解包后的 `unsquashfs -d .femu/modloop .femu/alpine-modloop` 提取模块。

## 能回答和不能回答的问题

这套脚本用于验证模拟介质完成事件是否真正回到活跃的 scheduler。TCG 的 CPU 执行开销、9p 日志写入、未锁页的 FEMU 后端都可能引入抖动；应报告实际计算窗口与超预算时间。未做硬件标定时，不能用绝对 TTFT 预测真实 GPU 服务性能。

尚未完成逐对象 → bucket 文件偏移 → LBA 的一一映射，也未验证跨 TP/GPU/RDMA、真实模型计算、长期 GC 压力和 SSD 寿命。混合功能与 Mooncake 故障边界见 [混合注意力验收报告](混合注意力验收报告.md)。

上游参考：[FEMU](https://github.com/MoatLab/FEMU)、[Mooncake SSD offload](https://kvcache-ai.github.io/Mooncake/design/store/ssd-offload.html)。
