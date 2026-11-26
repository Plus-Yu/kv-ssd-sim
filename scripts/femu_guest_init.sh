#!/bootstrap/busybox sh
# Runs only as PID 1 in the disposable FEMU VM. Never execute on the host.
bb=/bootstrap/busybox
[ "$$" = 1 ] || exit 90
fail() { echo "FEMU_GUEST_FAILED: $*"; $bb poweroff -f; exit 1; }
$bb mount -t proc proc /proc || fail proc
$bb mount -t sysfs sysfs /sys || fail sysfs
$bb mount -t devtmpfs devtmpfs /dev || fail dev
$bb mkdir -p /run /tmp /dev/shm
$bb mount -t tmpfs tmpfs /run
$bb mount -t tmpfs tmpfs /tmp
$bb mount -t tmpfs tmpfs /dev/shm
for module in virtio_pci 9pnet_virtio 9p nvme ext4; do
  $bb modprobe "$module" || fail "module $module"
done
# Host program/library exports are read-only; only this project is writable.
$bb mount -t 9p -o trans=virtio,version=9p2000.L,msize=262144,ro usr /usr || fail usr
$bb mkdir -p /home/plusyu/miniconda3 /home/plusyu/projects/kv-ssd-sim
$bb mount -t 9p -o trans=virtio,version=9p2000.L,msize=262144,ro conda /home/plusyu/miniconda3 || fail conda
$bb mount -t 9p -o trans=virtio,version=9p2000.L,msize=262144 project /home/plusyu/projects/kv-ssd-sim || fail project
$bb mkdir -p /lib64
# Alpine initramfs already supplies /lib -> usr/lib. Ubuntu's loader is here.
$bb ln -sf /usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2 /lib64/ld-linux-x86-64.so.2
export PATH=/usr/bin:/usr/sbin:/bin:/sbin
export HOME=/root
$bb mkdir -p /root /etc /mnt/femu
echo '127.0.0.1 localhost' > /etc/hosts
echo 'root:x:0:0:root:/root:/bin/sh' > /etc/passwd
/usr/sbin/ldconfig -X /usr/lib/x86_64-linux-gnu
$bb ip link set lo up
model=$($bb cat /sys/class/block/nvme0n1/device/model)
case "$model" in *FEMU*|*Femu*) ;; *) fail "unexpected NVMe model: $model" ;; esac
[ ! -e /dev/vda ] || fail 'unexpected persistent host disk'
# The launcher passes ONLY a new RAM-backed FEMU device, never a host block device.
echo disposable-femu-guest > /run/kv-ssd-femu-guest
project=/home/plusyu/projects/kv-ssd-sim
export PYTHONPATH="$project/src${PYTHONPATH:+:$PYTHONPATH}"
trial=$($bb sed -n 's/.*kvtrial=\([^ ]*\).*/\1/p' /proc/cmdline)
case "$trial" in femu-*) ;; *) fail 'invalid trial name' ;; esac
output="$project/outputs/$trial"
$bb mkdir -p "$output"
case " $($bb cat /proc/cmdline) " in
  *" kvnand=1 "*) export KV_SSD_NAND_STATS=1 ;;
esac
case " $($bb cat /proc/cmdline) " in
  *" kvworkload=nand-probe "*)
    cd "$project" || fail cd
    .venv/bin/python -m kv_ssd_sim.devices.femu.probe "$output" > "$output/replay.log" 2>&1
    result=$?
    echo "$result" > "$output/guest-exit-code.txt"
    $bb sync
    $bb poweroff -f
    exit "$result" ;;
esac
layout=$($bb sed -n 's/.*kvlayout=\([^ ]*\).*/\1/p' /proc/cmdline)
if [ -n "$layout" ]; then
  export KV_SSD_LAYOUT="$layout"
  /usr/sbin/mkfs.ext4 -q -F -U 12345678-0042-4000-8000-000000000001 -E lazy_itable_init=0,lazy_journal_init=0,hash_seed=12345678-0042-4000-8000-000000000002 /dev/nvme0n1 || fail format-layout
else
  /usr/sbin/mkfs.ext4 -q -F -E lazy_itable_init=0,lazy_journal_init=0 /dev/nvme0n1 || fail format
fi
$bb mount -t ext4 -o noatime /dev/nvme0n1 /mnt/femu || fail mount
$bb cat /proc/mounts > "$output/guest-mounts.txt"
$bb cat /proc/cpuinfo > "$output/guest-cpuinfo.txt"
$bb mount -t tracefs tracefs /sys/kernel/tracing || fail tracefs
if [ -n "$layout" ]; then
  /usr/sbin/dumpe2fs -g /dev/nvme0n1 > "$output/ext4-groups.txt" || fail groups
  /usr/sbin/dumpe2fs -h /dev/nvme0n1 > "$output/ext4-super.txt" || fail super
  /usr/sbin/debugfs -R 'blocks <8>' /dev/nvme0n1 > "$output/ext4-journal.txt" || fail journal
  inst=/sys/kernel/tracing/instances/kv_layout
  $bb mkdir "$inst" || fail ext4-instance
  echo mono > "$inst/trace_clock"
  echo 8192 > "$inst/buffer_size_kb"
  for event in ext4_allocate_blocks ext4_free_blocks; do
    echo 1 > "$inst/events/ext4/$event/enable" || fail "tracepoint $event"
    $bb cat "$inst/events/ext4/$event/format" > "$output/$event.format"
  done
  echo 1 > "$inst/tracing_on"
fi
echo mono > /sys/kernel/tracing/trace_clock
echo 8192 > /sys/kernel/tracing/buffer_size_kb
echo 1 > /sys/kernel/tracing/events/block/block_rq_issue/enable
echo 1 > /sys/kernel/tracing/events/block/block_rq_complete/enable
cd "$project" || fail cd
capture=$($bb sed -n 's/.*kvcapture=\([^ ]*\).*/\1/p' /proc/cmdline)
case "$capture" in
  stream|buffer)
    export KV_SSD_CAPTURE_ROOT="$output"
    .venv/bin/python -m kv_ssd_sim.devices.femu.block_capture "$capture" > "$output/capture.log" 2>&1 &
    collector=$!
    while [ ! -e /run/kv-ssd-capture-ready ]; do
      kill -0 "$collector" 2>/dev/null || fail collector-start
      $bb sleep 0.1
    done ;;
  '') echo 1 > /sys/kernel/tracing/tracing_on ;;
  *) fail capture-mode ;;
esac
echo FEMU_GUEST_REPLAY_BEGIN
export KV_SSD_DEBUG_STACK_SECONDS=120
export KV_SSD_METADATA_CACHE="$output/package-distributions.json"
export VLLM_CACHE_ROOT="$output/model-cache"
(
  while $bb sleep 30; do
    $bb ps > "$output/guest-processes.txt"
  done
) &
workload=$($bb sed -n 's/.*kvworkload=\([^ ]*\).*/\1/p' /proc/cmdline)
case "$workload" in
  k3-probe)
    .venv/bin/python scripts/verify_k3_shards.py --output "$output/probe" --guest-bundle "$output/input" > "$output/replay.log" 2>&1
    result=$?
    disk_name=k3-probe
    ;;
  aiperf)
    .venv/bin/python -m kv_ssd_sim.devices.femu.workload "$output" > "$output/replay.log" 2>&1
    result=$?
    disk_name=server
    ;;
  replay)
    .venv/bin/python -m kv_ssd_sim.devices.femu.workload "$output" > "$output/replay.log" 2>&1
    result=$?
    disk_name=replay
    ;;
  *) fail 'invalid workload' ;;
esac
if [ -n "$capture" ]; then
  $bb touch /run/kv-ssd-capture-stop
  wait "$collector" || fail collector-end
else
  echo 0 > /sys/kernel/tracing/tracing_on
  $bb cat /sys/kernel/tracing/trace > "$output/block-trace.txt"
fi
$bb cat /sys/class/block/nvme0n1/stat > "$output/guest-block-stat.txt"
if [ "$workload" = k3-probe ] || $bb grep -q k3-tp8 "$output/input/hybrid.json" 2>/dev/null; then
  $bb df -k /mnt/femu > "$output/guest-disk-usage.txt"
else
  $bb cp -r "/mnt/femu/$disk_name" "$output/disk-export"
fi
echo "$result" > "$output/guest-exit-code.txt"
$bb sync
echo "FEMU_GUEST_REPLAY_DONE exit=$result"
$bb poweroff -f
