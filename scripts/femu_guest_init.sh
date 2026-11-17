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
/usr/sbin/mkfs.ext4 -q -F -E lazy_itable_init=0,lazy_journal_init=0 /dev/nvme0n1 || fail format
$bb mount -t ext4 -o noatime /dev/nvme0n1 /mnt/femu || fail mount
echo disposable-femu-guest > /run/kv-ssd-femu-guest
project=/home/plusyu/projects/kv-ssd-sim
trial=$($bb sed -n 's/.*kvtrial=\([^ ]*\).*/\1/p' /proc/cmdline)
case "$trial" in femu-*) ;; *) fail 'invalid trial name' ;; esac
output="$project/outputs/$trial"
$bb mkdir -p "$output"
$bb cat /proc/mounts > "$output/guest-mounts.txt"
$bb cat /proc/cpuinfo > "$output/guest-cpuinfo.txt"
$bb mount -t tracefs tracefs /sys/kernel/tracing || fail tracefs
echo mono > /sys/kernel/tracing/trace_clock
echo 8192 > /sys/kernel/tracing/buffer_size_kb
echo 1 > /sys/kernel/tracing/events/block/block_rq_issue/enable
echo 1 > /sys/kernel/tracing/events/block/block_rq_complete/enable
echo 1 > /sys/kernel/tracing/tracing_on
cd "$project" || fail cd
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
  aiperf)
    .venv/bin/python scripts/run_aiperf.py --input examples/aiperf-two-requests.weka.json \
      --output "$output/joint" --cold-between-requests --femu-mount /mnt/femu \
      --startup-timeout 900 --timeout 900 > "$output/replay.log" 2>&1
    result=$?
    disk_name=server
    ;;
  replay)
    .venv/bin/python -m kv_ssd_sim replay examples/two_requests.weka.json \
      --output "$output/replay" --hybrid-config configs/hybrid-tiny.json \
      --cold-between-turns --femu-mount /mnt/femu > "$output/replay.log" 2>&1
    result=$?
    disk_name=replay
    ;;
  *) fail 'invalid workload' ;;
esac
echo 0 > /sys/kernel/tracing/tracing_on
$bb cat /sys/kernel/tracing/trace > "$output/block-trace.txt"
$bb cat /sys/class/block/nvme0n1/stat > "$output/guest-block-stat.txt"
$bb cp -r "/mnt/femu/$disk_name" "$output/disk-export"
echo "$result" > "$output/guest-exit-code.txt"
$bb sync
echo "FEMU_GUEST_REPLAY_DONE exit=$result"
$bb poweroff -f
