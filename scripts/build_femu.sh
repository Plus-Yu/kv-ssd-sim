#!/usr/bin/env bash
set -euo pipefail
project_dir=$(cd "$(dirname "$0")/.." && pwd)
cd "$project_dir"
revision=a898d22b2f894042afe0f4ef21867393c5c78a70
toolchain="$project_dir/.femu/toolchain"
source_dir="$project_dir/.femu/FEMU-sparse"
if [ ! -x "$toolchain/bin/x86_64-conda-linux-gnu-cc" ]; then
  echo "Missing private toolchain. See docs/FEMU-next.md for installation."
  exit 1
fi
actual_revision=$(git -C "$source_dir" rev-parse HEAD)
if [ "$actual_revision" != "$revision" ]; then
  echo "FEMU revision mismatch: $actual_revision"
  exit 1
fi
export PATH="$toolchain/bin:$PATH"
export CC="$toolchain/bin/x86_64-conda-linux-gnu-cc"
export CXX="$toolchain/bin/x86_64-conda-linux-gnu-c++"
export PKG_CONFIG_PATH="$toolchain/lib/pkgconfig:$toolchain/share/pkgconfig"
export LD_LIBRARY_PATH="$toolchain/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p .femu/build
cd .femu/build
"$source_dir/configure" --target-list=x86_64-softmmu --enable-kvm --enable-tcg \
  --enable-virtfs --enable-slirp \
  --disable-werror --disable-docs --disable-gtk --disable-sdl --disable-vnc \
  --disable-libnfs --disable-libiscsi --disable-curl --disable-tools \
  --extra-cflags="-I$toolchain/include" \
  --extra-ldflags="-L$toolchain/lib -Wl,-rpath,$toolchain/lib" \
  --python="$toolchain/bin/python"
ninja -j "${FEMU_BUILD_JOBS:-4}" qemu-system-x86_64
./qemu-system-x86_64 -device femu,help
