"""Build the metadata driver against the exact n15 Mooncake wheel and upstream headers."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REV = "e1ea5d31bb5e46ff2df9f90e7c2fbaf52fcdee8c"
ARCHIVE_SHA = "482df63a230655e2210789179110a6bcf581360cb577acda5b941939d33709a7"
DSO_SHA = "19573f3ed46c577e52e62851667266bd3cc8d40781300ae22ce8a73a624814a8"
MASTER_SHA = "0985bbe1e3662e2e0d935c08101322b88b654bcaf81eec925d6f5a8d01ecbe92"
URL = f"https://codeload.github.com/kvcache-ai/Mooncake/tar.gz/{REV}"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verified(path, expected):
    if digest(path) != expected:
        raise ValueError(f"SHA-256 mismatch for {path}. Expected {expected}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Previously downloaded, pinned source tarball")
    parser.add_argument(
        "--wheel-dir", type=Path, default=ROOT / ".venv/lib/python3.13/site-packages/mooncake"
    )
    parser.add_argument(
        "--libpython", type=Path, default=Path(sys.base_prefix) / "lib/libpython3.13.so"
    )
    parser.add_argument("--out", type=Path, default=ROOT / ".object-replay-build")
    parser.add_argument("--cxx", default="g++-12")
    args = parser.parse_args()
    wheel = args.wheel_dir.resolve()
    lib = wheel / "_store.cpython-313-x86_64-linux-gnu.so"
    verified(lib, DSO_SHA)
    verified(wheel / "mooncake_master", MASTER_SHA)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    driver = ROOT / "native/object_replay_master.cpp"
    with tempfile.TemporaryDirectory(dir=out) as scratch:
        temp = Path(scratch)
        archive = args.archive
        if archive is None:
            archive = temp / "source.tar.gz"
            urllib.request.urlretrieve(URL, archive)
        verified(archive, ARCHIVE_SHA)
        with tarfile.open(archive) as tar:
            tar.extractall(temp, filter="data")
        src = temp / f"Mooncake-{REV}"
        master_cpp = src / "mooncake-store/src/master_service.cpp"
        worker = master_cpp.read_text().split("void MasterService::EvictionThreadFunc()", 1)[1]
        first = worker.index("double evict_ratio_target =")
        last = "BatchEvict(evict_ratio_target, evict_ratio_lowerbound);"
        calculation = worker[first : worker.index(last, first) + len(last)]
        (temp / "native_round.inc").write_text(calculation + "\n")
        # Give the metadata fixture access to the native bucket index without
        # changing its layout or rebuilding the wheel's decision function.
        bucket_header = src / "mooncake-store/include/storage_backend.h"
        declaration = "class BucketStorageBackend : public StorageBackendInterface {"
        header = bucket_header.read_text()
        if header.count(declaration) != 1:
            raise ValueError("Pinned bucket header does not match")
        bucket_header.write_text(
            header.replace(declaration, declaration + "\n    friend class ReplayBucketIndex;")
        )
        includes = (
            "mooncake-store/include",
            "mooncake-common/include",
            "mooncake-transfer-engine/include",
            "mooncake-transfer-engine/tent/include",
            "mooncake-store/include/cachelib_memory_allocator",
            "mooncake-store/include/cachelib_memory_allocator/include",
            "mooncake-store/include/cachelib_memory_allocator/fake_include",
        )
        # The wheel exports the production MasterService. Only the driver is compiled.
        # The absolute DSO dependency and the hash checks bind it to the selected wheel.
        libpython = args.libpython.resolve(strict=True)
        binary = temp / "master"
        subprocess.run(
            [
                args.cxx,
                "-std=c++20",
                "-O1",
                "-DNDEBUG",
                "-pthread",
                "-Wl,--export-dynamic",
                *(f"-I{src / path}" for path in includes),
                f"-I{temp}",
                str(driver),
                str(lib),
                "-Wl,--no-as-needed",
                str(libpython),
                "-Wl,--as-needed",
                f"-Wl,-rpath,{libpython.parent}",
                "-o",
                str(binary),
            ],
            check=True,
        )
        manifest = {
            "backend": "mooncake.MasterService.BatchEvict",
            "mooncake_version": "0.3.14",
            "revision": REV,
            "archive_url": URL,
            "archive_sha256": ARCHIVE_SHA,
            "wheel_dso": str(lib),
            "wheel_dso_sha256": DSO_SHA,
            "online_master_sha256": MASTER_SHA,
            "master_source_sha256": digest(master_cpp),
            "driver_source_sha256": digest(driver),
            "builder_sha256": digest(Path(__file__)),
            "executable_sha256": digest(binary),
            "compiler": subprocess.check_output([args.cxx, "--version"], text=True).splitlines()[0],
        }
        (temp / "build.json").write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(binary, out / "master")
        os.replace(temp / "build.json", out / "build.json")
    print(out / "build.json")


if __name__ == "__main__":
    main()
