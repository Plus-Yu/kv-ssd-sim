"""Build a passive client observer using the n15 wheel's pinned types and ELF symbols."""

import argparse
import json
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path

from build_object_master import ARCHIVE_SHA, DSO_SHA, REV, ROOT, digest, verified

METHODS = {
    "ALLOCATE": "MasterClient::BatchPutStart(",
    "READY": "MasterClient::BatchPutEnd(",
    "SUBMIT": "BucketStorageBackend::BatchOffload(",
    "WRITE": "BucketStorageBackend::WriteBucket(",
    "PUBLISH": "MasterClient::NotifyOffloadSuccess(",
    "READ": "FileStorage::BatchLoad(",
    "READ_SPANS": "FileStorage::BatchLoadSpans(",
}


def build(archive, wheel, out):
    verified(archive, ARCHIVE_SHA)
    verified(wheel, DSO_SHA)
    out.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        tmp = Path(tmp)
        with tarfile.open(archive) as tar:
            tar.extractall(tmp, filter="data")
        src = tmp / f"Mooncake-{REV}"
        syms = subprocess.check_output(["nm", "-D", "--defined-only", str(wheel)], text=True)
        names = [s.split()[-1] for s in syms.splitlines() if s.strip()]
        pretty = subprocess.run(
            ["c++filt"], input="\n".join(names) + "\n", text=True, check=True, capture_output=True
        ).stdout.splitlines()
        definitions = {}
        for field, method in METHODS.items():
            matches = [
                a
                for a, b in zip(names, pretty, strict=True)
                if b.startswith("mooncake::" + method)
                and (field != "PUBLISH" or "OffloadTaskItem" in b)
            ]
            if len(matches) != 1:
                raise ValueError(f"Pinned method {method}: {matches}")
            definitions[field] = matches[0]
        (tmp / "observer_symbols.inc").write_text(
            "".join(f'#define OBS_{k} "{v}"\n' for k, v in definitions.items())
        )
        includes = [
            "mooncake-store/include",
            "mooncake-common/include",
            "mooncake-transfer-engine/include",
            "mooncake-transfer-engine/tent/include",
            "mooncake-store/include/cachelib_memory_allocator",
            "mooncake-store/include/cachelib_memory_allocator/include",
            "mooncake-store/include/cachelib_memory_allocator/fake_include",
        ]
        source = ROOT / "native/object_replay_observer.cpp"
        subprocess.run(
            [
                "g++-12",
                "-std=c++20",
                "-O1",
                "-DNDEBUG",
                "-fPIC",
                "-shared",
                "-pthread",
                *(f"-I{src / i}" for i in includes),
                f"-I{tmp}",
                str(source),
                "-ldl",
                "-o",
                str(tmp / "observer.so"),
            ],
            check=True,
        )
        so = tmp / "observer.so"
        # The observer is loaded before Python imports the native extension. Resolve its
        # non-inline type destructors from the same verified wheel.
        manifest = {
            "revision": REV,
            "wheel_dso_sha256": DSO_SHA,
            "source_sha256": digest(source),
            "builder_sha256": digest(Path(__file__)),
            "binary_sha256": digest(so),
            "symbols": definitions,
            "compiler": subprocess.check_output(["g++-12", "--version"], text=True).splitlines()[0],
        }
        (tmp / "observer.json").write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(so, out / "observer.so")
        os.replace(tmp / "observer.json", out / "observer.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument(
        "--wheel",
        type=Path,
        default=ROOT
        / ".venv/lib/python3.13/site-packages/mooncake/_store.cpython-313-x86_64-linux-gnu.so",
    )
    p.add_argument("--out", type=Path, default=ROOT / ".object-replay-build")
    args = p.parse_args()
    build(args.archive, args.wheel, args.out)
