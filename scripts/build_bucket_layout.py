"""Build the opt-in layout adapter against the installed Mooncake CPU wheel."""

import hashlib
import json
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if version("mooncake-transfer-engine-non-cuda") != "0.3.14":
    raise RuntimeError("The native bucket adapter targets Mooncake CPU 0.3.14")
wheel = next((ROOT / ".venv/lib/python3.13/site-packages/mooncake").glob("_store.*.so"))
source = ROOT / "native/bucket_layout.cpp"
out = ROOT / ".native/layout"
out.mkdir(parents=True, exist_ok=True)
compiler = ROOT / ".femu/toolchain/bin/x86_64-conda-linux-gnu-g++"
if not compiler.is_file():
    compiler = shutil.which("c++")
header = Path(sys.base_prefix) / "include/tl/expected.hpp"
if not compiler or not header.is_file():
    raise RuntimeError(f"Building the bucket adapter requires a C++ compiler and {header}")
target = out / "libbucketlayout.so"
subprocess.run(
    [
        str(compiler),
        "-std=c++17",
        "-O2",
        "-shared",
        "-fPIC",
        "-I" + str(header.parents[1]),
        str(source),
        "-Wl,--no-as-needed",
        "-L" + str(wheel.parent),
        "-l:" + wheel.name,
        "-Wl,-rpath," + str(wheel.parent),
        "-o",
        str(target),
    ],
    check=True,
)
(out / "build.json").write_text(
    json.dumps(
        {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (source, wheel, header, target)
        },
        indent=2,
    )
    + "\n"
)
