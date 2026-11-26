"""Fetch and hash the pinned public WEKA replay before installing it in the local cache."""

import argparse
import hashlib
import json
import os
import ssl
import tarfile
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

from kv_ssd_sim.workload.corpus import REVISION, SHA256

NAME = "replay-40d-oct14-337a03810637"
BLOB = "9f9476958fe8a8dda14f22795bc7de3711147b42cf3b65e8f4a5e2c832f21995"
SIZE = 8816390440
LAYER_SIZE = 1312904388


def verified_copy(stream, path, expected, size):
    h = hashlib.sha256()
    with path.open("xb") as out:
        while chunk := stream.read(4 << 20):
            h.update(chunk)
            out.write(chunk)
    if path.stat().st_size != size or h.hexdigest() != expected:
        raise ValueError(f"Wrong length or SHA-256 for {path.name}")


def fetch(cache, provider, ca):
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / (NAME + ".jsonl")
    if dest.exists():
        with dest.open("rb") as stream:
            if (
                dest.stat().st_size != SIZE
                or hashlib.file_digest(stream, "sha256").hexdigest() != SHA256
            ):
                raise ValueError(f"Cached file differs: {dest}")
        return dest
    ssl_context = ssl.create_default_context(cafile=ca)

    def get(url, headers=None):
        request = Request(url)
        for key, value in (headers or {}).items():
            request.add_unredirected_header(key, value)
        return urlopen(request, timeout=120, context=ssl_context)

    with tempfile.TemporaryDirectory(dir=cache) as scratch:
        temp = Path(scratch) / dest.name
        if provider == "hf":
            with get(
                f"https://huggingface.co/datasets/wekaio/{NAME}/resolve/{REVISION}/{dest.name}"
            ) as stream:
                verified_copy(stream, temp, SHA256, SIZE)
        else:
            with get(
                "https://quay.io/v2/auth?service=quay.io&scope=repository:weka.io/kvlab:pull"
            ) as stream:
                token = json.load(stream)["token"]
            layer = Path(scratch) / "layer.tgz"
            with get(
                f"https://quay.io/v2/weka.io/kvlab/blobs/sha256:{BLOB}",
                {"Authorization": f"Bearer {token}"},
            ) as stream:
                verified_copy(stream, layer, BLOB, LAYER_SIZE)
            with tarfile.open(layer, "r|gz") as tar:
                for member in tar:
                    if member.isfile() and member.name.endswith(".jsonl") and member.size == SIZE:
                        with tar.extractfile(member) as stream:
                            verified_copy(stream, temp, SHA256, SIZE)
                        break
                else:
                    raise ValueError("Verified image layer is missing the pinned replay")
        os.link(temp, dest)
    return dest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=Path(".object-replay-build/kvlab-source"))
    p.add_argument("--provider", choices=("quay", "hf"), default="quay")
    p.add_argument("--ca", help="Optional PEM CA bundle")
    a = p.parse_args()
    print(fetch(a.cache, a.provider, a.ca))
