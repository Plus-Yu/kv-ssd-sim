#!/usr/bin/env python3
"""Download an anonymized public trace and select its first two parent turns."""

import argparse
import hashlib
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

URL = "https://raw.githubusercontent.com/callanjfox/kv-cache-tester/master/traces/trace_0001.json"
SOURCE_SHA256 = "cfd45b93c58e16b1ae3cb177f91d45bb58590dc592e140e28e4155145746feb3"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-file", type=Path, help="Use an already downloaded, hash-matching original trace"
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.source_file:
        raw = args.source_file.read_bytes()
    else:
        with urllib.request.urlopen(URL, timeout=60) as response:
            raw = response.read()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != SOURCE_SHA256:
        raise RuntimeError("Public source changed; inspect it before updating the pinned hash")
    trace = json.loads(raw)
    trace["requests"] = trace["requests"][:2]
    if len(trace["requests"]) != 2 or any(r["type"] not in {"n", "s"} for r in trace["requests"]):
        raise ValueError("Expected the two validated parent requests")
    trace.pop("totals", None)
    trace["prototype_slice"] = {
        "source": URL,
        "source_sha256": actual,
        "selection": "first two parent requests; lengths, hashes and timing unchanged",
        "prepared_at": datetime.now(UTC).isoformat(),
        "source_license": "Apache-2.0; https://github.com/callanjfox/kv-cache-tester/blob/master/LICENSE",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(trace, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "source_sha256": actual, "requests": 2}))


if __name__ == "__main__":
    main()
