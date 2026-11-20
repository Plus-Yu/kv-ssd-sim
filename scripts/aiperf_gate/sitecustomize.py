"""Opt-in pre-send control handshake for the pinned AIPerf 0.13 worker."""

import os
import sys
from importlib.machinery import PathFinder


class GateFinder:
    def find_spec(self, fullname, path, target=None):
        if fullname != "aiperf.workers.inference_client":
            return None
        spec = PathFinder.find_spec(fullname, path)
        original_exec = spec.loader.exec_module

        def install(module):
            import asyncio
            import functools
            import json
            import time
            import urllib.request

            original_exec(module)
            send = module.InferenceClient.send_request

            def prepare(request_id):
                request = urllib.request.Request(
                    os.environ["KV_SSD_GATE_URL"],
                    data=b"",
                    headers={"X-Request-ID": request_id},
                )
                with urllib.request.urlopen(request, timeout=90) as response:
                    if json.load(response)["ready_for"] != request_id:
                        raise RuntimeError("Wrong cold-preparation acknowledgement")

            @functools.wraps(send)
            async def gated(self, request_info, *args, **kwargs):
                if os.environ.get("KV_SSD_CAPTURE_ROOT"):
                    from kv_ssd_sim.block_capture import mark

                    mark("http_send", client_id=request_info.x_request_id)
                if not os.environ.get("KV_SSD_GATE_URL"):
                    return await send(self, request_info, *args, **kwargs)
                await asyncio.to_thread(prepare, request_info.x_request_id)
                with open(os.environ["KV_SSD_GATE_LOG"], "a") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "request_id": request_info.x_request_id,
                                "before_native_send_ns": time.monotonic_ns(),
                            }
                        )
                        + "\n"
                    )
                return await send(self, request_info, *args, **kwargs)

            module.InferenceClient.send_request = gated

        spec.loader.exec_module = install
        return spec


if os.environ.get("KV_SSD_GATE_URL") or os.environ.get("KV_SSD_CAPTURE_ROOT"):
    sys.meta_path.insert(0, GateFinder())
