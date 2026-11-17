"""Loopback HTTP ingress; every generated token follows a native scheduler step.

This is a synthetic inference endpoint, not vLLM's production HTTP server.
AIPerf owns trace timing/dependencies; this server owns admission and execution.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
import time
import uuid
from contextlib import ExitStack
from pathlib import Path

from aiohttp import web

from .engine import ReplayEngine
from .events import EventLog
from .hybrid import HybridConfig
from .native import NativeRuntime
from .payload import MAX_PREPARED_BYTES, SYNTHETIC_TOKEN, PreparedPayloads
from .runtime import mooncake_master
from .trace import ReplayRequest


def parse_chat(body, tokenizer, *, model, token_limit):
    """Validate before HTTP headers or native admission; never truncate prompts."""
    if not isinstance(body, dict) or body.get("model") != model:
        raise ValueError(f"model must be {model!r}")
    if body.get("n", 1) != 1 or body.get("tools") or body.get("tool_choice"):
        raise ValueError("Only n=1 plain text chat is supported")
    stream = body.get("stream", False)
    if not isinstance(stream, bool):
        raise TypeError("stream must be boolean")
    if not isinstance(body.get("stream_options", {}), dict):
        raise TypeError("stream_options must be an object")
    count = body.get("max_completion_tokens", body.get("max_tokens"))
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("An explicit positive max_tokens or max_completion_tokens is required")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list")
    normalized = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {
            "system",
            "user",
            "assistant",
            "tool",
            "developer",
        }:
            raise ValueError("Invalid text message")
        content = message.get("content")
        if isinstance(content, list):
            if any(
                not isinstance(p, dict)
                or p.get("type") != "text"
                or not isinstance(p.get("text"), str)
                for p in content
            ):
                raise ValueError("Only text content parts are supported")
            content = "".join(p["text"] for p in content)
        if not isinstance(content, str) or message.get("tool_calls"):
            raise ValueError("Only text content is supported")
        normalized.append({"role": message["role"], "content": content})
    tokens = tokenizer.apply_chat_template(
        normalized, tokenize=True, add_generation_prompt=True, return_dict=False
    )
    if not tokens or len(tokens) + count > token_limit:
        raise ValueError(f"Prompt plus output must fit {token_limit} tokens")
    if any(t < 4 or t >= 32000 for t in tokens):
        raise ValueError("Prompt contains special, unknown, or out-of-vocabulary token IDs")
    return tuple(tokens), count, stream


@dataclasses.dataclass
class Job:
    spec: ReplayRequest
    chunks: asyncio.Queue = dataclasses.field(default_factory=asyncio.Queue)
    emitted: int = 0


class ServingLoop:
    def __init__(self, runtime, events, *, prefill_ms, decode_ms, cold, max_inflight):
        self.runtime, self.events = runtime, events
        self.engine = ReplayEngine(
            runtime,
            [],
            events,
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            timing="http",
            cold=False,
        )
        self.cold = cold
        self.max_inflight = max_inflight
        self.jobs = {}
        self.pending = asyncio.Queue()
        self.reserved = 0
        self.reserved_bytes = 0
        self.last_cold_count = 0
        self.failure = None
        self.closing = False
        self.task = None

    async def submit(self, tokens, count, received_id):
        if self.failure or self.closing:
            raise web.HTTPServiceUnavailable(text="Native serving loop is unavailable")
        size = (len(tokens) + count) * len(self.runtime.caches) * 32
        if self.reserved >= self.max_inflight or self.reserved_bytes + size > MAX_PREPARED_BYTES:
            raise web.HTTPTooManyRequests(text="Admission/payload capacity reached; no truncation")
        self.reserved += 1
        self.reserved_bytes += size
        rid = f"http-{uuid.uuid4().hex}"
        spec = ReplayRequest(rid, "http", "http", 0, 0, (), tokens, count, "http")
        started = time.perf_counter()
        try:
            prepared = await asyncio.to_thread(
                PreparedPayloads, [spec], num_layers=len(self.runtime.caches)
            )
            if self.failure or self.closing:
                raise web.HTTPServiceUnavailable(
                    text="Native serving loop stopped during preparation"
                )
            self.engine.add_request(spec)
            self.runtime.payloads.patterns.update(prepared.patterns)
            self.runtime.payloads.nbytes += prepared.nbytes
            job = Job(spec)
            self.jobs[rid] = job
            # Prefix digests allow auditing shared blocks without logging prompt text.
            digest = hashlib.sha256()
            blocks = []
            for position, token in enumerate(tokens, 1):
                digest.update(token.to_bytes(4, "big"))
                if position % self.runtime.block_size == 0:
                    blocks.append(digest.hexdigest())
            self.events.emit(
                "http_received",
                request_id=rid,
                client_request_id=received_id,
                input_tokens=len(tokens),
                target_output_tokens=count,
                prompt_sha256=digest.hexdigest(),
                prefix_block_sha256=blocks,
                preparation_ms=(time.perf_counter() - started) * 1000,
                prepared_bytes=size,
            )
            self.pending.put_nowait(job)
            return job
        except BaseException:
            self.reserved -= 1
            self.reserved_bytes -= size
            raise

    async def make_cold(self):
        await self.engine.drain()
        keys = await asyncio.to_thread(self.runtime.store.disk_barrier)
        await asyncio.to_thread(self.runtime.store.evict_memory_replicas, keys)
        if not self.runtime.scheduler.kv_cache_manager.reset_prefix_cache():
            raise RuntimeError("Local prefix blocks remain in use")
        self.events.emit("local_prefix_cache_reset", reason="explicit HTTP cold validation")
        if self.runtime.store.femu_device:
            await asyncio.to_thread(self.runtime.store.femu_device.clear_guest_page_cache)
            self.events.emit("guest_page_cache_dropped")

    async def run(self):
        try:
            while not self.closing or self.jobs:
                engine = self.engine
                active = len(engine.admitted) - len(engine.completed)
                if (
                    self.cold
                    and not active
                    and not self.pending.empty()
                    and engine.completed
                    and self.last_cold_count != len(engine.completed)
                ):
                    await self.make_cold()
                    self.last_cold_count = len(engine.completed)
                while not self.pending.empty() and (not self.cold or active == 0):
                    job = self.pending.get_nowait()
                    rid = job.spec.request_id
                    self.runtime.scheduler.add_request(engine.requests[rid])
                    engine.admitted.add(rid)
                    now = time.perf_counter() - engine.origin
                    engine.metrics[rid]["admitted_s"] = now
                    self.events.emit(
                        "request_admitted",
                        request_id=rid,
                        input_tokens=len(job.spec.prompt_token_ids),
                        timing="http",
                        dependencies_owner="AIPerf/client",
                    )
                    active += 1
                await engine.step()
                if self.runtime.store.errors:
                    raise RuntimeError("; ".join(self.runtime.store.errors))
                for rid, job in list(self.jobs.items()):
                    if rid not in engine.admitted:
                        continue
                    count = engine.metrics[rid]["output_tokens"]
                    for _ in range(count - job.emitted):
                        job.chunks.put_nowait("token")
                    job.emitted = count
                    if rid in engine.completed:
                        job.chunks.put_nowait(None)
                        self.jobs.pop(rid)
                        engine.requests.pop(rid)
                        self.runtime.payloads.patterns.pop(rid)
                        size = (
                            (len(job.spec.prompt_token_ids) + job.spec.output_tokens)
                            * len(self.runtime.caches)
                            * 32
                        )
                        self.runtime.payloads.nbytes -= size
                        self.reserved_bytes -= size
                        self.reserved -= 1
                if not active:
                    await asyncio.sleep(0.005)
            await engine.drain()
        except Exception as error:  # noqa: BLE001 -- propagate native failure to every HTTP waiter
            self.failure = str(error)
            self.events.emit("serving_failure", error=self.failure)
            for job in self.jobs.values():
                job.chunks.put_nowait(error)
            self.jobs.clear()

    async def close(self):
        self.closing = True
        await asyncio.wait_for(self.task, timeout=60)
        if self.failure:
            raise RuntimeError(self.failure)
        keys = (
            await asyncio.to_thread(self.runtime.store.disk_barrier) if self.engine.metrics else []
        )
        rows = list(self.engine.metrics.values())
        for row in rows:
            row["ttft_ms"] = (row["first_token_s"] - row["admitted_s"]) * 1000
            row["latency_ms"] = (row["finished_s"] - row["admitted_s"]) * 1000
        return {
            "status": "passed",
            "requests": len(rows),
            "request_metrics": rows,
            "verified_disk_objects": len(keys),
            "restore_batches": self.runtime.store.restore_batches,
            "checksum_errors": self.runtime.store.errors,
            "femu_verified": self.runtime.store.femu_device is not None,
            "femu_evidence": (
                self.runtime.store.femu_device.verify() if self.runtime.store.femu_device else None
            ),
        }


def create_app(loop, tokenizer, *, model, token_limit):
    app = web.Application(client_max_size=8 * 1024**2)
    output_text = tokenizer.decode([SYNTHETIC_TOKEN], clean_up_tokenization_spaces=False)
    if tokenizer.encode(output_text, add_special_tokens=False) != [SYNTHETIC_TOKEN]:
        raise ValueError("Synthetic output token must round-trip through tokenizer")

    async def health(request):
        return web.json_response(
            {"ready": not loop.failure and not loop.closing, "model": model},
            status=503 if loop.failure else 200,
        )

    async def models(request):
        return web.json_response({"object": "list", "data": [{"id": model, "object": "model"}]})

    async def chat(request):
        try:
            body = await request.json()
            tokens, count, streaming = parse_chat(
                body, tokenizer, model=model, token_limit=token_limit
            )
        except (ValueError, TypeError) as error:
            return web.json_response(
                {"error": {"message": str(error), "type": "invalid_request_error"}}, status=400
            )
        job = await loop.submit(tokens, count, request.headers.get("X-Request-ID"))
        rid = job.spec.request_id
        created = int(time.time())
        usage = {
            "prompt_tokens": len(tokens),
            "completion_tokens": count,
            "total_tokens": len(tokens) + count,
        }
        response = None
        if streaming:
            response = web.StreamResponse(
                headers={
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "X-Request-ID": rid,
                }
            )
            await response.prepare(request)

        async def send(choices, extra=None):
            data = {
                "id": rid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": choices,
            }
            if extra:
                data.update(extra)
            await response.write(("data: " + json.dumps(data) + "\n\n").encode())

        emitted = 0
        try:
            while True:
                chunk = await job.chunks.get()
                if isinstance(chunk, Exception):
                    if response:
                        await response.write(
                            (
                                "data: " + json.dumps({"error": {"message": str(chunk)}}) + "\n\n"
                            ).encode()
                        )
                        await response.write(b"data: [DONE]\n\n")
                        return response
                    raise web.HTTPInternalServerError(text=str(chunk))
                if chunk is None:
                    break
                if streaming:
                    await send(
                        [{"index": 0, "delta": {"content": output_text}, "finish_reason": None}]
                    )
                emitted += 1
                loop.events.emit(
                    "http_token_emitted", request_id=rid, output_tokens=emitted, streaming=streaming
                )
            if streaming:
                await send([{"index": 0, "delta": {}, "finish_reason": "length"}])
                if body.get("stream_options", {}).get("include_usage", False):
                    await send([], {"usage": usage})
                await response.write(b"data: [DONE]\n\n")
                await response.write_eof()
            loop.events.emit("http_response_finished", request_id=rid, output_tokens=emitted)
            return (
                response
                if streaming
                else web.json_response(
                    {
                        "id": rid,
                        "object": "chat.completion",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": output_text * count},
                                "finish_reason": "length",
                            }
                        ],
                        "usage": usage,
                    },
                    headers={"X-Request-ID": rid},
                )
            )
        except (ConnectionResetError, asyncio.CancelledError):
            # Native work continues to completion so pending Store references
            # and fences drain safely. Admission bounds cap disconnected work.
            loop.events.emit("http_client_disconnected", request_id=rid)
            raise

    app.router.add_get("/health", health)
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/chat/completions", chat)
    return app


def serve(args):
    from transformers import AutoTokenizer

    for value in (args.prefill_ms, args.decode_ms, args.read_delay_ms):
        if not math.isfinite(value) or value < 0:
            raise ValueError("Compute and read delays must be finite and nonnegative")
    if not 128 <= args.max_model_len <= 1048576 or not 4 <= args.num_blocks <= 32768:
        raise ValueError("Invalid model length or block pool size")
    if not 1 <= args.max_inflight <= 128 or not 1024 <= args.port <= 65535:
        raise ValueError("Invalid admission limit or port")
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    hybrid = HybridConfig.load(args.hybrid_config) if args.hybrid_config else None
    pool = args.num_blocks * (
        max(hybrid.layers.count(k) for k in ("mla", "kda")) * hybrid.page_bytes
        if hybrid
        else 64 * 512
    )
    staging = max(8 * 1024**2, pool * 2)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    disk = output / "disk"
    femu_device = None
    if args.femu_mount is not None:
        from .femu import FEMUDevice

        if args.read_delay_ms != 0 or not args.cold_between_requests:
            raise ValueError(
                "FEMU verification requires cold-between-requests and no injected delay"
            )
        femu_device = FEMUDevice(args.femu_mount)
        actual_disk = femu_device.mount / output.name
        actual_disk.mkdir(exist_ok=False)
        disk.symlink_to(actual_disk, target_is_directory=True)
    else:
        disk.mkdir()
    manifest = {
        "mode": "http",
        "model": args.model,
        "tokenizer": str(args.tokenizer.resolve()),
        "tokenizer_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(args.tokenizer.glob("*"))
            if p.is_file()
        },
        "harness_source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("vllm", "torch", "mooncake-transfer-engine-non-cuda", "aiohttp")
        },
        "prefill_ms": args.prefill_ms,
        "decode_ms": args.decode_ms,
        "read_delay_ms": args.read_delay_ms,
        "max_model_len": args.max_model_len,
        "num_blocks": args.num_blocks,
        "max_inflight": args.max_inflight,
        "cold_between_requests": args.cold_between_requests,
        "hybrid_config": dataclasses.asdict(hybrid) if hybrid else None,
        "fidelity": {
            "native_scheduler": True,
            "native_mooncake": True,
            "model_forward": False,
            "femu": femu_device is not None,
            "vllm_production_http_server": False,
        },
        "preparation": "per HTTP request before native admission; included in client latency",
        "storage_device": femu_device.describe() if femu_device else None,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    os.environ.update(
        {
            "MOONCAKE_OFFLOAD_FILE_STORAGE_PATH": str(disk.resolve()),
            "MOONCAKE_OFFLOAD_FLUSH_PARTIAL_BUCKETS": "1",
            "MOONCAKE_OFFLOAD_BUCKET_DIRECT_WRITE": "aligned",
            "MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES": str(staging),
            "MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES": str(max(256 * 1024**2, pool * 4)),
            "MOONCAKE_OFFLOAD_HEARTBEAT_INTERVAL_SECONDS": "1",
        }
    )
    with ExitStack() as stack:
        events = stack.enter_context(EventLog(output / "events.jsonl"))
        events.emit("clock_sync", monotonic_origin_ns=events._start)
        master = stack.enter_context(mooncake_master(output))
        config = output / "mooncake.json"
        config.write_text(
            json.dumps(
                {
                    "mode": "embedded",
                    "metadata_server": "P2PHANDSHAKE",
                    "master_server_address": master,
                    "protocol": "tcp",
                    "device_name": "",
                    "enable_offload": True,
                    "global_segment_size": max(32 * 1024**2, args.max_model_len * 1024),
                    "local_buffer_size": staging,
                }
            )
        )
        os.environ["MOONCAKE_CONFIG_PATH"] = str(config)
        native = NativeRuntime(
            output,
            events,
            read_delay_ms=args.read_delay_ms,
            max_model_len=args.max_model_len,
            num_blocks=args.num_blocks,
            hybrid_config=hybrid,
        )
        stack.callback(native.close)
        native.store.femu_device = femu_device
        loop = ServingLoop(
            native,
            events,
            prefill_ms=args.prefill_ms,
            decode_ms=args.decode_ms,
            cold=args.cold_between_requests,
            max_inflight=args.max_inflight,
        )
        app = create_app(
            loop,
            tokenizer,
            model=args.model,
            token_limit=min(args.max_model_len, (args.num_blocks - 2) * 64),
        )

        async def lifecycle(app):
            loop.task = asyncio.create_task(loop.run())
            events.emit("http_server_started", port=args.port)
            try:
                yield
            finally:
                try:
                    report = await loop.close()
                except Exception as error:
                    report = {"status": "failed", "error": str(error)}
                    raise
                finally:
                    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

        app.cleanup_ctx.append(lifecycle)
        web.run_app(app, host="127.0.0.1", port=args.port, access_log=None, shutdown_timeout=60)
