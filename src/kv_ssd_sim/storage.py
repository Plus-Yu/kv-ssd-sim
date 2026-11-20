"""Observe actual CPU-buffer Mooncake operations without replacing its store.

The optional delay is injected before a native receiver GET. It verifies
feedback into scheduling, and is deliberately not called a FEMU latency.
"""

from __future__ import annotations

import ctypes
import hashlib
import threading
import time
from typing import Any

from .events import EventLog


def buffers_digest(addresses: list[int], sizes: list[int]) -> str:
    digest = hashlib.sha256()
    for address, size in zip(addresses, sizes, strict=True):
        if not address or size < 0:
            raise ValueError("Invalid host buffer")
        # Native Store jobs retain the buffers for the duration of this call.
        # Hash their existing storage without creating a payload-sized bytes copy.
        digest.update(memoryview((ctypes.c_ubyte * size).from_address(address)).cast("B"))
    return digest.hexdigest()


def replica_types(descriptors) -> list[str]:
    result = []
    for item in descriptors:
        if item.is_memory_replica():
            result.append("MEMORY")
        elif item.is_local_disk_replica():
            result.append("LOCAL_DISK")
        elif item.is_disk_replica():
            result.append("DISK")
        else:
            result.append("UNKNOWN")
    return result


class ObservedStore:
    def __init__(self, native: Any, events: EventLog, *, read_delay_ms: float = 0):
        if read_delay_ms < 0:
            raise ValueError("Read delay must be non-negative")
        self.native = native
        self.events = events
        self.read_delay_ms = read_delay_ms
        self._lock = threading.Lock()
        self._context = threading.local()
        self.expected: dict[str, tuple[int, str]] = {}
        self.errors: list[str] = []
        self.restore_batches = 0
        self.on_published = None
        self.femu_device = None

    def trace_request_handler(self, handler, rank=None):
        def wrapped(metadata):
            self._context.rank = rank
            self._context.request_id = metadata.req_id
            try:
                return handler(metadata)
            finally:
                self._context.request_id = None

        return wrapped

    def __getattr__(self, name):
        return getattr(self.native, name)

    def batch_put_from_multi_buffers(self, keys, addresses, sizes, config):
        digests = [buffers_digest(a, s) for a, s in zip(addresses, sizes, strict=True)]
        self.events.emit(
            "store_put_begin",
            keys=keys,
            bytes=sum(map(sum, sizes)),
            request_id=getattr(self._context, "request_id", None),
        )
        result = self.native.batch_put_from_multi_buffers(keys, addresses, sizes, config)
        with self._lock:
            for key, byte_sizes, digest, code in zip(keys, sizes, digests, result, strict=True):
                if code == 0:
                    old = self.expected.get(key)
                    value = (sum(byte_sizes), digest)
                    if old is not None and old != value:
                        self.errors.append(f"Object content changed for identical key: {key}")
                    self.expected[key] = value
                else:
                    self.errors.append(f"Native PUT failed for {key}: {code}")
        self.events.emit(
            "store_put_end",
            keys=keys,
            result=list(result),
            digests=digests,
            logical_rank=getattr(self._context, "rank", None),
            object_bytes=[sum(parts) for parts in sizes],
            request_id=getattr(self._context, "request_id", None),
        )
        if self.on_published:
            self.on_published(
                getattr(self._context, "request_id", None),
                [key for key, code in zip(keys, result, strict=True) if code == 0],
            )
        return result

    def batch_get_into_multi_buffers(self, keys, addresses, sizes):
        tiers = self.native.batch_get_replica_desc(keys)
        tier_map = {key: replica_types(tiers.get(key, [])) for key in keys}
        self.events.emit(
            "store_get_begin",
            keys=keys,
            tiers=tier_map,
            bytes=sum(map(sum, sizes)),
            injected_delay_ms=self.read_delay_ms,
            request_id=getattr(self._context, "request_id", None),
        )
        if self.read_delay_ms:
            # This runs on the native receiver thread, never on the scheduler thread.
            time.sleep(self.read_delay_ms / 1000)
        self.events.emit(
            "native_get_begin", keys=keys, request_id=getattr(self._context, "request_id", None)
        )
        before_io = self.femu_device.counters() if self.femu_device else None
        result = self.native.batch_get_into_multi_buffers(keys, addresses, sizes)
        if self.femu_device:
            self.femu_device.record_get(
                before_io, self.events, getattr(self._context, "request_id", None)
            )
        checks = []
        for key, addrs, byte_sizes, code in zip(keys, addresses, sizes, result, strict=True):
            if code < 0:
                checks.append(False)
                with self._lock:
                    self.errors.append(f"Native GET failed for {key}: {code}")
                continue
            actual = (sum(byte_sizes), buffers_digest(addrs, byte_sizes))
            with self._lock:
                expected = self.expected.get(key)
                ok = actual == expected
                if not ok:
                    self.errors.append(f"Restored bytes failed SHA256 verification: {key}")
            checks.append(ok)
        with self._lock:
            self.restore_batches += 1
        self.events.emit(
            "store_get_end",
            keys=keys,
            result=list(result),
            checksum_ok=checks,
            request_id=getattr(self._context, "request_id", None),
        )
        return result

    def disk_barrier(self, keys=None, *, timeout_s: float = 15) -> list[str]:
        """Wait for confirmed LOCAL_DISK metadata, not just successful PUTs."""
        with self._lock:
            keys = list(self.expected if keys is None else keys)
        if not keys:
            raise RuntimeError("No objects were saved by the native connector")
        deadline = time.monotonic() + timeout_s
        tiers = {key: [] for key in keys}
        while time.monotonic() < deadline:
            descriptors = self.native.batch_get_replica_desc(keys)
            tiers = {key: replica_types(descriptors.get(key, [])) for key in keys}
            if all(any("DISK" in kind for kind in kinds) for kinds in tiers.values()):
                self.events.emit("disk_replicas_confirmed", keys=keys)
                return keys
            time.sleep(0.025)
        self.events.emit(
            "disk_barrier_timeout",
            timeout_s=timeout_s,
            missing_disk_keys=[
                key for key, kinds in tiers.items() if not any("DISK" in kind for kind in kinds)
            ],
            tiers=tiers,
        )
        raise TimeoutError("Mooncake did not establish all disk replicas")

    def evict_memory_replicas(self, keys: list[str]) -> None:
        """Use the Store's replica API and retain the disk replicas."""
        result = self.native.batch_replica_clear(keys, segment_name=self.native.get_hostname())
        descriptors = self.native.batch_get_replica_desc(keys)
        tiers = {key: replica_types(descriptors.get(key, [])) for key in keys}
        if any(any("MEMORY" in kind for kind in kinds) for kinds in tiers.values()):
            raise RuntimeError(f"Memory replicas remain after eviction: {tiers}")
        if any(not any("DISK" in kind for kind in kinds) for kinds in tiers.values()):
            raise RuntimeError("Memory eviction removed a required disk replica")
        self.events.emit("memory_replicas_evicted", keys=keys, result=list(result), tiers=tiers)
