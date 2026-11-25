"""Virtual-time object replicas and service queues for a single simulated drive."""

from __future__ import annotations

import json
import math
from collections import Counter, deque
from dataclasses import dataclass

from .object_master import NativeMaster, read_key, wire_key
from .simulation.clock import Clock
from .simulation.storage.device import MIB, FifoSSD
from .simulation.storage.restore import RestoreService
from .simulation.storage.state import Client, Replica, Restore

__all__ = [
    "MIB",
    "BucketTiming",
    "Client",
    "Clock",
    "ObjectStore",
    "OffloadTimeline",
    "Replica",
    "Restore",
    "StoreConfig",
]


@dataclass(frozen=True)
class BucketTiming:
    source_query_ns: int
    write_start_ns: int
    publish_ns: int


class OffloadTimeline:
    """Observed times, matched to buckets already formed by native Mooncake."""

    def __init__(self, path):
        self.rows = {}
        with path.open() as source:
            for line in source:
                row = json.loads(line)
                stamp = BucketTiming(
                    row["source_query_ns"], row["write_start_ns"], row["publish_ns"]
                )
                times = (stamp.source_query_ns, stamp.write_start_ns, stamp.publish_ns)
                keys, owner = row["keys"], row["owner"]
                if (
                    any(type(t) is not int or t < 0 for t in times)
                    or not times[0] <= times[1] < times[2]
                    or type(owner) is not int
                    or owner < 0
                    or not isinstance(keys, list)
                    or not keys
                    or any(not isinstance(k, str) or not k for k in keys)
                    or len(keys) != len(set(keys))
                ):
                    raise ValueError("Invalid bucket timing record")
                self.rows.setdefault((owner, tuple(sorted(keys))), deque()).append(stamp)

    def take(self, owner, keys):
        identity = (owner, tuple(sorted(keys)))
        if identity not in self.rows:
            raise ValueError(
                f"No timing for the bucket formed by Mooncake at client {owner}: {keys}"
            )
        stamps = self.rows[identity]
        stamp = stamps.popleft()
        if not stamps:
            del self.rows[identity]
        return stamp


@dataclass(frozen=True)
class StoreConfig:
    clients: int
    segment_bytes: int
    staging_bytes: int
    quota_bytes: int
    offload_delay_ms: float = 1000
    device_mib_s: float = 1024
    memory_mib_s: float = 8192
    lease_ms: float = 0
    save_ms: float = 1
    high_watermark: float = 0.90
    eviction_ratio: float = 0.05
    put_allocation_ms: float = 0
    first_offload_ms: float | None = None
    device_read_mib_s: float | None = None
    put_mib_s: float | None = None
    disk_high_watermark: float | None = None
    disk_low_watermark: float | None = None
    h2d_mib_s: float | None = 8192  # None explicitly selects the historical completion clock.

    def __post_init__(self):
        if self.clients not in (1, 8) or any(
            type(v) is not int or v <= 0
            for v in (self.segment_bytes, self.staging_bytes, self.quota_bytes)
        ):
            raise ValueError("Expected 1 or 8 clients with positive per-client byte budgets")
        for v in (self.offload_delay_ms, self.lease_ms, self.save_ms):
            if not math.isfinite(v) or v < 0:
                raise ValueError("Delay must be finite and nonnegative")
        if int(self.lease_ms) != self.lease_ms:
            raise ValueError("Mooncake lease TTL must be a whole number of milliseconds")
        if not 0 <= self.put_allocation_ms <= self.save_ms:
            raise ValueError("Allocation must lie inside the per-rank save service")
        if (
            self.first_offload_ms is not None
            and not 0 <= self.first_offload_ms <= self.offload_delay_ms
        ):
            raise ValueError("Initial heartbeat phase must lie in the first polling interval")
        if not 0 < self.eviction_ratio < self.high_watermark <= 1:
            raise ValueError("Expected 0 < eviction ratio < high watermark <= 1")
        low, high = self.disk_low_watermark, self.disk_high_watermark
        if (low, high) != (None, None) and (low is None or high is None or not 0 < low < high <= 1):
            raise ValueError("Expected both disk watermarks with 0 < low < high <= 1")
        rates = [self.device_mib_s, self.memory_mib_s]
        rates.extend(
            v for v in (self.device_read_mib_s, self.put_mib_s, self.h2d_mib_s) if v is not None
        )
        for v in rates:
            if not math.isfinite(v) or v <= 0:
                raise ValueError("Service rates must be finite and positive")


class ObjectStore:
    """Virtual payloads and I/O timing around the installed Mooncake metadata service."""

    def __init__(self, clock, config, emit, logfile=None, timeline=None):
        self.clock, self.config, self.emit = clock, config, emit
        self.clients = [Client() for _ in range(config.clients)]
        self.objects = {}
        self.inventory = {}
        self.allocating = {}
        self.counts = Counter()
        self.bytes = Counter()
        self.device = FifoSSD(config.device_read_mib_s or config.device_mib_s, config.device_mib_s)
        self.restores = RestoreService(clock, config, self.clients, self.device, self.record)
        self.poll_interval = round(config.offload_delay_ms * 1e6)
        self.timeline = timeline
        self.rounds = 0
        self.reclaiming_rounds = 0
        self.master = NativeMaster(config, self._native_event, logfile)
        _, self.capacity, self.period = map(int, self._call("usage"))
        self.clock.on_advance = self._advance

    def _call(self, op, *args):
        return self.master.call(op, self.clock.now, *args)

    def _advance(self, when):
        if when // self.period != self.clock.now // self.period:
            self.master.call("advance", when)

    def _native_event(self, tag, when, *fields):
        self.clock.now = int(when)
        if tag == "round":
            before, after = map(int, fields)
            self.rounds += 1
            self.reclaiming_rounds += before > after
            self.emit(
                "native_eviction_round",
                used_before=before,
                used_after=after,
                capacity_bytes=self.capacity,
            )
            return
        key, state = read_key(fields[0]), int(fields[1])
        obj = self.objects[key]
        if not obj.memory or state != (2 if obj.disk else 0):
            raise RuntimeError(f"Unexpected native eviction state for {key}: {state}")
        self._forget_memory(obj, {"source": "MasterService.BatchEvict"})

    def _forget_memory(self, obj, origin):
        obj.memory = False
        self.clients[obj.owner].memory_bytes -= obj.size
        self.record("evict_memory", obj, origin)
        if not obj.disk:
            del self.objects[obj.key]

    def close(self):
        self.clock.on_advance = None
        self.master.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def record(self, op, obj, origin, **extra):
        self.counts[op] += 1
        self.bytes[op] += obj.size
        self.emit(op, key=obj.key, bytes=obj.size, owner=obj.owner, **origin, **extra)

    def batch_is_exist(self, keys):
        if not keys:
            return []
        (result,) = self._call("exists", *(wire_key(k) for k in keys))
        hits = [k for k, bit in zip(keys, result, strict=True) if bit == "1"]
        if hits:
            self.emit("native_lookup_lease", keys=hits, ttl_ms=self.config.lease_ms)
        return list(map(int, result))

    def drop_published_memory(self, origin):
        """Controlled disk-only probe boundary after publication and before any read."""
        for obj in list(self.objects.values()):
            if not obj.disk or obj.readers:
                raise ValueError("The controlled boundary requires completed disk writes and reads")
            if obj.memory:
                state = self._call("clear_memory", obj.writer, obj.owner, wire_key(obj.key))
                if state != ["2"]:
                    raise ValueError("Native clear did not leave a disk-only replica")
                self._forget_memory(obj, dict(origin, source="MasterService.BatchReplicaClear"))

    def put(self, key, size, origin):
        code, obj = self.begin_put(key, size, origin)
        if obj is not None:
            self.finish_put(obj, origin)
        return code

    def begin_put(self, key, size, origin):
        if self.inventory.get(key, self.allocating.get(key, size)) != size:
            raise ValueError(f"Size changed for immutable key {key}")
        writer = origin["rank"] % self.config.clients
        result = list(map(int, self._call("put_start", writer, size, wire_key(key))))
        # The native batch client treats OBJECT_ALREADY_EXISTS as an
        # idempotent success, including a concurrent writer's reservation.
        if result == [-705] and (key in self.objects or key in self.allocating):
            return 0, None
        if result[0] != 0:
            self.record("put_rejected", Replica(key, size, writer, writer), origin, code=result[0])
            return result[0], None
        _, owner = result
        obj = Replica(key, size, owner, writer)
        self.allocating[key] = size
        client = self.clients[owner]
        client.memory_bytes += size
        client.peak_memory_bytes = max(client.peak_memory_bytes, client.memory_bytes)
        self.record("put_allocated", obj, origin)
        return 0, obj

    def finish_put(self, obj, origin):
        self._call("put_end", obj.writer, wire_key(obj.key))
        del self.allocating[obj.key]
        self.objects[obj.key] = obj
        self.inventory[obj.key] = obj.size
        self.record("put_complete", obj, origin)
        self.clients[obj.owner].pending[obj.key] = (obj, origin)
        self._schedule_poll(obj.owner)

    def _schedule_poll(self, owner):
        client = self.clients[owner]
        if client.offload_busy or not client.pending:
            return
        client.offload_busy = True
        due = client.next_poll_ns
        if due is None:
            phase = self.config.first_offload_ms
            due = self.poll_interval if phase is None else round(phase * 1e6)
        if self.poll_interval and due < self.clock.now:
            periods = (self.clock.now - due + self.poll_interval - 1) // self.poll_interval
            due += periods * self.poll_interval
        self.clock.at(max(self.clock.now, due), self._poll, owner)

    def _poll(self, owner):
        client = self.clients[owner]
        groups = deque()
        for group in self._call("poll", owner):
            if group.startswith("-"):
                self._reject_offload(*client.pending.pop(read_key(group[1:])))
            else:
                groups.append([client.pending.pop(read_key(k)) for k in group.split(",")])
        for obj, origin in client.pending.values():
            self.record("offload_unqueued", obj, origin)
        client.pending.clear()
        self._next_bucket(owner, groups)

    def _next_bucket(self, owner, groups):
        client = self.clients[owner]
        if not groups:
            high = self.config.disk_high_watermark
            target = None
            if high is not None and client.disk_bytes > int(self.config.quota_bytes * high):
                target = int(self.config.quota_bytes * self.config.disk_low_watermark)
            self._finish_poll(owner, target)
            return
        group = groups.popleft()
        keys = [obj.key for obj, _ in group]
        stamp = self.timeline.take(owner, keys) if self.timeline is not None else None
        if stamp and stamp.source_query_ns < self.clock.now:
            raise ValueError("Recorded source query precedes native task availability")
        if stamp and stamp.source_query_ns > self.clock.now:
            self.clock.at(stamp.source_query_ns, self._query_sources, owner, groups, group, stamp)
        else:
            self._query_sources(owner, groups, group, stamp)

    def _finish_poll(self, owner, target):
        if target is not None and not self._reclaim_to(
            owner, target, lambda: self._finish_poll(owner, target)
        ):
            return
        client = self.clients[owner]
        client.next_poll_ns = self.clock.now + self.poll_interval
        client.offload_busy = False
        self._schedule_poll(owner)

    def _query_sources(self, owner, groups, group, stamp):
        keys = [obj.key for obj, _ in group]
        # Native FileStorage queries source slices before each bucket write.
        # This grants read leases independently of the source's offload pin.
        states = self._call("source", *(wire_key(k) for k in keys))
        self.emit(
            "native_offload_source_lease", owner=owner, keys=keys, ttl_ms=self.config.lease_ms
        )
        accepted = []
        for (obj, origin), state in zip(group, states, strict=True):
            if int(state) & 1 and obj.size <= self.config.staging_bytes:
                accepted.append((obj, origin))
            else:
                self._reject_offload(obj, origin)
        if stamp and len(accepted) != len(group):
            raise ValueError("Recorded successful bucket contains a rejected source")
        self._write_bucket(owner, groups, accepted, self.clock.now, stamp)

    def _write_bucket(self, owner, remaining, group, source_ns, stamp):
        client = self.clients[owner]
        size = sum(obj.size for obj, _ in group)
        if size > self.config.quota_bytes:
            if stamp:
                raise ValueError("Recorded successful bucket exceeds the virtual payload quota")
            for obj, origin in group:
                self._reject_offload(obj, origin)
            group = []
        if not group:
            self.clock.at(self.clock.now, self._next_bucket, owner, remaining)
            return
        if not self._reclaim_to(
            owner,
            self.config.quota_bytes - size,
            lambda: self._write_bucket(owner, remaining, group, source_ns, stamp),
        ):
            return
        client.disk_bytes += size
        client.peak_disk_bytes = max(client.peak_disk_bytes, client.disk_bytes)
        observed = (stamp.write_start_ns, stamp.publish_ns) if stamp else None
        io = self.device.submit("write", size, self.clock.now, observed_write=observed)
        fields = {"staging_client": owner, "io_id": io.io_id}
        for obj, origin in group:
            self.record("disk_write_submit", obj, origin, **fields)

        def begin():
            for obj, origin in group:
                client.peak_staging_bytes = max(client.peak_staging_bytes, obj.size)
                self.record("disk_write_start", obj, origin, **fields)

        def publish():
            args = [field for obj, _ in group for field in (obj.size, wire_key(obj.key))]
            (bucket,) = self._call("publish", owner, *args)
            self.emit(
                "bucket_timing",
                owner=owner,
                keys=[obj.key for obj, _ in group],
                source_query_ns=source_ns,
                write_start_ns=io.start_ns,
                publish_ns=io.complete_ns,
            )
            for obj, origin in group:
                obj.disk = True
                obj.bucket_id = int(bucket)
                self.record(
                    "disk_publish",
                    obj,
                    origin,
                    bucket_id=int(bucket),
                    source="MasterService.NotifyOffloadSuccess",
                    **fields,
                )
            self._next_bucket(owner, remaining)

        self.clock.at(io.start_ns, begin)
        self.clock.at(io.complete_ns, publish)

    def _reclaim_to(self, owner, target_bytes, resume):
        client = self.clients[owner]
        while client.disk_bytes > target_bytes:
            candidates = [read_key(k) for k in self._call("disk_candidate", owner)]
            if not candidates:
                raise RuntimeError("Virtual quota and native bucket inventory disagree")
            for key in candidates:
                obj = self.objects[key]
                if obj.disk_readers:
                    # Staging admission may still be waiting; the file is needed only through SSD read.
                    if obj.on_disk_idle is not None:
                        raise RuntimeError("Concurrent reclaim on the same client")
                    obj.on_disk_idle = resume
                    return False
            removed = self._call("remove_disk", owner)
            for expected, record in zip(candidates, removed, strict=True):
                key, state = record.split(":")
                victim = self.objects[read_key(key)]
                if victim.key != expected or int(state) != int(victim.memory):
                    raise RuntimeError("Native disk eviction disagrees with the replica inventory")
                victim.disk = False
                victim.bucket_id = None
                client.disk_bytes -= victim.size
                self.record(
                    "evict_disk", victim, {"source": "BucketStorageBackend.SelectEvictionCandidate"}
                )
                if not victim.memory:
                    del self.objects[victim.key]
        return True

    def _reject_offload(self, obj, origin):
        state = self._call("nack", obj.owner, obj.size, wire_key(obj.key))
        if state != ["1"]:
            raise RuntimeError("Unexpected native offload NACK state")
        self.record("offload_rejected", obj, origin, native_data_size=-1)

    def get(self, key, size, origin, *, on_complete=None):
        """Return a clock-waitable restore (or the historical integer deadline in legacy mode)."""
        obj = self.objects[key]
        if obj.size != size:
            raise ValueError(f"Restore descriptor differs for {key}")
        if self.config.h2d_mib_s is not None and not 0 <= origin["rank"] < 8:
            raise ValueError("Expected a TP8 destination rank")
        state, ttl = map(int, self._call("get", wire_key(key)))
        if state != int(obj.memory) + 2 * int(obj.disk) or state == 0:
            raise RuntimeError(f"Native GET state differs for {key}: {state}")
        tier = "memory" if state & 1 else "disk"
        return self.restores.get(obj, origin, tier, ttl, on_complete)

    def mark_measurement(self, request, local_usage, local_cached_hashes):
        allocated, capacity, _ = map(int, self._call("usage"))
        self.emit(
            "measurement_anchor",
            request=request,
            allocated_bytes=allocated,
            capacity_bytes=capacity,
            local_pool_usage=local_usage,
            local_cached_hashes=local_cached_hashes,
            in_progress_allocations=len(self.allocating),
            memory_payload_bytes=sum(c.memory_bytes for c in self.clients),
            reserved_disk_bytes=sum(c.disk_bytes for c in self.clients),
            published_disk_bytes=sum(o.size for o in self.objects.values() if o.disk),
            counts=dict(self.counts),
        )

    def summary(self):
        if self.allocating:
            raise ValueError("Unfinished PUT allocations at the audit boundary")
        self.restores.assert_drained()
        if self.timeline is not None and self.timeline.rows:
            raise ValueError("Unused offload timing records at the final boundary")
        allocated, capacity, _ = map(int, self._call("usage"))
        for key in self.inventory:
            obj = self.objects.get(key)
            expected = int(obj.memory) + 2 * int(obj.disk) if obj else 0
            if self._call("inspect", wire_key(key)) != [str(expected)]:
                raise RuntimeError(f"Final native replica audit failed for {key}")
        return {
            "native_master": {
                "used_bytes": allocated,
                "resident_payload_bytes": sum(c.memory_bytes for c in self.clients),
                "process_peak_rss_kib": self.master.peak_rss_kib(),
                "capacity_bytes": capacity,
                "period_ns": self.period,
                "rounds": self.rounds,
                "reclaiming_rounds": self.reclaiming_rounds,
                "admin_verified_keys": len(self.inventory),
            },
            "unique_keys": len(self.inventory),
            "unique_logical_bytes": sum(self.inventory.values()),
            "counts": dict(self.counts),
            "bytes": dict(self.bytes),
            "clients": [
                {
                    "id": index,
                    "memory_bytes": c.memory_bytes,
                    "disk_bytes": c.disk_bytes,
                    "peak_memory_bytes": c.peak_memory_bytes,
                    "peak_disk_bytes": c.peak_disk_bytes,
                    "peak_staging_bytes": c.peak_staging_bytes,
                    **(
                        {
                            "restore_staging_bytes": c.restore_staging_bytes,
                            "peak_restore_staging_bytes": c.peak_restore_staging_bytes,
                            "restore_staging_capacity_bytes": self.config.staging_bytes,
                        }
                        if self.config.h2d_mib_s is not None
                        else {}
                    ),
                }
                for index, c in enumerate(self.clients)
            ],
        }
