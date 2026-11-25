"""Virtual-time object replicas and service queues for a single simulated drive."""

from __future__ import annotations

import heapq
import itertools
import json
import math
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass, field

from .object_master import NativeMaster, read_key, wire_key
from .simulation.storage.device import MIB, FifoSSD


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


class Clock:
    def __init__(self):
        self.now = 0
        self.events = []
        self._sequence = itertools.count()
        self.on_advance = None

    def at(self, when, callback, *args):
        if when < self.now:
            raise ValueError("Cannot enqueue in the past")
        heapq.heappush(self.events, (when, next(self._sequence), callback, args))

    def advance(self, until):
        if isinstance(until, Restore):
            while until.complete_ns is None:
                if not self.events:
                    raise RuntimeError("Restore stalled without a scheduled completion")
                self.advance(self.events[0][0])
            return
        if until < self.now:
            raise ValueError("Clock moved backwards")
        while self.events and self.events[0][0] <= until:
            when, _, callback, args = heapq.heappop(self.events)
            self._move(when)
            callback(*args)
        self._move(until)

    def _move(self, when):
        if self.on_advance and self.now != when:
            self.on_advance(when)
        self.now = when


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


@dataclass
class Client:
    memory_bytes: int = 0
    disk_bytes: int = 0  # includes space reserved for pending writes
    peak_memory_bytes: int = 0
    peak_disk_bytes: int = 0
    peak_staging_bytes: int = 0
    restore_staging_bytes: int = 0
    peak_restore_staging_bytes: int = 0
    staging_waiters: deque = field(default_factory=deque)
    memory_free_at: int = 0
    next_poll_ns: int | None = None
    offload_busy: bool = False
    pending: dict = field(default_factory=dict)


@dataclass
class Replica:
    key: str
    size: int
    owner: int
    writer: int
    memory: bool = True
    disk: bool = False
    readers: int = 0
    disk_readers: int = 0
    on_disk_idle: Callable[[], None] | None = None
    bucket_id: int | None = None


@dataclass
class Restore:
    obj: Replica
    origin: dict
    tier: str
    done: Callable[[], None] | None
    complete_ns: int | None = None


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
        self.restore_ids = itertools.count(1)
        self.h2d_waiters = [deque() for _ in range(8)]
        self.ready_order = itertools.count()
        self.busy_ranks = set()
        self.busy_memory_ports = set()
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

    def _admit_staged_reads(self, owner):
        client = self.clients[owner]
        while client.staging_waiters:
            transfer = client.staging_waiters[0]
            obj = transfer.obj
            if obj.size > self.config.staging_bytes - client.restore_staging_bytes:
                break
            client.staging_waiters.popleft()
            client.restore_staging_bytes += obj.size
            client.peak_restore_staging_bytes = max(
                client.peak_restore_staging_bytes, client.restore_staging_bytes
            )
            client.peak_staging_bytes = max(client.peak_staging_bytes, client.restore_staging_bytes)
            self.record(
                "staging_reserve",
                obj,
                transfer.origin,
                staging_client=owner,
                used_bytes=client.restore_staging_bytes,
            )
            self._disk_read(obj, transfer.origin, owner, transfer)

    def _disk_read(self, obj, origin, staging_client, transfer=None):
        io = self.device.submit("read", obj.size, self.clock.now)
        fields = {
            "staging_client": staging_client,
            "io_id": io.io_id,
            "bucket_id": obj.bucket_id,
        }
        self.record("disk_read_submit", obj, origin, **fields)
        client = self.clients[staging_client]

        def begin():
            client.peak_staging_bytes = max(client.peak_staging_bytes, obj.size)
            self.record("disk_read_start", obj, origin, **fields)

        def complete():
            self.record("disk_read_complete", obj, origin, **fields)
            obj.disk_readers -= 1
            if not obj.disk_readers and obj.on_disk_idle:
                self.clock.at(self.clock.now, obj.on_disk_idle)
                obj.on_disk_idle = None
            if transfer is None:
                self._finish_get(obj, origin)
            else:
                self._ready_h2d(transfer)

        self.clock.at(io.start_ns, begin)
        self.clock.at(io.complete_ns, complete)
        return io.complete_ns

    def _ready_h2d(self, transfer):
        self.record("h2d_ready", transfer.obj, transfer.origin, tier=transfer.tier)
        self.h2d_waiters[transfer.origin["rank"]].append((next(self.ready_order), transfer))
        self._dispatch_h2d()

    def _dispatch_h2d(self):
        # FIFO among ready copies for each rank. A busy source port need not idle other ranks.
        heads = sorted(
            (q[0][0], rank)
            for rank, q in enumerate(self.h2d_waiters)
            if q and rank not in self.busy_ranks
        )
        for _, rank in heads:
            queue = self.h2d_waiters[rank]
            transfer = queue[0][1]
            owner = transfer.obj.owner
            memory = transfer.tier == "memory"
            if memory and owner in self.busy_memory_ports:
                continue
            queue.popleft()
            self.busy_ranks.add(rank)
            rate = self.config.h2d_mib_s
            if memory:
                self.busy_memory_ports.add(owner)
                rate = min(rate, self.config.memory_mib_s)
            self.record("h2d_start", transfer.obj, transfer.origin, tier=transfer.tier)
            duration = math.ceil(transfer.obj.size * 1e9 / (rate * MIB))
            self.clock.at(self.clock.now + duration, self._complete_h2d, transfer)

    def _complete_h2d(self, transfer):
        obj, origin = transfer.obj, transfer.origin
        self.busy_ranks.remove(origin["rank"])
        self.record("h2d_complete", obj, origin, tier=transfer.tier)
        if transfer.tier == "memory":
            self.busy_memory_ports.remove(obj.owner)
        else:
            client = self.clients[obj.owner]
            client.restore_staging_bytes -= obj.size
            self.record(
                "staging_release",
                obj,
                origin,
                staging_client=obj.owner,
                used_bytes=client.restore_staging_bytes,
            )
            self._admit_staged_reads(obj.owner)
        self._finish_get(obj, origin)
        transfer.complete_ns = self.clock.now
        self._dispatch_h2d()
        if transfer.done:
            transfer.done()

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
        staged = self.config.h2d_mib_s is not None
        if staged and not 0 <= origin["rank"] < 8:
            raise ValueError("Expected a TP8 destination rank")
        state, ttl = map(int, self._call("get", wire_key(key)))
        if state != int(obj.memory) + 2 * int(obj.disk) or state == 0:
            raise RuntimeError(f"Native GET state differs for {key}: {state}")
        client = self.clients[obj.owner]
        tier = "memory" if state & 1 else "disk"
        if tier == "disk" and size > self.config.staging_bytes:
            raise ValueError("Object exceeds GET staging budget")
        obj.readers += 1
        if tier == "disk":
            obj.disk_readers += 1
        if staged:
            origin = dict(origin, restore_id=next(self.restore_ids))
        self.record("get_" + tier, obj, origin, native_lease_ms=ttl)
        if staged:
            transfer = Restore(obj, origin, tier, on_complete)
            if tier == "memory":
                self._ready_h2d(transfer)
            else:
                if (
                    client.staging_waiters
                    or size > self.config.staging_bytes - client.restore_staging_bytes
                ):
                    self.record(
                        "staging_wait",
                        obj,
                        origin,
                        staging_client=obj.owner,
                        used_bytes=client.restore_staging_bytes,
                    )
                client.staging_waiters.append(transfer)
                self._admit_staged_reads(obj.owner)
            return transfer
        if tier == "disk":
            end = self._disk_read(obj, origin, origin["rank"] % self.config.clients)
        else:
            end = max(self.clock.now, client.memory_free_at) + math.ceil(
                size * 1e9 / (self.config.memory_mib_s * MIB)
            )
            client.memory_free_at = end
            self.clock.at(end, self._finish_get, obj, origin)
        if on_complete:
            self.clock.at(end, on_complete)
        return end

    def _finish_get(self, obj, origin):
        obj.readers -= 1
        self.record("get_complete", obj, origin)

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
        if (
            any(self.h2d_waiters)
            or self.busy_ranks
            or self.busy_memory_ports
            or any(c.restore_staging_bytes or c.staging_waiters for c in self.clients)
        ):
            raise ValueError("Unfinished staged GET at the audit boundary")
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
