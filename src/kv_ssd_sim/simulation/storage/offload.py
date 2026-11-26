"""Schedule native Mooncake offload buckets, publish replicas and reclaim disk space."""

import json
from collections import deque
from dataclasses import dataclass

from ...adapters.mooncake import read_key, wire_key


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


class OffloadService:
    """One heartbeat per file owner, sharing the device and live replicas with GETs."""

    def __init__(
        self, clock, config, clients, device, *, replicas, native_call, record, emit, timeline=None
    ):
        self.clock = clock
        self.config = config
        self.clients = clients
        self.device = device
        self.replicas = replicas
        self._native_call = native_call
        self.record = record
        self.emit = emit
        self.timeline = timeline
        self.poll_interval = round(config.offload_delay_ms * 1e6)

    def enqueue(self, obj, origin):
        self.clients[obj.owner].pending[obj.key] = (obj, origin)
        self._schedule_poll(obj.owner)

    def verify_timeline_consumed(self):
        if self.timeline is not None and self.timeline.rows:
            raise ValueError("Unused offload timing records at the final boundary")

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
        for group in self._native_call("poll", owner):
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
        states = self._native_call("source", *(wire_key(k) for k in keys))
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
            (bucket,) = self._native_call("publish", owner, *args)
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
            candidates = [read_key(k) for k in self._native_call("disk_candidate", owner)]
            if not candidates:
                raise RuntimeError("Virtual quota and native bucket inventory disagree")
            for key in candidates:
                obj = self.replicas[key]
                if obj.disk_readers:
                    # A reader waiting for staging still needs the file until its SSD read ends.
                    if obj.on_disk_idle is not None:
                        raise RuntimeError("Concurrent reclaim on the same client")
                    obj.on_disk_idle = resume
                    return False
            removed = self._native_call("remove_disk", owner)
            for expected, record in zip(candidates, removed, strict=True):
                key, state = record.split(":")
                victim = self.replicas[read_key(key)]
                if victim.key != expected or int(state) != int(victim.memory):
                    raise RuntimeError("Native disk eviction disagrees with the replica inventory")
                victim.disk = False
                victim.bucket_id = None
                client.disk_bytes -= victim.size
                self.record(
                    "evict_disk", victim, {"source": "BucketStorageBackend.SelectEvictionCandidate"}
                )
                if not victim.memory:
                    del self.replicas[victim.key]
        return True

    def _reject_offload(self, obj, origin):
        state = self._native_call("nack", obj.owner, obj.size, wire_key(obj.key))
        if state != ["1"]:
            raise RuntimeError("Unexpected native offload NACK state")
        self.record("offload_rejected", obj, origin, native_data_size=-1)
