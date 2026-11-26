"""Virtual-time object replicas and service queues for a single simulated drive."""

from __future__ import annotations

from collections import Counter

from ...adapters.mooncake import NativeMaster, read_key, wire_key
from .device import default_device
from .offload import OffloadService
from .restore import RestoreService
from .state import Client, Replica
from .transfer import TransferPorts


class ObjectStore:
    """Virtual payloads and I/O timing around the installed Mooncake metadata service."""

    def __init__(
        self,
        clock,
        config,
        emit,
        logfile=None,
        timeline=None,
        *,
        device=None,
        transfer_factory=TransferPorts,
    ):
        self.clock, self.config, self.emit = clock, config, emit
        self.clients = [Client() for _ in range(config.clients)]
        self.objects = {}
        self.inventory = {}
        self.allocating = {}
        self.counts = Counter()
        self.bytes = Counter()
        self.device = default_device(config) if device is None else device
        self.restores = RestoreService(
            clock, config, self.clients, self.device, self.record, transfer_factory
        )
        self.rounds = 0
        self.reclaiming_rounds = 0
        self.master = NativeMaster(config, self._native_event, logfile)
        self.offload = OffloadService(
            clock,
            config,
            self.clients,
            self.device,
            replicas=self.objects,
            native_call=self._call,
            record=self.record,
            emit=emit,
            timeline=timeline,
        )
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
        self.offload.enqueue(obj, origin)

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
        self.offload.verify_timeline_consumed()
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
