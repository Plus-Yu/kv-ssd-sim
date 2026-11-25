"""Serve native-selected replicas through file-owner staging and destination H2D ports."""

import itertools
import math
from collections import deque

from .device import MIB
from .state import Restore


class RestoreService:
    def __init__(self, clock, config, clients, device, record):
        self.clock = clock
        self.config = config
        self.clients = clients
        self.device = device
        self.record = record
        self.restore_ids = itertools.count(1)
        self.h2d_waiters = [deque() for _ in range(8)]
        self.ready_order = itertools.count()
        self.busy_ranks = set()
        self.busy_memory_ports = set()

    def get(self, obj, origin, tier, ttl, on_complete):
        """Pin the selected replica until its reader has received the data."""
        staged = self.config.h2d_mib_s is not None
        size = obj.size
        client = self.clients[obj.owner]
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

    def assert_drained(self):
        if (
            any(self.h2d_waiters)
            or self.busy_ranks
            or self.busy_memory_ports
            or any(c.restore_staging_bytes or c.staging_waiters for c in self.clients)
        ):
            raise ValueError("Unfinished staged GET at the audit boundary")
