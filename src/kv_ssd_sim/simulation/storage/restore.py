"""Serve native-selected replicas through file-owner staging and destination H2D ports."""

import itertools
import math

from .device import MIB
from .staging import StagingPool
from .state import Restore
from .transfer import TransferPorts


class RestoreService:
    def __init__(self, clock, config, clients, device, record, transfer_factory=TransferPorts):
        self.clock = clock
        self.config = config
        self.clients = clients
        self.device = device
        self.record = record
        self.restore_ids = itertools.count(1)
        self.ports = transfer_factory(clock, config, record, self._received)
        self.staging = StagingPool(clients, config.staging_bytes, record, self._staged_read)

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
                self.ports.submit(transfer)
            else:
                self.staging.acquire(transfer)
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
                self.ports.submit(transfer)

        self.clock.at(io.start_ns, begin)
        self.clock.at(io.complete_ns, complete)
        return io.complete_ns

    def _staged_read(self, transfer):
        self._disk_read(transfer.obj, transfer.origin, transfer.obj.owner, transfer)

    def _received(self, transfer):
        if transfer.tier == "disk":
            self.staging.release(transfer)
        self._finish_get(transfer.obj, transfer.origin)
        transfer.complete_ns = self.clock.now

    def assert_drained(self):
        if self.ports.busy() or self.staging.busy():
            raise ValueError("Unfinished staged GET at the audit boundary")
