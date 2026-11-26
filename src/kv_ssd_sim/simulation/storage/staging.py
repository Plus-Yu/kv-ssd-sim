"""Bounded restore staging at the file owner, held through the destination copy."""


class StagingPool:
    def __init__(self, clients, capacity, record, start_read):
        self.clients = clients
        self.capacity = capacity
        self.record = record
        self.start_read = start_read

    def acquire(self, transfer):
        obj = transfer.obj
        client = self.clients[obj.owner]
        if client.staging_waiters or obj.size > self.capacity - client.restore_staging_bytes:
            self.record(
                "staging_wait",
                obj,
                transfer.origin,
                staging_client=obj.owner,
                used_bytes=client.restore_staging_bytes,
            )
        client.staging_waiters.append(transfer)
        self._admit_staged_reads(obj.owner)

    def release(self, transfer):
        obj = transfer.obj
        client = self.clients[obj.owner]
        client.restore_staging_bytes -= obj.size
        self.record(
            "staging_release",
            obj,
            transfer.origin,
            staging_client=obj.owner,
            used_bytes=client.restore_staging_bytes,
        )
        self._admit_staged_reads(obj.owner)

    def busy(self):
        return any(c.restore_staging_bytes or c.staging_waiters for c in self.clients)

    def _admit_staged_reads(self, owner):
        client = self.clients[owner]
        while client.staging_waiters:
            transfer = client.staging_waiters[0]
            obj = transfer.obj
            if obj.size > self.capacity - client.restore_staging_bytes:
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
            self.start_read(transfer)
