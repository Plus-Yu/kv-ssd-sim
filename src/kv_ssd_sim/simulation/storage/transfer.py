"""Source memory ports and TP rank H2D ports scheduled on the virtual clock."""

import itertools
import math
from collections import deque

from .device import MIB


class TransferPorts:
    def __init__(self, clock, config, record, received):
        self.clock = clock
        self.config = config
        self.record = record
        self.received = received
        self.h2d_waiters = [deque() for _ in range(8)]
        self.ready_order = itertools.count()
        self.busy_ranks = set()
        self.busy_memory_ports = set()

    def busy(self):
        return any(self.h2d_waiters) or bool(self.busy_ranks or self.busy_memory_ports)

    def submit(self, transfer):
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
        self.received(transfer)
        self._dispatch_h2d()
        if transfer.done:
            transfer.done()
