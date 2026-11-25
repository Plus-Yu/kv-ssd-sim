"""Virtual clock for object replay, with stable ordering at equal timestamps."""

import heapq
import itertools

from .storage.state import Restore


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
