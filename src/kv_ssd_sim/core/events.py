"""Append-only event stream with a single monotonic time domain."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any


class EventLog:
    def __init__(self, path: Path, *, flush_every: int = 64):
        if flush_every < 1:
            raise ValueError("flush_every must be positive")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._file = path.open("x", encoding="utf-8")
        self._start = time.perf_counter_ns()
        self._lock = threading.Lock()
        self._seq = 0
        self._flush_every = flush_every

    def emit(self, event: str, **fields: Any) -> dict:
        with self._lock:
            self._seq += 1
            item = {
                "seq": self._seq,
                "time_ns": time.perf_counter_ns() - self._start,
                "event": event,
                **fields,
            }
            self._file.write(json.dumps(item, ensure_ascii=False) + "\n")
            if self._seq % self._flush_every == 0:
                self._file.flush()
        return item

    def close(self) -> None:
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def load_events(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]
