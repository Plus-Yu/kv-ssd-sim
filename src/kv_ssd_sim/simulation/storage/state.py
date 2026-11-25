"""Per-client accounting, replica metadata and in-flight GETs."""

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field


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
