"""Opt-in capacity experiment and a single explicitly unsubmitted KDA producer."""

import dataclasses
import json
import threading
from pathlib import Path

from .storage import replica_types


@dataclasses.dataclass(frozen=True)
class PressureConfig:
    segment_bytes: int
    staging_bytes: int
    disk_bytes: int
    high_watermark: float
    eviction_ratio: float
    lease_ttl_ms: int
    producer: str
    restore: str
    incomplete_producer: str
    recompute: str

    @classmethod
    def load(cls, path: Path):
        config = cls(**json.loads(path.read_text()))
        if not all(
            type(n) is int and n > 0
            for n in (config.segment_bytes, config.staging_bytes, config.disk_bytes)
        ):
            raise ValueError("Storage capacities must be positive bytes")
        if not (0 < config.high_watermark < 1 and 0 < config.eviction_ratio <= 1):
            raise ValueError("Invalid eviction watermarks")
        if type(config.lease_ttl_ms) is not int or config.lease_ttl_ms <= 0:
            raise ValueError("Natural replay requires a positive read lease in milliseconds")
        names = [config.producer, config.restore, config.incomplete_producer, config.recompute]
        if not all(isinstance(n, str) and n for n in names) or len(set(names)) != 4:
            raise ValueError("Expected four distinct named experiment requests")
        return config


class PressureObserver:
    def __init__(self, runtime, config, events):
        self.config, self.store, self.events = config, runtime.store, events
        self._lock = threading.Lock()
        self.omitted = set()
        self.published = {config.producer: set(), config.incomplete_producer: set()}
        sender = runtime.worker.native.kv_send_thread
        original = sender._boundary_puts

        def boundary_puts(meta):
            puts = original(meta)
            if meta.req_id != config.incomplete_producer:
                return puts
            withheld = [p[0] for p in puts if "@store_format:mamba_state@" in p[0]]
            if withheld:
                with self._lock:
                    self.omitted.update(withheld)
                events.emit("kda_not_submitted", request_id=meta.req_id, keys=withheld)
            return [p for p in puts if "@store_format:mamba_state@" not in p[0]]

        enqueue, handle = sender.add_request, sender._handle_request

        def enqueue_job(meta):
            enqueue(meta)
            events.emit("save_job_submitted", request_id=meta.req_id, job=meta.store_job_id)

        def handle_job(meta):
            events.emit("save_job_started", request_id=meta.req_id, job=meta.store_job_id)
            result = handle(meta)
            events.emit("save_job_finished", request_id=meta.req_id, job=meta.store_job_id)
            return result

        sender.add_request = enqueue_job
        sender._handle_request = handle_job
        sender._boundary_puts = boundary_puts
        self.store.on_published = self.record_publish

    def tiers(self, keys):
        keys = sorted(keys)
        desc = self.store.native.batch_get_replica_desc(keys) if keys else {}
        return {key: replica_types(desc.get(key, [])) for key in keys}

    def record_publish(self, rid, keys):
        if rid in self.published:
            with self._lock:
                self.published[rid].update(keys)
            self.events.emit("producer_replicas", request_id=rid, tiers=self.tiers(keys))

    def before_admission(self, rid):
        with self._lock:
            if rid == self.config.restore:
                keys = self.published[self.config.producer].copy()
            elif rid == self.config.recompute:
                keys = self.published[self.config.incomplete_producer] | self.omitted
            else:
                return
        self.events.emit("probe_replicas", request_id=rid, tiers=self.tiers(keys))
