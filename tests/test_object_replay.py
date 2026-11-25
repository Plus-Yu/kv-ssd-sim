import asyncio
import json
from collections import deque
from types import SimpleNamespace

import pytest

from kv_ssd_sim.object_store import MIB, Clock, ObjectStore, OffloadTimeline, StoreConfig
from kv_ssd_sim.object_timing import StepTiming
from kv_ssd_sim.object_traffic import describe_traffic


def test_inflight_native_put_charges_capacity_before_becoming_readable():
    clock = Clock()
    cfg = StoreConfig(1, 8 * MIB, 8 * MIB, 16 * MIB, high_watermark=0.5)
    with ObjectStore(clock, cfg, lambda *a, **k: None) as store:
        code, pending = store.begin_put("still-copying", 6 * MIB, {"rank": 0})
        assert code == 0
        assert store.begin_put("still-copying", 6 * MIB, {"rank": 1}) == (0, None)
        assert list(map(int, store.master.call("usage", 0)))[:2] == [6 * MIB, 8 * MIB]
        assert store.batch_is_exist(["still-copying"]) == [0]
        clock.advance(50_000_000)
        assert store.rounds and store.reclaiming_rounds == 0
        store.finish_put(pending, {"rank": 0})
        assert store.batch_is_exist(["still-copying"]) == [1]


def test_native_source_pin_leases_and_eviction_during_idle():
    clock = Clock()
    events = []
    config = StoreConfig(
        1,
        12 * MIB,
        8 * MIB,
        64 * MIB,
        offload_delay_ms=200,
        lease_ms=1000,
        high_watermark=0.5,
        device_read_mib_s=256,
    )
    with ObjectStore(clock, config, lambda op, **e: events.append((op, clock.now, e))) as store:
        for key in "abcdefgh":
            assert store.put(key, MIB, {"rank": 0}) == 0
        clock.advance(190_000_000)
        assert store.rounds > 0 and store.reclaiming_rounds == 0
        assert all(o.memory and not o.disk for o in store.objects.values())
        clock.advance(1_100_000_000)
        assert all(o.memory and o.disk for o in store.objects.values())
        assert store.summary()["native_master"]["used_bytes"] == 8 * MIB
        assert store.batch_is_exist(["g", "missing"]) == [1, 0]
        clock.advance(store.get("h", MIB, {"rank": 0}))
        clock.advance(1_400_000_000)
        evicted = [(t, e["key"]) for op, t, e in events if op == "evict_memory"]
        assert evicted and all(t >= 1_200_000_000 and k in "abcdef" for t, k in evicted)
        assert store.objects["g"].memory and store.objects["h"].memory
        assert store.summary()["native_master"]["used_bytes"] <= 6 * MIB
        assert {e["key"] for op, _, e in events if op == "disk_publish"} == set("abcdefgh")
        assert len({e["bucket_id"] for op, _, e in events if op == "disk_publish"}) == 1
        # A real native eviction left this published key on disk. No tier override.
        key = evicted[0][1]
        assert store.batch_is_exist([key]) == [1]
        end = store.get(key, MIB, {"rank": 0})
        assert end - clock.now == 3_906_250
        clock.advance(end)
        assert store.bytes["disk_read_complete"] == store.bytes["get_disk"] == MIB


def test_step_time_uses_fresh_tokens_and_the_scheduled_batch():
    timing = StepTiming(None, 11.77, 14.10, 22000)
    assert timing.duration_ms(0, 0, 0) == 0
    assert timing.duration_ms(0, 8, 8) == pytest.approx(14.10)
    assert timing.duration_ms(1, 2, 1) == pytest.approx(11.77 + (14.10 - 11.77) / 7)
    assert timing.duration_ms(4096, 3, 1) == pytest.approx(4096 * 1000 / 22000)
    assert StepTiming(10, 2, 2).duration_ms(4096, 3, 1) == 10


def test_per_rank_put_time_uses_accepted_bytes_and_waits_for_the_previous_save():
    pytest.importorskip("vllm")
    from kv_ssd_sim.object_worker import MetadataRank

    rank = MetadataRank.__new__(MetadataRank)
    rank.clock = clock = Clock()
    events = []
    rank.port = SimpleNamespace(rank=0)
    rank.sender = SimpleNamespace(add_request=lambda meta: None)
    rank.store = SimpleNamespace(
        config=StoreConfig(1, MIB, MIB, MIB, put_mib_s=1000),
        emit=lambda op, **e: events.append((e["start_ns"], e["complete_ns"], e["copy_bytes"])),
    )
    rank.pending_saves = deque()
    # The second job finds an existing key. Only the first reserves new payload.
    rank._begin_save = lambda meta, batch: batch.writes.extend(meta.writes)
    rank._finish_save = lambda meta, batch: None
    for i, writes in enumerate(([(SimpleNamespace(size=MIB), {})], [])):
        rank.enqueue_save(SimpleNamespace(req_id=str(i), store_job_id=i, writes=writes), 1_000_000)
    clock.advance(5_000_000)
    assert events == [(1_000_000, 3_000_000, MIB), (3_000_000, 4_000_000, 0)]
    assert not rank.pending_saves


def test_native_allocation_pressure_and_nack_release_the_last_replica():
    clock = Clock()
    config = StoreConfig(1, 8 * MIB, MIB, 16 * MIB, offload_delay_ms=20, high_watermark=1)
    with ObjectStore(clock, config, lambda *a, **k: None) as store:
        assert store.put("large", 8 * MIB, {"rank": 0}) == 0
        assert store.put("more", MIB, {"rank": 0}) == -200
        clock.advance(10_000_000)
        assert store.objects["large"].memory
        clock.advance(20_000_000)
        assert store.counts["offload_rejected"] == 1
        assert not store.objects["large"].disk
        # The failed allocation asks the background worker to try at its next tick.
        assert store.put("more", MIB, {"rank": 0}) == -200
        clock.advance(40_000_000)
        assert store.batch_is_exist(["large"]) == [0]
        assert store.put("more", MIB, {"rank": 0}) == 0


def test_recorded_completion_keeps_sources_pinned_after_lease_expires(tmp_path):
    path = tmp_path / "timing.jsonl"
    path.write_text(
        json.dumps(
            {
                "owner": 0,
                "keys": list("abcdef"),
                "source_query_ns": 5_000_000,
                "write_start_ns": 8_000_000,
                "publish_ns": 1_000_000_000,
            }
        )
        + "\n"
    )
    clock = Clock()
    cfg = StoreConfig(
        1, 8 * MIB, 8 * MIB, 16 * MIB, offload_delay_ms=0, lease_ms=20, high_watermark=0.5
    )
    events = []
    with ObjectStore(
        clock, cfg, lambda op, **e: events.append((op, clock.now)), timeline=OffloadTimeline(path)
    ) as store:
        for key in "abcdef":
            assert store.put(key, MIB, {"rank": 0}) == 0
        clock.advance(999_999_999)
        assert store.rounds and store.reclaiming_rounds == 0
        assert all(o.memory and not o.disk for o in store.objects.values())
        clock.advance(1_000_000_000)
        assert all(o.memory and o.disk for o in store.objects.values())
        clock.advance(1_020_000_000)
        assert store.reclaiming_rounds > 0
        assert store.summary()["native_master"]["used_bytes"] <= 4 * MIB
        assert ("native_offload_source_lease", 5_000_000) in events
        assert ("disk_write_start", 8_000_000) in events
        assert ("disk_publish", 1_000_000_000) in events
    with pytest.raises(ValueError, match="No timing"):
        OffloadTimeline(path).take(0, ["a"])


def test_disk_quota_evicts_the_whole_native_fifo_bucket():
    clock = Clock()
    cfg = StoreConfig(1, 32 * MIB, 4 * MIB, 2 * MIB, offload_delay_ms=0, lease_ms=5000)
    with ObjectStore(clock, cfg, lambda *a, **k: None) as store:
        for k in "ab":
            assert store.put(k, MIB, {"rank": 0}) == 0
        clock.advance(10_000_000)
        assert store.batch_is_exist(["a"]) == [1]
        assert store.put("c", MIB, {"rank": 0}) == 0
        clock.advance(20_000_000)
        assert {k for k, o in store.objects.items() if o.disk} == {"c"}
        assert store.counts["evict_disk"] == 2
        assert store.summary()["native_master"]["used_bytes"] == 3 * MIB


def test_shared_drive_serializes_clients_and_native_pressure_is_global():
    clock = Clock()
    events = []
    cfg = StoreConfig(8, 8 * MIB, 8 * MIB, 16 * MIB, offload_delay_ms=0)
    with ObjectStore(clock, cfg, lambda op, **e: events.append((op, clock.now, e))) as store:
        for rank in (0, 7):
            assert store.put(str(rank), 8 * MIB, {"rank": rank}) == 0
        clock.advance(30_000_000)
        ordered = [(op, t) for op, t, _ in events if op in ("disk_publish", "disk_write_start")]
        assert ordered[1][0] == "disk_publish" and ordered[1][1] == ordered[2][1]
        assert [c.disk_bytes for c in store.clients] == [8 * MIB, 0, 0, 0, 0, 0, 0, 8 * MIB]
        native = store.summary()["native_master"]
        assert (native["used_bytes"], native["capacity_bytes"], native["rounds"]) == (
            16 * MIB,
            64 * MIB,
            0,
        )


def test_file_heartbeat_reclaims_whole_native_buckets_at_disk_watermarks():
    clock = Clock()
    cfg = StoreConfig(
        1,
        16 * MIB,
        4 * MIB,
        5 * MIB,
        offload_delay_ms=0,
        disk_high_watermark=0.75,
        disk_low_watermark=0.50,
    )
    with ObjectStore(clock, cfg, lambda *a, **k: None) as store:
        for i in range(4):
            assert store.put(str(i), MIB, {"rank": 0}) == 0
            clock.advance(clock.now + 2_000_000)
        assert {k for k, obj in store.objects.items() if obj.disk} == {"2", "3"}
        assert store.clients[0].peak_disk_bytes == 4 * MIB < cfg.quota_bytes
        assert store.bytes["evict_disk"] == 2 * MIB


def test_exact_reuse_distance_in_bytes(tmp_path):
    source = tmp_path / "events.jsonl"
    rows = [("a", "r1", 100), ("b", "r1", 20), ("c", "r2", 30), ("b", "r2", 20), ("a", "r3", 100)]
    source.write_text(
        "".join(
            json.dumps(
                {
                    "event": "put_complete" if i < 3 else "get_memory",
                    "key": k,
                    "bytes": n,
                    "request": r,
                    "time_ns": i,
                }
            )
            + "\n"
            for i, (k, r, n) in enumerate(rows)
        )
    )
    out = tmp_path / "reuse.jsonl"
    traffic, count = describe_traffic(source, out, len(rows))
    assert [
        json.loads(line)["distinct_intervening_bytes"] for line in out.read_text().splitlines()
    ] == [30, 50]
    assert traffic["r3"]["bytes"] == {"get_memory": 100} and count == 2


def test_load_ack_survives_multiple_scheduler_polls():
    pytest.importorskip("vllm")
    from kv_ssd_sim.object_worker import MetadataTransfers

    clock = Clock()
    rank = SimpleNamespace(
        clock=clock,
        worker=SimpleNamespace(_close_ended_store_requests=lambda *a: None),
        completed_saves={},
        load=lambda _: 100,
    )
    transfers = MetadataTransfers([rank])
    meta = SimpleNamespace(
        requests=[
            SimpleNamespace(
                req_id="r",
                can_save=False,
                load_spec=SimpleNamespace(
                    can_load=True, kvpool_cached_tokens=1536, vllm_cached_tokens=0
                ),
            )
        ]
    )
    assert transfers.submit(meta, set(), 0) == {"r": 1536}
    assert not transfers.output().finished_recving
    clock.advance(100)
    assert transfers.output().finished_recving == {"r"}
    assert not transfers.output().finished_recving


def test_native_shutdown_detects_errors_from_last_async_put():
    from kv_ssd_sim.server import ServingLoop

    async def close():
        store = SimpleNamespace(errors=[], femu_device=None)

        async def drain():
            store.errors.append("late PUT failure")

        loop = SimpleNamespace(
            task=asyncio.create_task(drain()),
            failure=None,
            cold=False,
            runtime=SimpleNamespace(store=store, workers=[]),
            engine=SimpleNamespace(metrics={}),
        )
        await ServingLoop.close(loop)

    with pytest.raises(RuntimeError, match="late PUT failure"):
        asyncio.run(close())
