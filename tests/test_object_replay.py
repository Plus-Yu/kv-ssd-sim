import asyncio
import json
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from kv_ssd_sim.analysis.reuse import describe_traffic
from kv_ssd_sim.simulation.clock import Clock
from kv_ssd_sim.simulation.storage.config import StoreConfig
from kv_ssd_sim.simulation.storage.device import MIB, FifoSSD
from kv_ssd_sim.simulation.storage.offload import OffloadTimeline
from kv_ssd_sim.simulation.storage.store import ObjectStore
from kv_ssd_sim.simulation.storage.transfer import TransferPorts
from kv_ssd_sim.simulation.timing import StepTiming


def test_supplied_device_and_ports_serve_both_offload_and_restore():
    clock = Clock()
    cfg = StoreConfig(1, 8 * MIB, MIB, 4 * MIB, offload_delay_ms=0, h2d_mib_s=1000)
    device = FifoSSD(read_mib_s=250, write_mib_s=500)
    ports = Mock(wraps=TransferPorts)
    events = []
    with ObjectStore(
        clock,
        cfg,
        lambda op, **row: events.append((op, clock.now)),
        device=device,
        transfer_factory=ports,
    ) as store:
        assert store.put("k", MIB, {"rank": 0}) == 0
        clock.advance(2_000_000)
        store.drop_published_memory({})
        clock.advance(store.get("k", MIB, {"rank": 7}))
        assert ("disk_publish", 2_000_000) in events
        assert ("disk_read_complete", 6_000_000) in events
        assert ("h2d_complete", 7_000_000) in events
        ports.assert_called_once()
        assert store.summary()["clients"][0]["restore_staging_bytes"] == 0


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
        start = clock.now
        restore = store.get(key, MIB, {"rank": 0})
        clock.advance(restore)
        assert clock.now - start == 3_906_250 + 122_071
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
    from kv_ssd_sim.simulation.worker import MetadataRank

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
        with pytest.raises(ValueError, match="Unused offload timing"):
            store.summary()
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


def test_shared_volume_keeps_independent_client_watermarks():
    clock = Clock()
    cfg = StoreConfig(
        8,
        16 * MIB,
        4 * MIB,
        5 * MIB,
        offload_delay_ms=0,
        disk_high_watermark=0.75,
        disk_low_watermark=0.50,
    )
    with ObjectStore(clock, cfg, lambda *a, **k: None) as store:
        store.put("other-client", MIB, {"rank": 7})
        for i in range(4):
            store.put(str(i), MIB, {"rank": 0})
            clock.advance(clock.now + 5_000_000)
        assert {k for k, r in store.objects.items() if r.disk} == {"other-client", "2", "3"}
        assert [c.disk_bytes for c in store.clients] == [2 * MIB, 0, 0, 0, 0, 0, 0, MIB]


def test_owner_staging_backpressure_and_two_stage_pipeline_across_target_ranks():
    clock, events = Clock(), []
    cfg = StoreConfig(
        8, 16 * MIB, 2 * MIB, 16 * MIB, offload_delay_ms=0, device_read_mib_s=1000, h2d_mib_s=100
    )
    with ObjectStore(clock, cfg, lambda op, **e: events.append((op, clock.now, e))) as store:
        for key in "abc":
            store.put(key, MIB, {"rank": 0})
        clock.advance(10_000_000)
        store.drop_published_memory({})
        start = clock.now
        for key, rank in (("a", 3), ("b", 3), ("c", 7)):
            store.get(key, MIB, {"rank": rank})
        assert store.clients[0].restore_staging_bytes == 2 * MIB
        clock.advance(start + 25_000_000)

        def times(event):
            return {e["key"]: (t - start) // 1_000_000 for op, t, e in events if op == event}

        assert times("staging_wait") == {"c": 0}
        assert times("disk_read_complete") == {"a": 1, "b": 2, "c": 12}
        assert times("h2d_start") == {"a": 1, "b": 11, "c": 12}
        assert times("get_complete") == {"a": 11, "b": 21, "c": 22}
        assert times("staging_release") == times("h2d_complete") == times("get_complete")
        clients = store.summary()["clients"]
        assert [c["peak_restore_staging_bytes"] for c in clients] == [2 * MIB] + [0] * 7
        assert all(c["restore_staging_bytes"] == 0 for c in clients)


def test_memory_and_disk_share_destination_only_once_and_do_not_reserve_an_unready_port():
    clock, events = Clock(), []
    cfg = StoreConfig(
        8,
        8 * MIB,
        4 * MIB,
        8 * MIB,
        offload_delay_ms=0,
        device_read_mib_s=100,
        memory_mib_s=500,
        h2d_mib_s=1000,
    )
    with ObjectStore(clock, cfg, lambda op, **e: events.append((op, clock.now, e))) as store:
        store.put("disk", MIB, {"rank": 0})
        clock.advance(2_000_000)
        store.drop_published_memory({})
        for key, owner in (("m1", 1), ("m2", 1), ("m3", 3)):
            store.put(key, MIB, {"rank": owner})
        start = clock.now
        for key, rank in (("disk", 6), ("m1", 6), ("m2", 2), ("m3", 7)):
            store.get(key, MIB, {"rank": rank})
        clock.advance(start + 15_000_000)
        ends = {e["key"]: t - start for op, t, e in events if op == "get_complete"}
        assert ends == {"m1": 2_000_000, "m3": 2_000_000, "m2": 4_000_000, "disk": 11_000_000}


def test_disk_eviction_waits_for_staging_waiter_to_read_but_can_overlap_its_h2d():
    clock, events = Clock(), []
    cfg = StoreConfig(
        1, 8 * MIB, MIB, MIB, offload_delay_ms=0, device_read_mib_s=1000, h2d_mib_s=100
    )
    with ObjectStore(clock, cfg, lambda op, **e: events.append((op, clock.now, e))) as store:
        store.put("old", MIB, {"rank": 0})
        clock.advance(2_000_000)
        store.drop_published_memory({})
        start = clock.now
        store.get("old", MIB, {"rank": 0})
        store.get("old", MIB, {"rank": 1})
        store.put("new", MIB, {"rank": 0})
        clock.advance(start + 25_000_000)
        evicted = [t - start for op, t, e in events if op == "evict_disk" and e["key"] == "old"]
        ends = [t - start for op, t, _ in events if op == "get_complete"]
        assert evicted == [12_000_000]
        assert ends == [11_000_000, 22_000_000]
        assert store.summary()["clients"][0]["restore_staging_bytes"] == 0


def test_reclaim_waits_for_all_bucket_readers_and_preserves_newly_queued_put():
    clock, events = Clock(), []
    cfg = StoreConfig(
        1, 16 * MIB, 2 * MIB, 2 * MIB, offload_delay_ms=0, device_mib_s=1000, h2d_mib_s=100
    )
    with ObjectStore(clock, cfg, lambda op, **e: events.append((op, clock.now, e))) as store:
        for key in ("a", "b"):
            assert store.put(key, MIB, {"rank": 0}) == 0
        clock.advance(5_000_000)
        assert store.objects["a"].bucket_id == store.objects["b"].bucket_id
        store.drop_published_memory({})
        start = clock.now
        store.get("a", MIB, {"rank": 0})
        store.get("b", MIB, {"rank": 1})
        assert store.put("c", MIB, {"rank": 0}) == 0
        clock.advance(start + 1_000_000)
        assert store.counts["evict_disk"] == 0
        assert store.put("d", MIB, {"rank": 0}) == 0
        clock.advance(start + 12_000_000)

        def timings(operation):
            return [(e["key"], t - start) for op, t, e in events if op == operation]

        assert timings("disk_read_complete") == [("a", 1_000_000), ("b", 2_000_000)]
        assert sorted(timings("evict_disk")) == [("a", 2_000_000), ("b", 2_000_000)]
        assert [(k, t) for k, t in timings("disk_publish") if k in ("c", "d")] == [
            ("c", 3_000_000),
            ("d", 4_000_000),
        ]
        assert timings("get_complete") == [("a", 11_000_000), ("b", 12_000_000)]
        assert set(store.objects) == {"c", "d"}
        assert store.summary()["clients"][0]["disk_bytes"] == 2 * MIB


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
    from kv_ssd_sim.simulation.worker import MetadataTransfers

    clock = Clock()
    rank = SimpleNamespace(
        clock=clock,
        store=SimpleNamespace(config=SimpleNamespace(h2d_mib_s=None)),
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


def test_staged_subbatches_and_rank_receiver_wait_for_actual_get_callbacks():
    pytest.importorskip("vllm")
    from kv_ssd_sim.simulation.worker import MetadataRank, MetadataTransfers

    clock, issued = Clock(), []
    store = SimpleNamespace(config=SimpleNamespace(h2d_mib_s=100), emit=lambda *a, **k: None)

    def get(key, size, origin, *, on_complete):
        issued.append((clock.now, origin["request"], origin["rank"], key))
        duration = (10 if key == "slow" else 1) + origin["rank"]
        clock.at(clock.now + duration, on_complete)

    store.get = get
    ranks = []
    for i in range(8):
        rank = MetadataRank.__new__(MetadataRank)
        rank.clock, rank.store, rank.port = clock, store, SimpleNamespace(rank=i)
        rank.worker = SimpleNamespace(_close_ended_store_requests=lambda *a: None)
        rank.completed_saves, rank.pending_loads = {}, deque()

        def describe(meta, rank=rank):
            origin = {"request": meta.req_id, "rank": rank.port.rank}
            rank.port.load_batches.extend(
                [
                    [("slow", 10, origin), ("fast", 1, origin)],
                    [("next", 1, origin)],
                ]
            )

        rank._describe_load = describe
        ranks.append(rank)
    transfers = MetadataTransfers(ranks)
    requests = [
        SimpleNamespace(
            req_id=rid,
            can_save=False,
            load_spec=SimpleNamespace(can_load=True, kvpool_cached_tokens=8, vllm_cached_tokens=0),
        )
        for rid in ("first", "second")
    ]
    transfers.submit(SimpleNamespace(requests=requests), set(), 0)
    clock.advance(24)
    assert not transfers.output().finished_recving
    clock.advance(25)
    assert transfers.output().finished_recving == {"first"}
    clock.advance(50)
    assert transfers.output().finished_recving == {"second"}
    assert not transfers.output().finished_recving
    assert [t for t, job, rank, key in issued if job == "first" and key == "next"] == list(
        range(10, 18)
    )
    assert (25, "second", 7, "slow") in issued


def test_native_shutdown_detects_errors_from_last_async_put():
    from kv_ssd_sim.online.server import ServingLoop

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
