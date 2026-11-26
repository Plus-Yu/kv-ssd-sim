from concurrent.futures import ThreadPoolExecutor

from kv_ssd_sim.core.events import EventLog, load_events


def test_buffered_log_closes_with_complete_ordered_thread_events(tmp_path):
    path = tmp_path / "events.jsonl"
    with EventLog(path, flush_every=64) as log, ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda i: log.emit("test", value=i), range(137)))
    events = load_events(path)
    assert [e["seq"] for e in events] == list(range(1, 138))
    assert sorted(e["value"] for e in events) == list(range(137))
    assert [e["time_ns"] for e in events] == sorted(e["time_ns"] for e in events)
