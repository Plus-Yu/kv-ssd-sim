import ctypes
from unittest.mock import Mock

import pytest

from kv_ssd_sim.events import EventLog, load_events
from kv_ssd_sim.storage import ObservedStore


class MemoryDescriptor:
    def is_memory_replica(self):
        return True


def test_final_snapshot_waits_for_every_local_disk_replica(tmp_path):
    memory = MemoryDescriptor()
    local = Mock()
    local.is_memory_replica.return_value = False
    local.is_local_disk_replica.return_value = True
    partial = {"a": [local], "b": [memory]}
    complete = {"a": [local], "b": [memory, local]}
    native = Mock()
    native.batch_get_replica_desc.side_effect = [partial, partial, complete]
    with EventLog(tmp_path / "events.jsonl") as events:
        store = ObservedStore(native, events)
        keys = ["a", "b"]
        assert store.replica_snapshot(keys)["b"] == ["MEMORY"]
        assert store.replica_snapshot(keys, disk_types={"LOCAL_DISK"}) == {
            "a": ["LOCAL_DISK"],
            "b": ["MEMORY", "LOCAL_DISK"],
        }
    assert native.batch_get_replica_desc.call_count == 3


def test_local_disk_snapshot_timeout_records_unconfirmed_keys(tmp_path):
    remote = Mock()
    remote.is_memory_replica.return_value = False
    remote.is_local_disk_replica.return_value = False
    remote.is_disk_replica.return_value = True
    native = Mock()
    native.batch_get_replica_desc.return_value = {"a": [remote]}
    path = tmp_path / "events.jsonl"
    with EventLog(path) as events:
        store = ObservedStore(native, events)
        with pytest.raises(TimeoutError, match="2/2 missing"):
            store.replica_snapshot(["a", "b"], disk_types={"LOCAL_DISK"}, timeout_s=0)
        assert store.disk_barrier(["a"]) == ["a"]
    timeout, confirmed = load_events(path)
    assert timeout["missing_disk_keys"] == ["a", "b"]
    assert timeout["tiers"] == {"a": ["DISK"], "b": []}
    assert confirmed["keys"] == ["a"]


class CorruptingStore:
    def batch_put_from_multi_buffers(self, keys, addresses, sizes, config):
        return [0] * len(keys)

    def batch_get_replica_desc(self, keys):
        return {key: [MemoryDescriptor()] for key in keys}

    def batch_get_into_multi_buffers(self, keys, addresses, sizes):
        for addrs, byte_sizes in zip(addresses, sizes, strict=True):
            for addr, size in zip(addrs, byte_sizes, strict=True):
                ctypes.memset(addr, 0, size)
        return [sum(s) for s in sizes]


def test_corrupt_restored_bytes_fail_verification(tmp_path):
    source = ctypes.create_string_buffer(b"hello", 5)
    dest = ctypes.create_string_buffer(5)
    with EventLog(tmp_path / "events.jsonl") as events:
        store = ObservedStore(CorruptingStore(), events)
        store.batch_put_from_multi_buffers(["key"], [[ctypes.addressof(source)]], [[5]], None)
        store.batch_get_into_multi_buffers(["key"], [[ctypes.addressof(dest)]], [[5]])
        assert len(store.errors) == 1
        assert "SHA256" in store.errors[0]


def test_failed_write_is_not_reported_as_verified_storage(tmp_path):
    class FailingStore(CorruptingStore):
        def batch_put_from_multi_buffers(self, keys, addresses, sizes, config):
            return [-1] * len(keys)

    source = ctypes.create_string_buffer(b"hello", 5)
    with EventLog(tmp_path / "events.jsonl") as events:
        store = ObservedStore(FailingStore(), events)
        store.batch_put_from_multi_buffers(["key"], [[ctypes.addressof(source)]], [[5]], None)
        assert store.expected == {}
        assert "Native PUT failed" in store.errors[0]
