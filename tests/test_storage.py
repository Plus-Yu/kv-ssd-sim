import ctypes

from kv_ssd_sim.events import EventLog
from kv_ssd_sim.storage import ObservedStore


class MemoryDescriptor:
    def is_memory_replica(self):
        return True


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
