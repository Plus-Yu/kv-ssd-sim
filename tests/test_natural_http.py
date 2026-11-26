"""Natural FEMU accepts absent reads while controlled-cold verification remains strict."""

import pytest

from kv_ssd_sim.analysis.natural import restored_batches
from kv_ssd_sim.devices.femu.device import FEMUDevice


def test_natural_namespace_without_get_or_sectors(monkeypatch):
    device = object.__new__(FEMUDevice)
    monkeypatch.setattr(device, "describe", lambda: {"device": "nvme0n1"})
    for samples in ([], [{"delta": {"read_sectors": 0}}]):
        device.samples = samples
        assert device.verify(require_disk_read=False)["get_io_samples"] == samples
        with pytest.raises(RuntimeError, match="No FEMU block reads"):
            device.verify()


def test_natural_no_get_and_multiple_tiers():
    assert restored_batches([]) == []
    events = []
    for i, (tiers, sectors) in enumerate(
        [(["MEMORY"], 0), (["LOCAL_DISK"], 8), (["MEMORY", "LOCAL_DISK"], 0)]
    ):
        key = f"object-{i}"
        events.extend(
            [
                {"seq": i * 4, "event": "store_get_begin", "keys": [key], "tiers": {key: tiers}},
                {"seq": i * 4 + 1, "event": "native_get_begin", "keys": [key]},
                {
                    "seq": i * 4 + 2,
                    "event": "femu_block_io_during_get",
                    "device": "nvme0n1",
                    "delta": {"read_sectors": sectors},
                },
                {
                    "seq": i * 4 + 3,
                    "event": "store_get_end",
                    "keys": [key],
                    "result": [4096],
                    "checksum_ok": [True],
                },
            ]
        )
    rows = restored_batches(events)
    assert [r["device_read_bytes_in_get"] for r in rows] == [0, 4096, 0]
    assert rows[-1]["available_replicas_by_object"] == {"LOCAL_DISK+MEMORY": 1}
    events[-1]["checksum_ok"] = [False]
    with pytest.raises(ValueError, match="restored object"):
        restored_batches(events)
