import gzip
from importlib.util import find_spec
from pathlib import Path

import pytest

from kv_ssd_sim.devices.femu import launcher


def test_firmware_directory_before_kernel(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "ASSETS", tmp_path)
    modules = tmp_path / "modloop/modules"
    firmware = modules / "firmware"
    kernel = modules / "6.18.60-0-virt"
    firmware.mkdir(parents=True)
    kernel.mkdir()
    (firmware / "driver.bin").write_bytes(b"FIRMWARE_PAYLOAD")
    index = kernel / "modules.dep"
    index.write_text("ext4.ko: helper.ko\nhelper.ko:\nunused.ko:\n")
    for name in ("ext4", "helper", "unused"):
        (kernel / f"{name}.ko").write_bytes(f"PAYLOAD_{name}".encode())
    busybox = tmp_path / "guest-tools/usr/bin/busybox"
    busybox.parent.mkdir(parents=True)
    busybox.touch()
    (tmp_path / "alpine-initramfs").touch()
    iterdir = Path.iterdir
    monkeypatch.setattr(
        Path, "iterdir", lambda p: iter((firmware, kernel)) if p == modules else iterdir(p)
    )
    archive = gzip.decompress(launcher.initramfs().read_bytes())
    for entry in ("probe", "block_capture", "workload"):
        module = f"kv_ssd_sim.devices.femu.{entry}"
        assert f"-m {module} ".encode() in archive
        assert find_spec(module) is not None
    assert b"PAYLOAD_ext4" in archive and b"PAYLOAD_helper" in archive
    assert b"FIRMWARE_PAYLOAD" in archive and b"PAYLOAD_unused" not in archive
    (firmware / "modules.dep").touch()
    with pytest.raises(ValueError, match="Expected one kernel version"):
        launcher.initramfs()
    (firmware / "modules.dep").unlink()
    index.unlink()
    with pytest.raises(FileNotFoundError, match="No kernel dependency index"):
        launcher.initramfs()
