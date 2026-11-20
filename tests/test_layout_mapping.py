import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_layout import Span, allocation_state, intersect, shared_units


def test_byte_splits_and_unknown_lifetimes():
    spans = [Span(50, 80, "MLA", 7), Span(80, 95, "bucket_key", 7), Span(95, 140, "KDA", 11)]
    ends = [s.stop for s in spans]
    assert intersect(spans, ends, 0, 125, 11) == {
        "unmapped": 50,
        "MLA": 30,
        "bucket_key": 15,
        "lifetime_unverified": 30,
    }
    assert intersect(spans, ends, 70, 145, 12) == {
        "MLA": 10,
        "bucket_key": 15,
        "KDA": 45,
        "unmapped": 5,
    }
    assert shared_units(spans, 512) == 1


def test_ext4_block_reallocation(tmp_path):
    path = tmp_path / "trace"
    path.write_text(
        "# entries-in-buffer/entries-written: 3/3\n"
        "x 1.000001: ext4_allocate_blocks: dev 259,0 ino 7 flags x len 2 block 5 lblk 1 goal 5\n"
        "x 1.000002: ext4_free_blocks: dev 259,0 ino 7 mode 0 block 5 count 2 flags x\n"
        "x 1.000003: ext4_allocate_blocks: dev 259,0 ino 9 flags x len 1 block 5 lblk 0 goal 5\n"
    )
    inode, logical, stamp = allocation_state(path, "259,0", 10)
    assert (inode[5], logical[5], stamp[5]) == (9, 0, 1000003000)
    assert inode[6] == 0


def test_inventory_rejects_payload_changes(tmp_path, monkeypatch):
    import hashlib

    import pytest

    from kv_ssd_sim import layout

    monkeypatch.setattr(layout, "extent_map", lambda _: [[0, 4096, 4096]])
    key = "k3-online-tp8@store_format:attention_head_major@group:0@" + "a" * 64
    header = b"default\0" + key.encode()
    path = tmp_path / "1.bucket"
    path.write_bytes(header + b"abc")
    expected = {key: (3, hashlib.sha256(b"abc").hexdigest())}
    obj = layout.inventory(tmp_path, expected)[0]["objects"][0]
    assert obj["header_offset"] == 0 and obj["file_offset"] == len(header)
    path.write_bytes(header + b"abd")
    with pytest.raises(ValueError, match="digest"):
        layout.inventory(tmp_path, expected)
    path.write_bytes(header + b"ab")
    with pytest.raises(ValueError, match="Truncated"):
        layout.inventory(tmp_path, expected)
