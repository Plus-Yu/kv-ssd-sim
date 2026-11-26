import json

import pytest

from kv_ssd_sim.analysis.nand import interval, sample


def save_snapshot(path, name, stamp, sectors=0, data=0, gc=0, erases=0, valid=0, invalid=0):
    stat = [0] * 17
    stat[6] = sectors
    guest = {
        "label": name,
        "before_ns": stamp,
        "after_ns": stamp + 1,
        "stat_before": stat,
        "stat_after": stat,
        "namespace_sectors": 512,
    }
    ns = {
        "nsid": 1,
        "mode": "bbssd",
        "offset": 0,
        "geometry": {
            "pages-per-line": 64,
            "pages-per-block": 64,
            "page-size": 4096,
            "channels": 1,
            "luns-per-channel": 1,
            "planes-per-lun": 1,
            "blocks-per-plane": 1,
        },
        "counters": {
            "host-write-pages": data,
            "nand-write-pages": data,
            "gc-write-pages": gc,
            "block-erases": erases,
        },
        "lines": [
            {"id": 0, "vpc": valid, "ipc": invalid, "erase-min": erases, "erase-max": erases}
        ],
        "line-counts": {"total": 1},
    }
    (path / f"nand-{name}.guest.json").write_text(json.dumps(guest))
    (path / f"nand-{name}.qmp.json").write_text(json.dumps({"mode": "bbssd", "namespaces": [ns]}))


def test_window_uses_sector_bytes_and_counts_gc_once(tmp_path):
    save_snapshot(tmp_path, "a", 1, 8, 1, valid=1)
    save_snapshot(tmp_path, "b", 4, 9, 2, valid=1, invalid=1)
    a, b = (sample(tmp_path, n) for n in ("a", "b"))
    assert interval(a, b)["byte_waf"] == 8
    save_snapshot(tmp_path, "c", 7, 9, 2, 62, 1, valid=0)
    result = interval(b, sample(tmp_path, "c"))
    assert result["nand_program_bytes"] == result["gc_relocation_bytes"] == 62 * 4096
    assert result["byte_waf"] is None


def test_reject_incomplete_line_inventory_and_busy_boundaries(tmp_path):
    save_snapshot(tmp_path, "a", 1)
    path = tmp_path / "nand-a.qmp.json"
    raw = json.loads(path.read_text())
    raw["namespaces"][0]["next-offset"] = 1
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="Incomplete"):
        sample(tmp_path, "a")
    save_snapshot(tmp_path, "a", 1)
    p = tmp_path / "nand-a.guest.json"
    raw = json.loads(p.read_text())
    raw["stat_after"][6] = 8
    p.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="crossed"):
        sample(tmp_path, "a")


def test_reject_impossible_occupancy(tmp_path):
    save_snapshot(tmp_path, "a", 1)
    save_snapshot(tmp_path, "b", 4, sectors=8, data=1, valid=0)
    with pytest.raises(ValueError, match="occupied"):
        interval(sample(tmp_path, "a"), sample(tmp_path, "b"))
