import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "quota", Path(__file__).resolve().parents[1] / "scripts/resume_after_quota.py"
)
quota = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quota)


def event(used=100, minutes=300, reset=2000000000, secondary=None):
    return {
        "type": "event_msg",
        "payload": {
            "type": "token_count",
            "rate_limits": {
                "primary": {"used_percent": used, "window_minutes": minutes, "resets_at": reset},
                "secondary": secondary,
            },
        },
    }


def test_reset_requires_exhausted_five_hour_window():
    assert quota.reset_from_event(event()) == 2000000000
    assert quota.reset_from_event(event(used=99)) is None
    assert quota.reset_from_event(event(minutes=60)) is None
    assert quota.reset_from_event(event(reset=None)) is None
    assert quota.reset_from_event(event(secondary={"used_percent": 100})) is None
    assert (
        quota.reset_from_event(event(secondary={"used_percent": 100, "resets_at": 2100000000}))
        == 2100000000
    )


def test_tail_does_not_consume_partial_record(tmp_path):
    path = tmp_path / "rollout.jsonl"
    data = json.dumps(event()).encode()
    path.write_bytes(data[:30])
    assert quota.read_events(path, 0) == ([], 0)
    path.write_bytes(data + b"\n")
    events, offset = quota.read_events(path, 0)
    assert events == [event()]
    assert quota.read_events(path, offset) == ([], offset)
