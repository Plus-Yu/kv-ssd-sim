import json

import pytest

from kv_ssd_sim.hybrid import HybridConfig


def test_geometry_and_namespaces(tmp_path):
    p = tmp_path / "geometry.json"
    p.write_text(json.dumps({"layers": ["mla", "kda", "kda"], "kda_head_dim": 64}))
    config = HybridConfig.load(p)
    assert config.page_bytes >= 2 * 64 * 64 * 4 + 3 * 2 * 64 * 3 * 2
    assert config.fingerprint != HybridConfig().fingerprint


@pytest.mark.parametrize(
    "values",
    [
        {"layers": ["mla", "mla"]},
        {"conv_kernel": 1},
        {"recurrent_dtype": "float16"},
        {"kda_heads": -1},
    ],
)
def test_invalid_geometry_rejected(tmp_path, values):
    p = tmp_path / "geometry.json"
    p.write_text(json.dumps(values))
    with pytest.raises(ValueError):
        HybridConfig.load(p)
