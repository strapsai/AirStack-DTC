from pathlib import Path

import yaml


def test_config_loads():
    config_path = Path(__file__).parents[1] / "config" / "agile_autonomy_airstack.yaml"
    data = yaml.safe_load(config_path.read_text())

    params = data["/**"]["ros__parameters"]
    assert params["enabled_default"] is False
    assert params["backend_type"] == "mock"
    assert params["max_speed_mps"] <= 2.0
