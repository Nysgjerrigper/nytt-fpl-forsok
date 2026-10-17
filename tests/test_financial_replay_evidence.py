"""Historical price reconstruction labels imputation without claiming deadline knowledge."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def price_evidence():
    path = Path(__file__).resolve().parents[1] / "experiments/financial_replay_2026-10-17/run.py"
    spec = importlib.util.spec_from_file_location("financial_replay_evidence", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.price_evidence


def test_price_reconstruction_matches_first_origin_and_labels_backward_default_sources():
    source = pd.DataFrame({"player_id": [1, 1, 2, 3, 3, 4], "GW": [1, 3, 2, 2, 2, 4],
        "origin_gw": [1, 3, 2, 2, 1, 4], "value": [float("nan"), 60, 40, 46, 45, 99]})
    before = source.copy(deep=True)
    table = price_evidence()(source, 1, 3).set_index(["player_id", "GW"])
    assert table.loc[(1, 1), "value"] == table.loc[(1, 2), "value"] == 50
    assert table.loc[(1, 2), "price_basis"] == "FORWARD_FILL"
    assert bool(table.loc[(1, 2), "raw_missing_default"])
    assert table.loc[(2, 1), "value"] == 40
    assert table.loc[(2, 1), "price_basis"] == "BACKWARD_FILL"
    assert table.loc[(2, 1), "source_GW"] == 2
    assert table.loc[(3, 2), "value"] == 45
    assert bool(table.loc[(3, 2), "price_copies_disagree"])
    assert 4 not in table.index.get_level_values("player_id")
    assert table.source_GW.max() == 3
    pd.testing.assert_frame_equal(source, before)


def test_price_reconstruction_does_not_silently_coerce_invalid_numeric_strings():
    with pytest.raises(ValueError):
        price_evidence()(pd.DataFrame({"player_id": [1], "GW": [1], "value": ["unknown"]}), 1, 1)
