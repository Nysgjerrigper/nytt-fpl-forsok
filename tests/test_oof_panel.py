"""Shared OOF panel identity and exact extraction contracts."""
import numpy as np
import pandas as pd
import pytest

from fpl.model import oof_panel, screen


def panel():
    return pd.DataFrame([
        {"player_id": 1, "position": "MID", "GW_global": 5, "expert": expert,
         "prediction": 3.0, "actual_total_points": 4.0, "mase_scale": 2.0,
         "train_max_gw": 2, "seed": 0, "params_hash": expert,
         "stage": "selection", "fixture_count": 2}
        for expert in ("lightgbm_l2", "catboost_rmse")
    ])


def test_screen_exports_the_unchanged_shared_panel_validator():
    assert screen.validate_oof_panel is oof_panel.validate_oof_panel
    oof_panel.validate_oof_panel(panel(), positions=("MID",))


@pytest.mark.parametrize("field,value", [("actual_total_points", 99), ("mase_scale", 3),
                                          ("train_max_gw", 3), ("seed", 1),
                                          ("fixture_count", 1), ("prediction", np.inf)])
def test_shared_panel_rejects_inconsistent_or_nonfinite_evidence(field, value):
    oof = panel()
    oof.loc[0, field] = value
    with pytest.raises(ValueError):
        oof_panel.validate_oof_panel(oof, positions=("MID",))
