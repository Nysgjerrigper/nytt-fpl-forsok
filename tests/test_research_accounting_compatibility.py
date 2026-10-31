"""Historical labelled net exports retain one hit deduction under current admission."""
import pandas as pd
import pytest

from fpl.milp.compare_backtests import paired_gw_differences, scored_points


def _legacy_net_export():
    return pd.DataFrame({"gameweek": [153, 154], "actual_total_points": [46., 52.],
                         "gross_actual_total_points": [50., 60.], "alpha": [1, 2],
                         "actual_net_points": [46., 52.], "transfer_hit_points": [4, 8],
                         "points_accounting": "net_after_transfer_hits",
                         "scoring_protocol": "autosubs_v2"})


def test_labelled_historical_net_schema_does_not_pay_hits_twice():
    frame = _legacy_net_export()
    assert scored_points(frame).tolist() == [46., 52.]
    assert scored_points(frame, points_col="actual_net_points").tolist() == [46., 52.]
    candidate = frame.copy()
    for column in ("actual_total_points", "gross_actual_total_points", "actual_net_points"):
        candidate[column] += 1
    assert paired_gw_differences(candidate, frame).tolist() == [1., 1.]


@pytest.mark.parametrize("column", ["actual_total_points", "gross_actual_total_points", "actual_net_points"])
def test_inconsistent_historical_net_schema_fails_closed(column):
    frame = _legacy_net_export()
    frame.loc[0, column] += 1
    with pytest.raises(ValueError, match="net points|actual_net_points"):
        scored_points(frame)


def test_canonical_gross_schema_retains_main_contract():
    frame = _legacy_net_export().drop(columns=["gross_actual_total_points", "points_accounting"])
    frame["actual_total_points"] = [50., 60.]
    assert scored_points(frame).tolist() == [50., 60.]
    assert scored_points(frame, points_col="actual_net_points").tolist() == [46., 52.]
