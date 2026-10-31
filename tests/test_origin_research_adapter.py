"""Origin-horizon admission for separately registered player-history adapters."""
import numpy as np
import pandas as pd
import pytest

from fpl import features
from fpl.model import predict
from fpl.model.sequence_features import add_sequence_features, sequence_columns, sequence_snapshot
from fpl.run_week import build_live_snapshot


def raw_history():
    frame = pd.DataFrame({"player_id": [1] * 6, "name": ["A"] * 6,
                          "position": ["MID"] * 6, "team": ["AAA"] * 6,
                          "opponent_team": ["BBB"] * 6, "GW_global": np.arange(1, 7),
                          "fixture": np.arange(101, 107), "was_home": [True] * 6,
                          "value": [50.] * 6, "minutes": [90.] * 6,
                          "total_points": np.arange(1., 7.), "xP": [3.] * 6})
    for column in features.FORM_STATS:
        if column not in frame:
            frame[column] = 0.
    return frame


def snapshot(history):
    ordinary = build_live_snapshot(history)
    seq = sequence_snapshot(history).set_index("player_id")
    aligned = pd.DataFrame(seq.loc[ordinary.player_id].to_numpy(),
                           columns=sequence_columns(), index=ordinary.index)
    return pd.concat([ordinary, aligned], axis=1)


def forecasts(raw, monkeypatch, **kwargs):
    class Echo:
        def predict(self, X):
            return X.cnn_total_points_t7.to_numpy()
    monkeypatch.setattr(predict, "fit_position_ensembles", lambda *args, **kwargs: {"MID": Echo()})
    sequence = add_sequence_features(raw)
    frame = features.build_feature_frame(raw)
    values = sequence.set_index(["player_id", "GW_global"]).loc[
        pd.MultiIndex.from_frame(frame[["player_id", "GW_global"]]), sequence_columns()].to_numpy()
    frame = pd.concat([frame, pd.DataFrame(values, columns=sequence_columns(), index=frame.index)], axis=1)
    columns = features.feature_columns(frame) + sequence_columns()
    return predict.origin_based_predictions(frame, raw, columns, 4, 6, 3,
                                            weight_strategy="single:lightgbm", **kwargs)


def test_all_sequence_steps_freeze_at_origin_for_every_horizon(monkeypatch):
    raw = raw_history()
    base = forecasts(raw, monkeypatch, snapshot_builder=snapshot, extra_player_features=sequence_columns())
    changed = raw.copy()
    changed.loc[changed.GW_global >= 4, "total_points"] = 999.
    after = forecasts(changed, monkeypatch, snapshot_builder=snapshot, extra_player_features=sequence_columns())
    a, b = base[base.origin_gw == 4], after[after.origin_gw == 4]
    assert a.GW.tolist() == [4, 5, 6]
    np.testing.assert_array_equal(a.predicted_total_points, [3., 3., 3.])
    np.testing.assert_array_equal(a.predicted_total_points, b.predicted_total_points)
    assert (b.actual_total_points == 999).all()
    assert base.loc[base.origin_gw == 5, "predicted_total_points"].eq(4.).all()


def test_extra_history_requires_rebuild_and_declared_columns(monkeypatch):
    with pytest.raises(ValueError, match="require an origin snapshot builder"):
        forecasts(raw_history(), monkeypatch, extra_player_features=sequence_columns())
    with pytest.raises(ValueError, match="unique fitted frame columns"):
        forecasts(raw_history(), monkeypatch, snapshot_builder=snapshot, extra_player_features=["unknown"])


def test_missing_rebuilt_history_fails_instead_of_retaining_future_form(monkeypatch):
    with pytest.raises(ValueError, match="frozen features"):
        forecasts(raw_history(), monkeypatch, snapshot_builder=build_live_snapshot,
                  extra_player_features=sequence_columns())


def test_ambiguous_snapshot_identity_is_rejected(monkeypatch):
    def duplicate(history):
        frame = snapshot(history)
        return pd.concat([frame, frame], ignore_index=True)
    with pytest.raises(ValueError, match="unambiguous"):
        forecasts(raw_history(), monkeypatch, snapshot_builder=duplicate,
                  extra_player_features=sequence_columns())


def test_dated_snapshot_receives_actual_origin_and_freezes_entire_horizon(monkeypatch):
    calls = []
    def dated(history, origin):
        calls.append((origin, history.GW_global.max()))
        rebuilt = snapshot(history)
        rebuilt['cnn_total_points_t7'] = origin * 10.
        return rebuilt
    raw = raw_history().loc[lambda x: x.GW_global != 3].copy()
    result = forecasts(raw, monkeypatch, origin_snapshot_builder=dated,
                       extra_player_features=sequence_columns())
    assert calls == [(4, 2), (5, 4), (6, 5)]
    assert result.loc[result.origin_gw == 4, 'predicted_total_points'].tolist() == [40., 40., 40.]
    assert result.loc[result.origin_gw == 5, 'predicted_total_points'].tolist() == [50., 50.]


def test_dated_snapshot_rejects_ambiguous_or_noncallable_builder(monkeypatch):
    with pytest.raises(ValueError, match='mutually exclusive'):
        forecasts(raw_history(), monkeypatch, snapshot_builder=snapshot,
                  origin_snapshot_builder=lambda history, origin: snapshot(history))
    with pytest.raises(ValueError, match='dated origin snapshot builder must be callable'):
        forecasts(raw_history(), monkeypatch, origin_snapshot_builder=123)
