"""The opt-in weekly CLI admits new owned codes with a deadline-bound global origin."""
from datetime import datetime, timezone
import sys

import pandas as pd
import pytest

from fpl import run_week
from fpl.model import cold_start


def test_cold_start_cli_uses_target_origin_and_complete_registered_pool(monkeypatch, tmp_path):
    monkeypatch.setattr(run_week.config, "LIVE_FORECAST_DIR", tmp_path / "captures")
    bootstrap = {
        "events": [{"id": 1, "deadline_time": "2026-08-15T10:00:00Z", "finished": True},
                   {"id": 2, "deadline_time": "2026-11-01T10:00:00Z", "finished": False}],
        "teams": [{"id": 1, "name": "A"}],
        "elements": [{"id": i, "code": i, "team": 1, "element_type": 3, "now_cost": 50}
                     for i in range(1, 17)],
    }
    raw = pd.DataFrame({"player_id": range(1, 16), "GW_global": 229,
                        "team": "A", "position": "MID", "value": 50.})
    priors = object()
    monkeypatch.setattr(sys, "argv", ["run_week", "--team-id", "42", "--free-transfers", "1",
                                     "--horizon", "1", "--cold-start-priors"])
    monkeypatch.setattr(run_week, "_utc_now", lambda now: datetime(2026, 10, 31, tzinfo=timezone.utc))
    monkeypatch.setattr(run_week, "fetch_bootstrap", lambda: bootstrap)
    monkeypatch.setattr(run_week.fetch, "build_master_dataset", lambda: None)
    monkeypatch.setattr(run_week.pd, "read_csv", lambda *a, **k: raw.copy())
    monkeypatch.setattr(run_week.features, "build_feature_frame", lambda frame: frame)
    monkeypatch.setattr(run_week.features, "feature_columns", lambda frame: ["value"])
    monkeypatch.setattr(run_week.features, "team_form_asof", lambda frame: None)
    monkeypatch.setattr(run_week, "build_live_snapshot", lambda frame: frame.copy())
    monkeypatch.setattr(run_week, "apply_live_identity", lambda frame, register: frame)
    monkeypatch.setattr(run_week, "fit_holdout_weights", lambda *a, **k: {})
    monkeypatch.setattr(run_week, "fit_position_ensembles", lambda *a, **k: {})

    def fit(history, origin_gw):
        assert origin_gw == 230 and history.GW_global.max() < origin_gw
        return priors

    def owned(team, register, known):
        assert team == 42 and known == set(range(1, 17))
        return [*range(1, 15), 16], 0.

    class ReachedPredictions(Exception):
        pass

    def predict(snapshot, *args, **kwargs):
        assert snapshot.GW_global.eq(230).all()
        assert snapshot.loc[snapshot.is_cold_start, "player_id"].tolist() == [16]
        assert kwargs["cold_start_priors"] is priors
        raise ReachedPredictions

    monkeypatch.setattr(cold_start, "fit_pooled_priors", fit)
    monkeypatch.setattr(run_week, "get_user_squad", owned)
    monkeypatch.setattr(run_week, "build_future_predictions", predict)
    with pytest.raises(ReachedPredictions):
        run_week.main()
