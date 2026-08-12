"""Leakage-safe, interpretable routing for experimental MID experts.

The gate is deliberately narrow: it may route only on the deadline-known
``mins60_rate_roll5`` feature.  Thresholds are learned from a training frame, while
expert assignments are selected on a later causal validation frame.  Model fitting is
outside this module; callers supply frozen candidate predictions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from fpl.model.metrics import mase, naive_lag1_scale


ROUTE_FEATURE = "mins60_rate_roll5"
REGIMES = ("low", "medium", "high")
FORBIDDEN_ROUTING_COLUMNS = frozenset(
    {
        "minutes",
        "target_minutes",
        "actual_minutes",
        "player_id",
        "element",
        "element_code",
        "name",
        "web_name",
    }
)


def _finite_1d(values: Sequence[float], label: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite one-dimensional array")
    return array


def validate_routing_features(features: Sequence[str]) -> None:
    """Reject routing inputs other than the one pre-registered deadline feature.

    This explicit allow-list prevents a later integration from silently adding actual
    target-gameweek minutes or player identity to the router.
    """

    supplied = tuple(features)
    forbidden = sorted(set(supplied) & FORBIDDEN_ROUTING_COLUMNS)
    if forbidden:
        raise ValueError(f"forbidden MID routing columns: {', '.join(forbidden)}")
    if supplied != (ROUTE_FEATURE,):
        raise ValueError(f"MID gate must route only on {ROUTE_FEATURE!r}")


@dataclass(frozen=True)
class DiversityEligibility:
    """Audit record for the pre-registered two-expert diversity trigger."""

    eligible: bool
    champion_mase: float
    challenger_mase: float
    challenger_mase_ratio: float
    residual_correlation: float
    champion_gw_wins: int
    challenger_gw_wins: int
    selection_gameweeks: int
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable audit record."""

        result = asdict(self)
        result["reasons"] = list(self.reasons)
        return result


def check_diversity_eligibility(
    y_true: Sequence[float],
    champion_predictions: Sequence[float],
    challenger_predictions: Sequence[float],
    gameweeks: Sequence[int],
    mase_scale: float,
    *,
    max_mase_ratio: float = 1.03,
    max_residual_correlation: float = 0.95,
    min_wins_per_expert: int = 4,
    required_gameweeks: int = 16,
) -> DiversityEligibility:
    """Evaluate whether two frozen MID experts are accurate and diverse enough.

    Gameweek wins compare each expert's pooled MAE within a gameweek. Ties count for
    neither expert. The check requires exactly the pre-registered 16 selection weeks so
    a truncated evaluation cannot accidentally pass the gate.
    """

    truth = _finite_1d(y_true, "y_true")
    champion = _finite_1d(champion_predictions, "champion_predictions")
    challenger = _finite_1d(challenger_predictions, "challenger_predictions")
    weeks = np.asarray(gameweeks)
    if not (len(truth) == len(champion) == len(challenger) == len(weeks)):
        raise ValueError("truth, predictions, and gameweeks must have equal lengths")
    if len(truth) < 2:
        raise ValueError("at least two validation rows are required")
    if not np.isfinite(mase_scale) or mase_scale <= 0:
        raise ValueError("mase_scale must be finite and positive")

    unique_weeks = pd.unique(weeks)
    champion_mase = mase(truth, champion, mase_scale)
    challenger_mase = mase(truth, challenger, mase_scale)
    ratio = challenger_mase / champion_mase if champion_mase > 0 else float("inf")
    champion_residual = truth - champion
    challenger_residual = truth - challenger
    if np.std(champion_residual) == 0 or np.std(challenger_residual) == 0:
        correlation = float("nan")
    else:
        correlation = float(np.corrcoef(champion_residual, challenger_residual)[0, 1])

    champion_wins = 0
    challenger_wins = 0
    for week in unique_weeks:
        mask = weeks == week
        champion_error = float(np.mean(np.abs(truth[mask] - champion[mask])))
        challenger_error = float(np.mean(np.abs(truth[mask] - challenger[mask])))
        if champion_error < challenger_error:
            champion_wins += 1
        elif challenger_error < champion_error:
            challenger_wins += 1

    reasons: list[str] = []
    if len(unique_weeks) != required_gameweeks:
        reasons.append(
            f"expected {required_gameweeks} selection gameweeks, got {len(unique_weeks)}"
        )
    if not np.isfinite(ratio) or ratio > max_mase_ratio:
        reasons.append(f"challenger MASE ratio {ratio:.6g} exceeds {max_mase_ratio:.6g}")
    if not np.isfinite(correlation) or correlation >= max_residual_correlation:
        reasons.append(
            "residual correlation is not finite or is at least "
            f"{max_residual_correlation:.6g}"
        )
    if champion_wins < min_wins_per_expert:
        reasons.append(f"champion wins {champion_wins} gameweeks; need {min_wins_per_expert}")
    if challenger_wins < min_wins_per_expert:
        reasons.append(
            f"challenger wins {challenger_wins} gameweeks; need {min_wins_per_expert}"
        )

    return DiversityEligibility(
        eligible=not reasons,
        champion_mase=champion_mase,
        challenger_mase=challenger_mase,
        challenger_mase_ratio=ratio,
        residual_correlation=correlation,
        champion_gw_wins=champion_wins,
        challenger_gw_wins=challenger_wins,
        selection_gameweeks=len(unique_weeks),
        reasons=tuple(reasons),
    )


@dataclass(frozen=True)
class TertileThresholds:
    """Training-only boundaries for low, medium, and high nailedness."""

    low_upper: float
    medium_upper: float

    def __post_init__(self) -> None:
        if not np.isfinite([self.low_upper, self.medium_upper]).all():
            raise ValueError("tertile thresholds must be finite")
        if self.low_upper > self.medium_upper:
            raise ValueError("low threshold cannot exceed medium threshold")

    def route(self, values: Sequence[float]) -> np.ndarray:
        """Return deterministic regime labels; missing live values use the low regime."""

        array = np.asarray(values, dtype=float)
        return np.where(
            np.isnan(array) | (array <= self.low_upper),
            "low",
            np.where(array <= self.medium_upper, "medium", "high"),
        )


def fit_tertile_thresholds(
    training_frame: pd.DataFrame, *, feature: str = ROUTE_FEATURE
) -> TertileThresholds:
    """Fit nailedness tertiles using training rows only.

    The caller owns the chronological split. Requiring a training-named argument and
    accepting no validation frame makes the data dependency explicit and testable.
    """

    validate_routing_features((feature,))
    if feature not in training_frame:
        raise ValueError(f"training frame is missing routing feature {feature!r}")
    observed = pd.to_numeric(training_frame[feature], errors="coerce").dropna()
    if observed.empty:
        raise ValueError("cannot fit tertiles without observed training values")
    low, high = observed.quantile([1 / 3, 2 / 3]).to_numpy(dtype=float)
    return TertileThresholds(float(low), float(high))


@dataclass(frozen=True)
class RegimeSelection:
    """Validation evidence and chosen expert for one nailedness regime."""

    expert: str
    rows: int
    gameweeks: int
    champion_mase: float | None
    selected_mase: float | None
    improvement: float
    fallback_reason: str | None = None


@dataclass(frozen=True)
class MidGateConfig:
    """Serializable result of causal regime selection."""

    champion: str
    candidates: tuple[str, ...]
    thresholds: TertileThresholds
    selections: Mapping[str, RegimeSelection]
    route_feature: str = ROUTE_FEATURE
    min_rows: int = 500
    min_gameweeks: int = 8
    min_mase_improvement: float = 0.02
    mase_scale: float | None = None
    mase_scale_training_max_gw: int | None = None
    training_max_gw: int | None = None
    validation_min_gw: int | None = None
    validation_max_gw: int | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_routing_features((self.route_feature,))
        if self.champion not in self.candidates:
            raise ValueError("champion must be included in candidates")
        if set(self.selections) != set(REGIMES):
            raise ValueError(f"selections must contain exactly {REGIMES}")
        unknown = {item.expert for item in self.selections.values()} - set(self.candidates)
        if unknown:
            raise ValueError(f"selected unknown experts: {sorted(unknown)}")

    def route(self, frame: pd.DataFrame) -> np.ndarray:
        """Route rows using only the configured deadline-known feature."""

        if self.route_feature not in frame:
            raise ValueError(f"prediction frame is missing {self.route_feature!r}")
        regimes = self.thresholds.route(frame[self.route_feature].to_numpy())
        return np.asarray([self.selections[regime].expert for regime in regimes])

    def predict(
        self, frame: pd.DataFrame, predictions: Mapping[str, Sequence[float]]
    ) -> np.ndarray:
        """Select supplied frozen predictions according to deterministic routing."""

        missing = set(self.candidates) - set(predictions)
        if missing:
            raise ValueError(f"missing candidate predictions: {sorted(missing)}")
        arrays = {name: _finite_1d(predictions[name], name) for name in self.candidates}
        if any(len(values) != len(frame) for values in arrays.values()):
            raise ValueError("every candidate prediction array must match the frame length")
        experts = self.route(frame)
        result = np.empty(len(frame), dtype=float)
        for expert in self.candidates:
            mask = experts == expert
            result[mask] = arrays[expert][mask]
        return result

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration and provenance record."""

        return {
            "champion": self.champion,
            "candidates": list(self.candidates),
            "thresholds": asdict(self.thresholds),
            "selections": {name: asdict(value) for name, value in self.selections.items()},
            "route_feature": self.route_feature,
            "min_rows": self.min_rows,
            "min_gameweeks": self.min_gameweeks,
            "min_mase_improvement": self.min_mase_improvement,
            "mase_scale": self.mase_scale,
            "mase_scale_training_max_gw": self.mase_scale_training_max_gw,
            "training_max_gw": self.training_max_gw,
            "validation_min_gw": self.validation_min_gw,
            "validation_max_gw": self.validation_max_gw,
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MidGateConfig":
        """Restore a configuration produced by :meth:`to_dict`."""

        values = dict(payload)
        values["candidates"] = tuple(values["candidates"])
        values["thresholds"] = TertileThresholds(**values["thresholds"])
        values["selections"] = {
            name: RegimeSelection(**selection)
            for name, selection in values["selections"].items()
        }
        return cls(**values)


def select_mid_gate(
    training_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    candidate_prediction_columns: Mapping[str, str],
    *,
    champion: str,
    target_column: str = "total_points",
    gameweek_column: str = "GW_global",
    min_rows: int = 500,
    min_gameweeks: int = 8,
    min_mase_improvement: float = 0.02,
    training_max_gw: int | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> MidGateConfig:
    """Choose a frozen expert per training-derived nailedness regime.

    Sparse regimes and improvements below the pre-registered absolute MASE threshold
    fall back to the positional champion. Candidate order is preserved for deterministic
    ties, with the champion evaluated first.
    """

    if champion not in candidate_prediction_columns:
        raise ValueError("champion must be present in candidate predictions")
    if validation_frame.empty:
        raise ValueError("validation frame cannot be empty")
    training_required = {ROUTE_FEATURE, target_column, gameweek_column}
    training_missing = training_required - set(training_frame.columns)
    if training_missing:
        raise ValueError(f"training frame is missing columns: {sorted(training_missing)}")
    required = {ROUTE_FEATURE, target_column, gameweek_column} | set(
        candidate_prediction_columns.values()
    )
    missing = required - set(validation_frame.columns)
    if missing:
        raise ValueError(f"validation frame is missing columns: {sorted(missing)}")
    if training_max_gw is not None and gameweek_column in training_frame:
        observed_training_max = int(training_frame[gameweek_column].max())
        if observed_training_max > training_max_gw:
            raise ValueError("training frame exceeds the declared chronological cutoff")
        validation_min = int(validation_frame[gameweek_column].min())
        if validation_min <= training_max_gw:
            raise ValueError("validation gameweeks must be strictly after the training cutoff")

    # MASE's denominator is an in-sample quantity.  Derive it here, from the
    # same pre-validation training frame that fits the thresholds, rather than
    # accepting a caller-provided value that could include validation rows.
    mase_scale = naive_lag1_scale(
        training_frame, value_col=target_column, gw_col=gameweek_column
    )
    if not np.isfinite(mase_scale) or mase_scale <= 0:
        raise ValueError("training-only MASE scale must be finite and positive")
    training_cutoff = int(pd.to_numeric(training_frame[gameweek_column], errors="raise").max())
    if training_max_gw is None:
        training_max_gw = training_cutoff
    thresholds = fit_tertile_thresholds(training_frame)
    regimes = thresholds.route(validation_frame[ROUTE_FEATURE].to_numpy())
    truth = _finite_1d(validation_frame[target_column], target_column)
    weeks = validation_frame[gameweek_column].to_numpy()
    candidate_order = (champion,) + tuple(
        expert for expert in candidate_prediction_columns if expert != champion
    )
    predictions = {
        expert: _finite_1d(validation_frame[column], column)
        for expert, column in candidate_prediction_columns.items()
    }

    selections: dict[str, RegimeSelection] = {}
    for regime in REGIMES:
        mask = regimes == regime
        rows = int(mask.sum())
        gameweek_count = int(pd.unique(weeks[mask]).size)
        if rows < min_rows or gameweek_count < min_gameweeks:
            selections[regime] = RegimeSelection(
                expert=champion,
                rows=rows,
                gameweeks=gameweek_count,
                champion_mase=None,
                selected_mase=None,
                improvement=0.0,
                fallback_reason="insufficient validation support",
            )
            continue

        scores = {
            expert: mase(truth[mask], predictions[expert][mask], mase_scale)
            for expert in candidate_order
        }
        champion_score = scores[champion]
        best = min(candidate_order, key=lambda expert: scores[expert])
        best_score = scores[best]
        improvement = champion_score - best_score
        if best == champion or improvement < min_mase_improvement:
            reason = (
                "champion has best regime MASE"
                if best == champion
                else "MASE improvement below threshold"
            )
            selected = champion
            selected_score = champion_score
        else:
            reason = None
            selected = best
            selected_score = best_score
        selections[regime] = RegimeSelection(
            expert=selected,
            rows=rows,
            gameweeks=gameweek_count,
            champion_mase=champion_score,
            selected_mase=selected_score,
            improvement=max(0.0, improvement),
            fallback_reason=reason,
        )

    validation_weeks = pd.to_numeric(validation_frame[gameweek_column], errors="raise")
    return MidGateConfig(
        champion=champion,
        candidates=candidate_order,
        thresholds=thresholds,
        selections=selections,
        min_rows=min_rows,
        min_gameweeks=min_gameweeks,
        min_mase_improvement=min_mase_improvement,
        mase_scale=float(mase_scale),
        mase_scale_training_max_gw=training_cutoff,
        training_max_gw=training_max_gw,
        validation_min_gw=int(validation_weeks.min()),
        validation_max_gw=int(validation_weeks.max()),
        provenance=dict(provenance or {}),
    )
