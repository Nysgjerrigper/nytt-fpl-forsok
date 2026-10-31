"""Position/group signed CQR with explicit sparse-group position fallback.

Groups must be caller-provided deadline-known labels, fitted exclusively on earlier
calibration outcomes. Temporal FPL dependence does not establish exchangeability
or coverage guarantees. All numerical adjustments reuse interval_calibration.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fpl.model.interval_calibration import (
    _arrays, _validate_alpha, _validate_bounds, apply_adjustment, conformal_adjustment,
)


def _label(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")


def _count(value: object, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive Python integer excluding bool")


def _labels(value: ArrayLike, n: int, name: str) -> NDArray[np.str_]:
    array = np.asarray(value, dtype=object)
    if array.ndim != 1 or array.size != n or n == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional label array aligned with intervals")
    # Type checks are per distinct label, not an ETL row loop. Object-array
    # comparisons avoid coercing missing/numeric labels into plausible strings.
    try:
        distinct = set(array.tolist())
    except TypeError as exc:
        raise ValueError(f"{name} must contain only nonempty text labels") from exc
    for label in distinct:
        _label(label, name)
    return array.astype(str)


def _alpha(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("alpha must be a numeric miscoverage probability")
    _validate_alpha(value)


@dataclass(frozen=True)
class CalibrationCell:
    """An immutable fitted position fallback or sufficiently populated group cell."""
    position: str
    group: str | None
    n: int
    adjustment: float

    def __post_init__(self) -> None:
        _label(self.position, "position")
        if self.group is not None:
            _label(self.group, "group")
        _count(self.n, "cell count")
        if isinstance(self.adjustment, bool) or not isinstance(self.adjustment, Real) or not np.isfinite(self.adjustment):
            raise ValueError("cell adjustment must be finite numeric")


@dataclass(frozen=True)
class GroupedCalibration:
    """Fitted cells plus all observed group counts, including sparse groups.

    Counts partition each position fallback. Every group reaching the declared
    minimum has exactly one fitted cell; other groups use that position fallback.
    """
    alpha: float
    min_group_size: int
    cells: tuple[CalibrationCell, ...]
    group_counts: tuple[tuple[str, str, int], ...]

    def __post_init__(self) -> None:
        _alpha(self.alpha)
        _count(self.min_group_size, "min_group_size")
        if not isinstance(self.cells, tuple) or not self.cells or any(not isinstance(cell, CalibrationCell) for cell in self.cells):
            raise ValueError("cells must be a nonempty immutable tuple of CalibrationCell")
        keys = [(cell.position, cell.group) for cell in self.cells]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate calibration position/group cell")
        fallback = {cell.position: cell.n for cell in self.cells if cell.group is None}
        fitted = {(cell.position, cell.group): cell.n for cell in self.cells if cell.group is not None}
        if not fallback or any(position not in fallback for position, _ in fitted):
            raise ValueError("every group cell requires its position fallback")
        if not isinstance(self.group_counts, tuple) or not self.group_counts:
            raise ValueError("group_counts must be a nonempty immutable tuple")
        counts = {}
        for record in self.group_counts:
            if not isinstance(record, tuple) or len(record) != 3:
                raise ValueError("group count records must be immutable position/group/count triples")
            position, group, n = record
            _label(position, "group count position")
            _label(group, "group count group")
            _count(n, "group count")
            if (position, group) in counts:
                raise ValueError("duplicate group count metadata")
            if position not in fallback:
                raise ValueError("group count has no position fallback")
            counts[position, group] = n
        if self.group_counts != tuple(sorted(self.group_counts)):
            raise ValueError("group_counts must be sorted by position/group")
        totals = {position: sum(n for (p, _), n in counts.items() if p == position) for position in fallback}
        if totals != fallback:
            raise ValueError("group counts must partition each position fallback count")
        eligible = {key: n for key, n in counts.items() if n >= self.min_group_size}
        if fitted != eligible:
            raise ValueError("fitted group counts must match all minimum-eligible metadata groups")


def fit_grouped_calibration(y: ArrayLike, lower: ArrayLike, upper: ArrayLike,
                            positions: ArrayLike, groups: ArrayLike, *, alpha: float,
                            min_group_size: int) -> GroupedCalibration:
    """Fit exact signed adjustments on calibration data, never evaluation outcomes.

    Every position gets a fallback. An eligible group's insufficient exact-order
    sample raises rather than silently changing the rank or the minimum threshold.
    """
    y, lower, upper = _arrays(y, lower, upper)
    _alpha(alpha)
    _count(min_group_size, "min_group_size")
    _validate_bounds(lower, upper)
    position_labels = _labels(positions, len(y), "positions")
    group_labels = _labels(groups, len(y), "groups")
    cells, counts = [], []
    for position in np.unique(position_labels):
        position_mask = position_labels == position
        n = int(position_mask.sum())
        cells.append(CalibrationCell(str(position), None, n,
                     conformal_adjustment(y[position_mask], lower[position_mask], upper[position_mask], alpha=alpha)))
        for group in np.unique(group_labels[position_mask]):
            mask = position_mask & (group_labels == group)
            n = int(mask.sum())
            counts.append((str(position), str(group), n))
            if n >= min_group_size:
                cells.append(CalibrationCell(str(position), str(group), n,
                             conformal_adjustment(y[mask], lower[mask], upper[mask], alpha=alpha)))
    cells.sort(key=lambda cell: (cell.position, cell.group is not None, cell.group or ""))
    return GroupedCalibration(alpha, min_group_size, tuple(cells), tuple(sorted(counts)))


def apply_grouped_calibration(calibration: GroupedCalibration, lower: ArrayLike,
                              median: ArrayLike, upper: ArrayLike, positions: ArrayLike,
                              groups: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.str_]]:
    """Apply fitted cells without outcomes; unseen/sparse groups use position cells.

    Unknown positions are rejected. Existing apply_adjustment supplies conservative
    median bracketing; iteration is over distinct position/group pairs, not rows.
    """
    if not isinstance(calibration, GroupedCalibration):
        raise ValueError("calibration must be GroupedCalibration")
    lower, median, upper = _arrays(lower, median, upper)
    _validate_bounds(lower, upper)
    position_labels = _labels(positions, len(lower), "positions")
    group_labels = _labels(groups, len(lower), "groups")
    cells = {(cell.position, cell.group): cell.adjustment for cell in calibration.cells}
    if any((position, None) not in cells for position in np.unique(position_labels)):
        raise ValueError("unknown calibration position")
    result_lower, result_upper, adjustments = np.empty_like(lower), np.empty_like(upper), np.empty_like(lower)
    sources = np.empty(len(lower), dtype="U8")
    for position, group in np.unique(np.column_stack([position_labels, group_labels]), axis=0):
        mask = (position_labels == position) & (group_labels == group)
        use_group = (position, group) in cells
        adjustment = cells[position, group] if use_group else cells[position, None]
        result_lower[mask], result_upper[mask] = apply_adjustment(lower[mask], median[mask], upper[mask], adjustment)
        adjustments[mask] = adjustment
        sources[mask] = "group" if use_group else "position"
    return result_lower, result_upper, adjustments, sources
