"""Immutable research-dataset admission control.

The live dataset is intentionally mutable: ``fpl.data.fetch`` appends newly
published rounds and ``fpl.run_week`` consumes that state.  Registered model
research must instead use the one SHA-bound snapshot frozen through 2026-27
GW3/global GW231.  This module is deliberately separate from the live fetch
path so a normal weekly recommendation can continue to refresh its inputs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from io import BytesIO

import numpy as np

import pandas as pd

from fpl import config


FROZEN_MANIFEST_PATH = Path(__file__).with_name("frozen_research_dataset_gw231.json")
_REQUIRED_MANIFEST_FIELDS = frozenset({
    "schema_version", "dataset_sha256", "data_rows", "max_gw_global",
    "current_season", "current_season_max_gw",
})


class FrozenDatasetProvenanceError(ValueError):
    """Raised when an on-disk research input is not the registered snapshot."""


def dataframe_fingerprint(frame: pd.DataFrame) -> str:
    """Hash a frame's index, values, ordered column labels, and dtypes.

    Attributes are intentionally excluded so an exact defensive copy of an admitted
    frame retains its content identity while mutation of its data or schema does not.
    """
    schema = {
        "columns": [repr(column) for column in frame.columns],
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "index_names": [repr(name) for name in frame.index.names],
        "index_dtypes": [str(frame.index.get_level_values(level).dtype)
                         for level in range(frame.index.nlevels)],
    }
    digest = hashlib.sha256(json.dumps(schema, sort_keys=True).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, object]:
    try:
        manifest = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise FrozenDatasetProvenanceError(f"frozen dataset manifest is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise FrozenDatasetProvenanceError(f"frozen dataset manifest is invalid JSON: {path}") from exc
    if not isinstance(manifest, dict) or set(manifest) != _REQUIRED_MANIFEST_FIELDS:
        raise FrozenDatasetProvenanceError("frozen dataset manifest has unexpected fields")
    if manifest["schema_version"] != 1:
        raise FrozenDatasetProvenanceError("unsupported frozen dataset manifest schema")
    if not isinstance(manifest["dataset_sha256"], str) or len(manifest["dataset_sha256"]) != 64:
        raise FrozenDatasetProvenanceError("frozen dataset manifest has invalid SHA-256")
    return manifest


def load_frozen_dataset_manifest(manifest_path: str | Path = FROZEN_MANIFEST_PATH) -> dict[str, object]:
    """Return a defensive copy of the validated frozen-dataset manifest."""
    return dict(_load_manifest(Path(manifest_path)))


def load_frozen_research_dataset(dataset_path: str | Path | None = None,
                                 manifest_path: str | Path = FROZEN_MANIFEST_PATH) -> pd.DataFrame:
    """Hash and parse one immutable byte buffer for the registered snapshot.

    Admission checks precede all feature construction. The frame carries both the
    registered byte digest and a content fingerprint for later mutation detection.
    """
    dataset = Path(config.FROZEN_RESEARCH_DATASET_PATH if dataset_path is None else dataset_path)
    manifest = _load_manifest(Path(manifest_path))
    try:
        content = dataset.read_bytes()
    except FileNotFoundError as exc:
        raise FrozenDatasetProvenanceError(f"frozen research dataset is missing: {dataset}") from exc
    try:
        frame = pd.read_csv(BytesIO(content), low_memory=False)
    except (ValueError, pd.errors.ParserError) as exc:
        raise FrozenDatasetProvenanceError("frozen research dataset cannot be parsed") from exc
    if frame.empty:
        raise FrozenDatasetProvenanceError("frozen research dataset is empty")
    required = {"GW_global", "season", "GW"}
    if missing := required - set(frame):
        raise FrozenDatasetProvenanceError(f"dataset lacks required columns: {sorted(missing)}")
    for column in ("GW_global", "GW"):
        values = frame[column]
        if (not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values)
                or not np.isfinite(values).all() or not (values % 1 == 0).all()):
            raise FrozenDatasetProvenanceError(f"dataset {column} must contain finite integer gameweeks")
    max_gw = int(frame["GW_global"].max())
    if max_gw != manifest["max_gw_global"]:
        raise FrozenDatasetProvenanceError(
            f"dataset global-GW cutoff {max_gw} does not equal frozen cutoff "
            f"{manifest['max_gw_global']}"
        )
    digest = hashlib.sha256(content).hexdigest()
    if digest != manifest["dataset_sha256"]:
        raise FrozenDatasetProvenanceError("dataset SHA-256 does not match frozen manifest")
    if len(frame) != manifest["data_rows"]:
        raise FrozenDatasetProvenanceError("dataset row count does not match frozen manifest")
    if frame["season"].isna().any() or frame["season"].max() != manifest["current_season"]:
        raise FrozenDatasetProvenanceError("dataset maximum season does not match frozen manifest")
    current = frame[frame["season"] == manifest["current_season"]]
    if current.empty or int(current["GW"].max()) != manifest["current_season_max_gw"]:
        raise FrozenDatasetProvenanceError("dataset current-season gameweek does not match frozen manifest")
    frame.attrs["frozen_dataset_sha256"] = digest
    frame.attrs["frozen_dataframe_fingerprint"] = dataframe_fingerprint(frame)
    print("Loaded frozen research dataset "
          f"(rows={len(frame)}, max_gw={max_gw}, sha256={digest})")
    return frame
