"""Content-addressed cache for the expensive historical feature frame."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path

import pandas as pd

from fpl import config, features
from fpl.data.provenance import (
    FrozenDatasetProvenanceError,
    dataframe_fingerprint,
    load_frozen_dataset_manifest,
    load_frozen_research_dataset,
)


CACHE_SCHEMA_VERSION = 1
DEFAULT_CACHE_DIR = config.ROOT / ".cache" / "feature_frames"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_identity(dataset_sha256: str, feature_source_path: Path) -> dict[str, object]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "dataset_sha256": dataset_sha256,
        "feature_source_sha256": _sha256(feature_source_path),
        "pandas_version": pd.__version__,
    }


def load_or_build_feature_frame(raw: pd.DataFrame, dataset_sha256: str,
                                cache_dir: str | Path = DEFAULT_CACHE_DIR,
                                feature_source_path: str | Path | None = None) -> pd.DataFrame:
    """Load an exact cache hit or atomically build and persist the feature frame.

    Pickle is deliberate here: the cache is local, ignored, and written only by this process,
    while Parquet would add a binary runtime dependency solely for a regenerable artifact.
    """
    source = Path(feature_source_path or features.__file__)
    identity = _cache_identity(dataset_sha256, source)
    identity["raw_dataframe_fingerprint"] = dataframe_fingerprint(raw)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    directory = Path(cache_dir)
    frame_path = directory / f"features-{key}.pkl"
    metadata_path = directory / f"features-{key}.json"

    if frame_path.exists() and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
            if (metadata.get("identity") == identity
                    and metadata.get("frame_sha256") == _sha256(frame_path)):
                frame = pd.read_pickle(frame_path)
                if len(frame) == metadata.get("rows") and list(frame.columns) == metadata.get("columns"):
                    print(f"Loaded cached feature frame ({len(frame)} rows, key={key})")
                    return frame
        except (OSError, ValueError, json.JSONDecodeError, EOFError, pickle.UnpicklingError):
            pass

    frame = features.build_feature_frame(raw)
    directory.mkdir(parents=True, exist_ok=True)
    tmp_frame = frame_path.with_name(f"{frame_path.name}.tmp-{os.getpid()}")
    tmp_metadata = metadata_path.with_name(f"{metadata_path.name}.tmp-{os.getpid()}")
    frame.to_pickle(tmp_frame)
    metadata = {
        "identity": identity,
        "rows": len(frame),
        "columns": list(frame.columns),
        "frame_sha256": _sha256(tmp_frame),
    }
    tmp_metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    os.replace(tmp_frame, frame_path)
    os.replace(tmp_metadata, metadata_path)
    print(f"Built cached feature frame ({len(frame)} rows, key={key})")
    return frame


def load_frozen_feature_frame(raw: pd.DataFrame | None = None,
                              cache_dir: str | Path = DEFAULT_CACHE_DIR) -> pd.DataFrame:
    """Return features for the registered frozen dataset, using its content identity."""
    manifest = load_frozen_dataset_manifest()
    admitted = load_frozen_research_dataset() if raw is None else raw
    if (len(admitted) != manifest["data_rows"]
            or "GW_global" not in admitted
            or admitted.attrs.get("frozen_dataset_sha256") != manifest["dataset_sha256"]
            or admitted.attrs.get("frozen_dataframe_fingerprint") != dataframe_fingerprint(admitted)
            or int(admitted["GW_global"].max()) != manifest["max_gw_global"]):
        raise FrozenDatasetProvenanceError(
            "feature-cache input is not the complete registered frozen dataset"
        )
    frame = load_or_build_feature_frame(
        admitted, manifest["dataset_sha256"], cache_dir=cache_dir
    )
    # Feature merges may discard raw attrs; retain the verified byte identity explicitly.
    frame.attrs["frozen_dataset_sha256"] = manifest["dataset_sha256"]
    return frame
