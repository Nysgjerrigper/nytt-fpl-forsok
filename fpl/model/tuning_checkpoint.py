"""Opt-in, identity-bound serial Optuna resumes; no sampler pickle is loaded."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import os
import sys
import time
from typing import Any, Callable, Iterator

import pandas as pd

from fpl import config, features
from fpl.data.provenance import dataframe_fingerprint
from fpl.model import metrics, models

PROTOCOL = "serial_trial_seed_v1"


def checkpoint_identity(frame: pd.DataFrame, feature_cols: list[str], *, position: str,
                        model_name: str, seed: int, stage: str | None,
                        train_max_gw: int | None, n_splits: int,
                        folds: list[tuple[set, set]]) -> dict[str, Any]:
    """Bind actual capped rows, ordered inputs, fold boundaries, code and runtime.

    Trial count and invocation timeout are operational limits: increasing the total
    trial target extends this same search, without changing its statistical identity.
    """
    from fpl.model import tuning

    packages = {}
    for name in ("numpy", "pandas", "scikit-learn", "scipy", "lightgbm", "xgboost",
                 "catboost", "optuna", "pytabkit", "torch", "skorch", "faiss-cpu"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    sources = {module.__name__: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
               for module in (config, features, metrics, models, tuning)}
    sources[__name__] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    spec = models.EXPERT_SPECS[model_name]
    return {
        "protocol": PROTOCOL, "sampler": "TPESampler defaults; per-trial derived seed",
        "pruner": "NopPruner", "position": position, "model": model_name,
        "seed": seed, "stage": stage, "train_max_gw": train_max_gw,
        "n_splits": n_splits, "features": list(feature_cols),
        "frame_sha256": dataframe_fingerprint(frame),
        "dataset_sha256": frame.attrs.get("frozen_dataset_sha256"),
        "folds": [{"train": sorted(train), "validation": sorted(val)} for train, val in folds],
        "sources": sources, "search_space": {"source": sources[models.__name__],
            "preprocessing": spec.preprocessing, "provenance": spec.provenance},
        "runtime": {"python": sys.version, "platform": platform.platform(),
                    "machine": platform.machine(), "packages": packages,
                    "logical_cpu_count": os.cpu_count(),
                    "thread_device_environment": {name: os.environ.get(name) for name in
                        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                         "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                         "CUDA_VISIBLE_DEVICES", "PYTHONHASHSEED")}},
    }


@contextmanager
def _exclusive_checkpoint(path: Path) -> Iterator[None]:
    """Hold a nonblocking OS lock for the complete read/validate/optimize operation."""
    try:
        import fcntl
    except ImportError as exc:
        raise RuntimeError("Checkpoint protocol requires POSIX file locking") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(path) + ".lock").open("a+b") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Checkpoint is already owned by another process") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run_checkpoint(objective: Callable, *, storage_path: str | Path, study_name: str,
                   identity: dict[str, Any], seed: int, n_trials: int,
                   timeout: float | None) -> Any:
    """Resume a matching serial study using deterministic per-trial TPE seeds.

    This versioned sampler differs from legacy continuous-RNG TPE. Each trial seed
    depends on the registered seed and trial number; TPE still uses completed history.
    A clean resume reproduces the uninterrupted sequence. RUNNING/WAITING trials
    fail closed after crashes; no stale trial or ownership is silently repaired.
    Timeout is a fresh wall-clock allowance per invocation, not lifetime accounting.
    """
    import optuna

    if identity.get("protocol") != PROTOCOL or identity.get("seed") != seed:
        raise ValueError("Checkpoint protocol/seed does not match the sampler request")
    path = Path(storage_path).resolve()
    storage = f"sqlite:///{path}"
    with _exclusive_checkpoint(path):
        try:
            study = optuna.create_study(storage=storage, study_name=study_name,
                direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed),
                pruner=optuna.pruners.NopPruner())
        except optuna.exceptions.DuplicatedStudyError:
            study = optuna.load_study(storage=storage, study_name=study_name)
            if study.user_attrs.get("resume_identity") != identity:
                raise ValueError("Checkpoint identity mismatch; preserve it and use a new study")
        else:
            study.set_user_attr("resume_identity", identity)
        if study.direction != optuna.study.StudyDirection.MINIMIZE:
            raise ValueError("Checkpoint direction must be minimize")
        if [trial.number for trial in study.trials] != list(range(len(study.trials))):
            raise ValueError("Checkpoint trial numbering is not contiguous")
        study.pruner = optuna.pruners.NopPruner()
        if any(trial.state in (optuna.trial.TrialState.RUNNING, optuna.trial.TrialState.WAITING)
               for trial in study.trials):
            raise ValueError("Checkpoint contains unresolved RUNNING/WAITING trials; resume refused")
        if len(study.trials) > n_trials:
            raise ValueError("Checkpoint already exceeds the requested total trial target")
        deadline = None if timeout is None else time.monotonic() + timeout
        while len(study.trials) < n_trials:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                break
            number = len(study.trials)
            trial_seed = int.from_bytes(hashlib.sha256(
                f"{PROTOCOL}:{seed}:{number}".encode()).digest()[:4], "big")
            study.sampler = optuna.samplers.TPESampler(seed=trial_seed)
            study.optimize(objective, n_trials=1, timeout=remaining, show_progress_bar=False)
        if any(not trial.state.is_finished() for trial in study.trials):
            raise ValueError("Checkpoint has unresolved trials after optimization")
        if len(study.trials) < n_trials:
            raise ValueError("Checkpoint trial target incomplete after timeout; no tuned artifact can be saved")
        if len(study.trials) != n_trials:
            raise ValueError("Checkpoint total changed outside exclusive ownership")
        if not any(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials):
            raise ValueError("Checkpoint has no completed trial; no tuned artifact can be saved")
        return study
