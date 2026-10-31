"""Research-only CPU CNN estimator with training-local sequence preprocessing.

Torch is imported lazily by fit, leaving production imports independent of the
optional neural runtime. Causal sequence construction belongs to sequence_features;
this adapter rejects populated padding and never observes evaluation outcomes.
"""
from __future__ import annotations

from numbers import Integral, Real
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from fpl import config
from fpl.model.sequence_features import sequence_columns


class SequenceCNNRegressor(RegressorMixin, BaseEstimator):
    """Two fixed convolution layers, masked pooling and numeric fixture context.

    Sequence channel statistics pool only finite observed training entries across
    lag positions. Context imputation/scaling and target scaling are training-only.
    Predictions retain original point units and are never clipped.
    """

    def __init__(
        self, seed: int = 0, epochs: int = config.CNN_EPOCHS,
        batch_size: int = config.CNN_BATCH_SIZE, channels: int = config.CNN_CHANNELS,
        hidden: int = config.CNN_HIDDEN, learning_rate: float = config.CNN_LEARNING_RATE,
        weight_decay: float = config.CNN_WEIGHT_DECAY, grad_clip: float = config.CNN_GRAD_CLIP,
    ) -> None:
        self.seed = seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.channels = channels
        self.hidden = hidden
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip

    def _validate_parameters(self) -> None:
        for name in ("epochs", "batch_size", "channels", "hidden"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name in ("learning_rate", "grad_clip", "weight_decay"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value < 0 or (value == 0 and name != "weight_decay"):
                raise ValueError(f"{name} must be positive (weight_decay may be zero)")

    def _inputs(self, X: pd.DataFrame, *, fitting: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not isinstance(X, pd.DataFrame) or X.empty or X.columns.duplicated().any():
            raise ValueError("SequenceCNN requires a nonempty DataFrame with unique ordered columns")
        if not fitting and list(X.columns) != list(self.feature_names_in_):
            raise ValueError("SequenceCNN prediction feature columns/order differ from training")
        required = list(sequence_columns())
        if missing := set(required).difference(X.columns):
            raise ValueError(f"SequenceCNN missing sequence columns: {sorted(missing)}")
        sequence_names = [f"cnn_{stat}_t{step}" for stat in config.CNN_SEQUENCE_STATS
                          for step in range(config.CNN_SEQUENCE_LENGTH)]
        presence_names = [f"cnn_present_t{step}" for step in range(config.CNN_SEQUENCE_LENGTH)]
        context_names = [column for column in X.columns if column not in required]
        try:
            sequence = X[sequence_names].to_numpy(dtype=float).reshape(
                len(X), len(config.CNN_SEQUENCE_STATS), config.CNN_SEQUENCE_LENGTH)
            presence = X[presence_names].to_numpy(dtype=float)
            context = X[context_names].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("SequenceCNN features must be numeric") from exc
        if np.isinf(sequence).any() or np.isinf(context).any():
            raise ValueError("SequenceCNN features cannot contain infinities")
        if not np.isfinite(presence).all() or not np.isin(presence, [0., 1.]).all():
            raise ValueError("SequenceCNN presence masks must be finite binary values")
        if np.any(np.isfinite(sequence) & (presence[:, None, :] == 0)):
            raise ValueError("SequenceCNN padded steps must contain NaN statistics")
        if fitting:
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
            self.n_features_in_ = len(X.columns)
            self.context_columns_ = context_names
        return sequence, presence, context

    def _learn_preprocessing(self, sequence: np.ndarray, context: np.ndarray) -> None:
        medians, means, scales = [], [], []
        # Eight statistic channels, not player-row ETL: each reduces all observed lags.
        for channel in range(sequence.shape[1]):
            observed = sequence[:, channel, :]
            finite = observed[np.isfinite(observed)]
            medians.append(float(np.median(finite)) if finite.size else 0.)
            means.append(float(np.mean(finite)) if finite.size else 0.)
            std = float(np.std(finite)) if finite.size else 0.
            scales.append(std if std > 0 else 1.)
        self.sequence_median_ = np.asarray(medians)
        self.sequence_mean_ = np.asarray(means)
        self.sequence_scale_ = np.asarray(scales)
        context_medians = []
        for column in range(context.shape[1]):
            finite = context[:, column][np.isfinite(context[:, column])]
            context_medians.append(float(np.median(finite)) if finite.size else 0.)
        self.context_median_ = np.asarray(context_medians)
        filled = np.where(np.isnan(context), self.context_median_[None, :], context)
        self.context_mean_ = np.mean(filled, axis=0)
        standard_deviation = np.std(filled, axis=0)
        self.context_scale_ = np.where(standard_deviation > 0, standard_deviation, 1.)

    def _transform(self, sequence: np.ndarray, presence: np.ndarray, context: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sequence_missing = np.isnan(sequence)
        filled = np.where(sequence_missing, self.sequence_median_[None, :, None], sequence)
        normalized = (filled - self.sequence_mean_[None, :, None]) / self.sequence_scale_[None, :, None]
        sequence_input = np.concatenate([normalized, sequence_missing.astype(float), presence[:, None, :]], axis=1)
        context_missing = np.isnan(context)
        filled_context = np.where(context_missing, self.context_median_[None, :], context)
        normalized_context = (filled_context - self.context_mean_[None, :]) / self.context_scale_[None, :]
        context_input = np.concatenate([normalized_context, context_missing.astype(float)], axis=1)
        outputs = tuple(array.astype(np.float32) for array in (sequence_input, presence, context_input))
        if any(not np.isfinite(array).all() for array in outputs):
            raise ValueError("SequenceCNN preprocessing produced nonfinite inputs")
        return outputs

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> SequenceCNNRegressor:
        """Fit fixed CPU epochs using shuffled seeded training minibatches only."""
        self._validate_parameters()
        started = time.perf_counter()
        sequence, presence, context = self._inputs(X, fitting=True)
        target = np.asarray(y, dtype=float)
        if target.ndim != 1 or target.shape != (len(X),) or not np.isfinite(target).all():
            raise ValueError("SequenceCNN y must be a finite one-dimensional target matching X")
        self._learn_preprocessing(sequence, context)
        sequence_input, presence_input, context_input = self._transform(sequence, presence, context)
        self.target_mean_ = float(np.mean(target))
        target_std = float(np.std(target))
        self.target_scale_ = target_std if target_std > 0 else 1.
        normalized_target = ((target - self.target_mean_) / self.target_scale_).astype(np.float32)
        if not np.isfinite(normalized_target).all():
            raise ValueError("SequenceCNN target preprocessing produced nonfinite values")
        import torch
        torch.set_num_threads(config.RESEARCH_MODEL_THREADS)
        torch.manual_seed(int(self.seed))
        self._torch_ = torch
        channel_count, context_count = sequence_input.shape[1], context_input.shape[1]
        class Network(torch.nn.Module):
            def __init__(network_self) -> None:
                super().__init__()
                network_self.convolution = torch.nn.Sequential(
                    torch.nn.Conv1d(channel_count, self.channels, kernel_size=3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv1d(self.channels, self.channels, kernel_size=3, padding=1),
                    torch.nn.ReLU())
                network_self.head = torch.nn.Sequential(
                    torch.nn.Linear(self.channels + context_count, self.hidden), torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden, 1))
            def forward(network_self, seq, present, ctx):
                convolved = network_self.convolution(seq)
                pooled = (convolved * present[:, None, :]).sum(dim=2) / present.sum(dim=1).clamp(min=1)[:, None]
                return network_self.head(torch.cat([pooled, ctx], dim=1)).squeeze(1)
        self.network_ = Network().cpu()
        optimizer = torch.optim.Adam(self.network_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        seq_tensor, mask_tensor, ctx_tensor = (torch.from_numpy(array) for array in (sequence_input, presence_input, context_input))
        target_tensor = torch.from_numpy(normalized_target)
        generator = torch.Generator(device="cpu").manual_seed(int(self.seed))
        self.loss_history_ = []
        self.network_.train()
        for _ in range(self.epochs):
            order = torch.randperm(len(X), generator=generator)
            total_loss = 0.
            for offset in range(0, len(X), self.batch_size):
                indices = order[offset:offset + self.batch_size]
                optimizer.zero_grad()
                prediction = self.network_(seq_tensor[indices], mask_tensor[indices], ctx_tensor[indices])
                loss = torch.nn.functional.mse_loss(prediction, target_tensor[indices])
                if not torch.isfinite(loss):
                    raise ValueError("SequenceCNN CPU training produced a nonfinite loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network_.parameters(), self.grad_clip)
                optimizer.step()
                total_loss += float(loss.detach()) * len(indices)
            self.loss_history_.append(total_loss / len(X))
        self.fit_seconds_ = time.perf_counter() - started
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Transform by immutable training statistics and return unclipped point predictions."""
        check_is_fitted(self, ["network_", "sequence_median_", "target_scale_"])
        sequence, presence, context = self._inputs(X, fitting=False)
        inputs = self._transform(sequence, presence, context)
        torch = self._torch_
        tensors = tuple(torch.from_numpy(array) for array in inputs)
        predictions = []
        self.network_.eval()
        with torch.no_grad():
            for offset in range(0, len(X), self.batch_size):
                batch = tuple(tensor[offset:offset + self.batch_size] for tensor in tensors)
                predictions.append(self.network_(*batch).cpu().numpy())
        output = np.concatenate(predictions).astype(float) * self.target_scale_ + self.target_mean_
        if not np.isfinite(output).all():
            raise ValueError("SequenceCNN CPU prediction produced nonfinite values")
        return output
