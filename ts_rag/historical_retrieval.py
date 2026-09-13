from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class RetrievalFeatureConfig:
    tail_len: int = 48
    pooled_steps: int = 12
    include_differences: bool = True
    include_forecast: bool = True
    include_calendar: bool = True
    max_channels: int = 16
    include_level_features: bool = True

    def to_dict(self):
        return asdict(self)


def _pool_steps(values, steps):
    if values.shape[1] < steps:
        raise ValueError(f"Cannot pool {values.shape[1]} time steps into {steps} bins")
    chunks = np.array_split(np.arange(values.shape[1]), steps)
    return np.stack([values[:, chunk, :].mean(axis=1) for chunk in chunks], axis=1)


def _normalize_window(values, eps=1e-6):
    mean = values.mean(axis=1, keepdims=True)
    scale = values.std(axis=1, keepdims=True)
    return (values - mean) / (scale + eps)


def _pool_channels(values, max_channels):
    if max_channels <= 0 or values.shape[2] <= max_channels:
        return values
    groups = np.array_split(np.arange(values.shape[2]), max_channels)
    return np.stack([values[:, :, group].mean(axis=2) for group in groups], axis=2)


def _series_features(values, pooled_steps, include_differences, include_level_features=True):
    normalized = _normalize_window(values)
    features = [_pool_steps(normalized, pooled_steps).reshape(values.shape[0], -1)]

    if include_level_features:
        time = np.linspace(-1.0, 1.0, values.shape[1], dtype=np.float32)
        centered_time = time - time.mean()
        slope = np.einsum("ntd,t->nd", values, centered_time)
        slope /= np.square(centered_time).sum()
        features.extend(
            [
                values[:, -1, :],
                values.mean(axis=1),
                values.std(axis=1),
                slope,
            ]
        )

    if include_differences:
        differences = np.diff(values, axis=1)
        scale = values.std(axis=1, keepdims=True) + 1e-6
        scaled_differences = differences / scale
        features.append(_pool_steps(scaled_differences, pooled_steps).reshape(values.shape[0], -1))
    return features


def extract_retrieval_features(bundle, config):
    tail_len = min(config.tail_len, bundle["x"].shape[1])
    tail = np.asarray(bundle["x"][:, -tail_len:, :], dtype=np.float32)
    tail = _pool_channels(tail, config.max_channels)
    features = _series_features(
        tail,
        config.pooled_steps,
        config.include_differences,
        include_level_features=config.include_level_features,
    )

    if config.include_forecast:
        forecast = np.asarray(bundle["y_base"], dtype=np.float32)
        forecast = _pool_channels(forecast, config.max_channels)
        forecast_steps = min(config.pooled_steps, forecast.shape[1])
        features.extend(
            _series_features(
                forecast,
                forecast_steps,
                include_differences=False,
                include_level_features=config.include_level_features,
            )
        )

    if config.include_calendar:
        if "x_mark" not in bundle or "y_mark" not in bundle:
            raise KeyError("Calendar retrieval requires x_mark and y_mark in the prediction bundle")
        x_mark = np.asarray(bundle["x_mark"], dtype=np.float32)
        y_mark = np.asarray(bundle["y_mark"], dtype=np.float32)
        future_positions = np.unique(np.linspace(0, y_mark.shape[1] - 1, 4, dtype=int))
        features.append(x_mark[:, -1, :])
        features.append(y_mark[:, future_positions, :].reshape(y_mark.shape[0], -1))

    return np.concatenate(features, axis=1).astype(np.float32)


class HistoricalResidualIndex:
    def __init__(self, feature_config, chunk_size=256):
        self.feature_config = feature_config
        self.chunk_size = chunk_size

    def fit(self, bundle, indices=None):
        features = extract_retrieval_features(bundle, self.feature_config)
        if indices is None:
            indices = np.arange(features.shape[0])
        indices = np.asarray(indices)
        memory = features[indices]
        self.feature_mean = memory.mean(axis=0, keepdims=True)
        self.feature_std = memory.std(axis=0, keepdims=True) + 1e-6
        memory = (memory - self.feature_mean) / self.feature_std
        self.memory_features = self._l2_normalize(memory)
        self.memory_residuals = np.asarray(
            bundle["y"][indices] - bundle["y_base"][indices], dtype=np.float32
        )
        self.memory_offset_forecasts = np.asarray(
            bundle["y_base"][indices] - bundle["x"][indices][:, -1:, :],
            dtype=np.float32,
        )
        self.memory_indices = indices
        return self

    @staticmethod
    def _l2_normalize(values):
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        return values / (norms + 1e-8)

    def query_neighbors(self, bundle, indices=None, max_k=32):
        if not hasattr(self, "memory_features"):
            raise RuntimeError("HistoricalResidualIndex.fit must be called before querying")
        features = extract_retrieval_features(bundle, self.feature_config)
        if indices is None:
            indices = np.arange(features.shape[0])
        indices = np.asarray(indices)
        query = (features[indices] - self.feature_mean) / self.feature_std
        query = self._l2_normalize(query)

        k = min(max_k, self.memory_features.shape[0])
        neighbor_indices = np.empty((len(indices), k), dtype=np.int64)
        neighbor_scores = np.empty((len(indices), k), dtype=np.float32)
        for start in range(0, len(indices), self.chunk_size):
            end = min(start + self.chunk_size, len(indices))
            similarities = query[start:end] @ self.memory_features.T
            selected = np.argpartition(similarities, -k, axis=1)[:, -k:]
            selected_scores = np.take_along_axis(similarities, selected, axis=1)
            order = np.argsort(selected_scores, axis=1)[:, ::-1]
            neighbor_indices[start:end] = np.take_along_axis(selected, order, axis=1)
            neighbor_scores[start:end] = np.take_along_axis(selected_scores, order, axis=1)
        return neighbor_indices, neighbor_scores

    @staticmethod
    def neighbor_weights(neighbor_scores, k, temperature):
        scores = neighbor_scores[:, :k].astype(np.float64)
        if temperature <= 0:
            return np.full_like(scores, 1.0 / k)
        logits = (scores - scores.max(axis=1, keepdims=True)) / temperature
        weights = np.exp(logits)
        weights /= weights.sum(axis=1, keepdims=True)
        return weights

    def aggregate_offset_forecast(
        self,
        neighbor_indices,
        neighbor_scores,
        k,
        temperature=0.1,
    ):
        """Weighted mean of each neighbour's own forecast, offset by its context endpoint.

        Adding the query endpoint and subtracting the query forecast turns this
        into the forecast drift term that carries the neighbour's predicted
        trajectory instead of only its realized error.
        """

        k = min(k, neighbor_indices.shape[1])
        selected = neighbor_indices[:, :k]
        weights = self.neighbor_weights(neighbor_scores, k, temperature)
        drift = np.empty(
            (selected.shape[0],) + self.memory_offset_forecasts.shape[1:],
            dtype=np.float32,
        )
        for start in range(0, selected.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, selected.shape[0])
            drift[start:end] = np.einsum(
                "nk,nkpd->npd",
                weights[start:end],
                self.memory_offset_forecasts[selected[start:end]],
                optimize=True,
            )
        return drift

    def aggregate(
        self,
        neighbor_indices,
        neighbor_scores,
        k,
        temperature=0.1,
        shrinkage=0.0,
    ):
        k = min(k, neighbor_indices.shape[1])
        selected = neighbor_indices[:, :k]
        weights = self.neighbor_weights(neighbor_scores, k, temperature)

        corrections = np.empty(
            (selected.shape[0],) + self.memory_residuals.shape[1:], dtype=np.float32
        )
        for start in range(0, selected.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, selected.shape[0])
            residuals = self.memory_residuals[selected[start:end]]
            local_weights = weights[start:end]
            mean = np.einsum(
                "nk,nkpd->npd", weights[start:end], residuals, optimize=True
            )
            if shrinkage > 0:
                second_moment = np.einsum(
                    "nk,nkpd->npd", local_weights, np.square(residuals), optimize=True
                )
                variance = np.maximum(second_moment - np.square(mean), 0.0)
                effective_k = 1.0 / np.square(local_weights).sum(axis=1)
                variance_of_mean = variance / effective_k[:, None, None]
                reliability = np.square(mean) / (
                    np.square(mean) + shrinkage * variance_of_mean + 1e-8
                )
                mean = mean * reliability
            corrections[start:end] = mean
        return corrections


class CausalHistoricalResidualIndex(HistoricalResidualIndex):
    def fit_timeline(self, bundle, normalization_indices):
        features = extract_retrieval_features(bundle, self.feature_config)
        normalization_indices = np.asarray(normalization_indices)
        reference = features[normalization_indices]
        self.feature_mean = reference.mean(axis=0, keepdims=True)
        self.feature_std = reference.std(axis=0, keepdims=True) + 1e-6
        standardized = (features - self.feature_mean) / self.feature_std
        self.memory_features = self._l2_normalize(standardized)
        self.memory_residuals = np.asarray(
            bundle["y"] - bundle["y_base"], dtype=np.float32
        )
        self.memory_offset_forecasts = np.asarray(
            bundle["y_base"] - bundle["x"][:, -1:, :], dtype=np.float32
        )
        self.memory_indices = np.arange(features.shape[0])
        return self

    def query_causal(self, query_indices, origins, min_lag, max_k=32):
        if not hasattr(self, "memory_features"):
            raise RuntimeError("fit_timeline must be called before querying")
        query_indices = np.asarray(query_indices)
        origins = np.asarray(origins)
        k = min(max_k, self.memory_features.shape[0])
        neighbor_indices = np.empty((len(query_indices), k), dtype=np.int64)
        neighbor_scores = np.empty((len(query_indices), k), dtype=np.float32)

        for start in range(0, len(query_indices), self.chunk_size):
            end = min(start + self.chunk_size, len(query_indices))
            local_queries = query_indices[start:end]
            similarities = self.memory_features[local_queries] @ self.memory_features.T
            latest_valid_origin = origins[local_queries, None] - min_lag
            valid = origins[None, :] <= latest_valid_origin
            valid_counts = valid.sum(axis=1)
            if np.any(valid_counts < k):
                raise ValueError("Not enough causal history for requested top-k")
            similarities[~valid] = -np.inf
            selected = np.argpartition(similarities, -k, axis=1)[:, -k:]
            selected_scores = np.take_along_axis(similarities, selected, axis=1)
            order = np.argsort(selected_scores, axis=1)[:, ::-1]
            neighbor_indices[start:end] = np.take_along_axis(selected, order, axis=1)
            neighbor_scores[start:end] = np.take_along_axis(selected_scores, order, axis=1)
        return neighbor_indices, neighbor_scores
