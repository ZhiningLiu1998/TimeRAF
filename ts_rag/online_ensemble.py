from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge


@dataclass(frozen=True)
class OnlineEnsembleConfig:
    window: int
    ridge_strength: float
    prior: tuple[float, ...]
    horizon_block: int = 0

    def to_dict(self):
        return {
            "window": self.window,
            "ridge_strength": self.ridge_strength,
            "prior": list(self.prior),
            "horizon_block": self.horizon_block,
        }


class SharedNLinearRidge:
    """A channel-shared direct linear forecaster used as an ensemble member."""

    def __init__(self, alpha=100.0):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)

    @staticmethod
    def _features(bundle):
        x = np.asarray(bundle["x"], dtype=np.float32)
        sample_count, seq_len, channel_count = x.shape
        x = x.transpose(0, 2, 1).reshape(sample_count * channel_count, seq_len)
        level = x[:, -1:]
        return x - level, level

    @staticmethod
    def _targets(bundle, level):
        y = np.asarray(bundle["y"], dtype=np.float32)
        sample_count, pred_len, channel_count = y.shape
        y = y.transpose(0, 2, 1).reshape(sample_count * channel_count, pred_len)
        return y - level

    def fit(self, bundle):
        x, level = self._features(bundle)
        self.model.fit(x, self._targets(bundle, level))
        return self

    def predict(self, bundle):
        x, level = self._features(bundle)
        prediction = self.model.predict(x) + level
        sample_count, pred_len, channel_count = bundle["y"].shape
        return (
            prediction.reshape(sample_count, channel_count, pred_len)
            .transpose(0, 2, 1)
            .astype(np.float32)
        )


def overlapping_forecast_ensemble(base, max_age=8, decay=0.75):
    """Average revisions that forecast the same target from recent origins."""

    base = np.asarray(base, dtype=np.float32)
    sample_count, pred_len, _ = base.shape
    result = base.astype(np.float64)
    denominator = np.ones((sample_count, pred_len, 1), dtype=np.float64)
    for age in range(1, min(max_age, pred_len - 1) + 1):
        weight = decay**age
        usable_horizons = pred_len - age
        result[age:, :usable_horizons] += (
            weight * base[:-age, age:]
        )
        denominator[age:, :usable_horizons] += weight
    return (result / denominator).astype(np.float32)


def seasonal_forecast(bundle, period=24):
    pred_len = bundle["y_base"].shape[1]
    repeats = int(np.ceil(pred_len / period))
    return np.tile(bundle["x"][:, -period:, :], (1, repeats, 1))[:, :pred_len]


def make_auxiliary_forecasts(bundle, linear_model):
    return (
        linear_model.predict(bundle),
        overlapping_forecast_ensemble(bundle["y_base"]),
        seasonal_forecast(bundle),
    )


def build_online_statistics(
    bundles,
    auxiliary_forecasts,
    origins,
    horizon_block=0,
):
    """Build target-time prefix sums for causal rolling least squares."""

    member_count = len(auxiliary_forecasts[0])
    pred_len = bundles[0]["y"].shape[1]
    channel_count = bundles[0]["y"].shape[2]
    timeline_length = max(int(split_origins[-1]) + pred_len for split_origins in origins)
    horizon_group_count = (
        int(np.ceil(pred_len / horizon_block)) if horizon_block else 0
    )
    if horizon_group_count:
        gram = np.zeros(
            (
                timeline_length,
                horizon_group_count,
                channel_count,
                member_count,
                member_count,
            ),
            dtype=np.float64,
        )
        cross = np.zeros(
            (timeline_length, horizon_group_count, channel_count, member_count),
            dtype=np.float64,
        )
    else:
        gram = np.zeros(
            (timeline_length, channel_count, member_count, member_count),
            dtype=np.float64,
        )
        cross = np.zeros(
            (timeline_length, channel_count, member_count),
            dtype=np.float64,
        )

    for bundle, split_auxiliary, split_origins in zip(
        bundles, auxiliary_forecasts, origins
    ):
        base = np.asarray(bundle["y_base"], dtype=np.float64)
        residual = np.asarray(bundle["y"], dtype=np.float64) - base
        directions = np.stack(
            [np.asarray(member, dtype=np.float64) - base for member in split_auxiliary],
            axis=-1,
        )
        for horizon in range(pred_len):
            target_times = split_origins + horizon
            local_directions = directions[:, horizon]
            local_residual = residual[:, horizon]
            if horizon_group_count:
                group = horizon // horizon_block
                local_cross = cross[:, group]
                local_gram = gram[:, group]
            else:
                local_cross = cross
                local_gram = gram
            for left in range(member_count):
                np.add.at(
                    local_cross[:, :, left],
                    target_times,
                    local_directions[:, :, left] * local_residual,
                )
                for right in range(member_count):
                    np.add.at(
                        local_gram[:, :, left, right],
                        target_times,
                        local_directions[:, :, left]
                        * local_directions[:, :, right],
                    )

    gram_prefix = np.concatenate(
        [np.zeros_like(gram[:1]), np.cumsum(gram, axis=0)],
        axis=0,
    )
    cross_prefix = np.concatenate(
        [np.zeros_like(cross[:1]), np.cumsum(cross, axis=0)],
        axis=0,
    )
    return gram_prefix, cross_prefix


def causal_online_weights(statistics, query_origins, config):
    gram_prefix, cross_prefix = statistics
    query_origins = np.asarray(query_origins, dtype=np.int64)
    starts = np.maximum(query_origins - config.window, 0)
    local_gram = gram_prefix[query_origins] - gram_prefix[starts]
    local_cross = cross_prefix[query_origins] - cross_prefix[starts]

    member_count = local_cross.shape[-1]
    coefficient_shape = local_cross.shape[1:-1]
    flattened_cross = local_cross.reshape(len(query_origins), -1, member_count)
    flattened_gram = local_gram.reshape(
        len(query_origins), -1, member_count, member_count
    )
    prior = np.asarray(config.prior, dtype=np.float64)
    if prior.shape != (member_count,):
        raise ValueError(
            f"Expected a {member_count}-element prior, received {config.prior}"
        )

    weights = np.empty_like(flattened_cross)
    identity = np.eye(member_count)
    for sample_index in range(len(query_origins)):
        for group_index in range(flattened_cross.shape[1]):
            gram = flattened_gram[sample_index, group_index]
            cross = flattened_cross[sample_index, group_index]
            scale = np.trace(gram) / member_count
            ridge = config.ridge_strength * max(scale, 1e-8)
            coefficients = np.linalg.solve(
                gram + ridge * identity,
                cross + ridge * prior,
            )
            coefficients = np.clip(coefficients, 0.0, 1.0)
            coefficient_sum = coefficients.sum()
            if coefficient_sum > 1.0:
                coefficients /= coefficient_sum
            weights[sample_index, group_index] = coefficients
    return weights.reshape(len(query_origins), *coefficient_shape, member_count)


def apply_online_ensemble(bundle, auxiliary_forecasts, weights):
    base = np.asarray(bundle["y_base"], dtype=np.float64)
    directions = np.stack(
        [np.asarray(member, dtype=np.float64) - base for member in auxiliary_forecasts],
        axis=-1,
    )
    if weights.ndim == 3:
        horizon_weights = weights[:, None, :, :]
    elif weights.ndim == 4:
        pred_len = base.shape[1]
        horizon_block = int(np.ceil(pred_len / weights.shape[1]))
        horizon_weights = np.repeat(weights, horizon_block, axis=1)[:, :pred_len]
    else:
        raise ValueError(f"Unsupported online weight shape: {weights.shape}")
    correction = np.sum(horizon_weights * directions, axis=-1)
    return (base + correction).astype(np.float32)
