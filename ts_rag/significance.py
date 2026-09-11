import numpy as np


def per_origin_loss(pred, true, loss):
    error = np.asarray(pred) - np.asarray(true)
    if loss == "mse":
        values = np.square(error)
    elif loss == "mae":
        values = np.abs(error)
    else:
        raise ValueError(f"Unsupported loss: {loss}")
    return values.mean(axis=tuple(range(1, values.ndim)))


def paired_moving_block_bootstrap(
    baseline_pred,
    candidate_pred,
    true,
    loss="mse",
    block_length=96,
    n_resamples=5000,
    seed=2021,
    batch_size=128,
):
    baseline_loss = per_origin_loss(baseline_pred, true, loss)
    candidate_loss = per_origin_loss(candidate_pred, true, loss)
    differences = candidate_loss - baseline_loss
    return moving_block_bootstrap_differences(
        differences,
        loss=loss,
        block_length=block_length,
        n_resamples=n_resamples,
        seed=seed,
        batch_size=batch_size,
    )


def moving_block_bootstrap_differences(
    differences,
    loss="mse",
    block_length=96,
    n_resamples=5000,
    seed=2021,
    batch_size=128,
):
    differences = np.asarray(differences, dtype=np.float64)
    if differences.ndim != 1:
        raise ValueError(
            f"Expected one loss difference per forecast origin, got {differences.shape}"
        )
    sample_count = len(differences)
    block_length = min(block_length, sample_count)
    blocks_per_sample = int(np.ceil(sample_count / block_length))
    offsets = np.arange(block_length)
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(n_resamples, dtype=np.float64)

    for start in range(0, n_resamples, batch_size):
        end = min(start + batch_size, n_resamples)
        starts = rng.integers(
            0,
            sample_count - block_length + 1,
            size=(end - start, blocks_per_sample),
        )
        indices = starts[:, :, None] + offsets[None, None, :]
        indices = indices.reshape(end - start, -1)[:, :sample_count]
        bootstrap_means[start:end] = differences[indices].mean(axis=1)

    lower, upper = np.percentile(bootstrap_means, (2.5, 97.5))
    observed = float(differences.mean())
    return {
        "loss": loss,
        "candidate_minus_baseline": observed,
        "confidence_interval_95": [float(lower), float(upper)],
        "one_sided_p_candidate_not_better": float(
            (np.count_nonzero(bootstrap_means >= 0.0) + 1) / (n_resamples + 1)
        ),
        "block_length": int(block_length),
        "n_resamples": int(n_resamples),
        "forecast_origin_count": int(sample_count),
    }


def paired_multi_seed_moving_block_bootstrap(
    baseline_predictions,
    candidate_predictions,
    truths,
    loss="mse",
    block_length=96,
    n_resamples=5000,
    seed=2021,
    batch_size=128,
):
    if not (
        len(baseline_predictions)
        == len(candidate_predictions)
        == len(truths)
    ):
        raise ValueError("Baseline, candidate, and truth lists must have equal length")
    if not baseline_predictions:
        raise ValueError("At least one model seed is required")

    seed_differences = []
    for baseline, candidate, true in zip(
        baseline_predictions, candidate_predictions, truths
    ):
        baseline_loss = per_origin_loss(baseline, true, loss)
        candidate_loss = per_origin_loss(candidate, true, loss)
        seed_differences.append(candidate_loss - baseline_loss)
    lengths = {len(values) for values in seed_differences}
    if len(lengths) != 1:
        raise ValueError(
            "All model seeds must contain the same forecast-origin timeline"
        )

    result = moving_block_bootstrap_differences(
        np.mean(np.stack(seed_differences, axis=0), axis=0),
        loss=loss,
        block_length=block_length,
        n_resamples=n_resamples,
        seed=seed,
        batch_size=batch_size,
    )
    result["model_seed_count"] = len(seed_differences)
    return result
