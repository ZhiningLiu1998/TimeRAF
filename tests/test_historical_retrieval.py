import numpy as np

from ts_rag.historical_retrieval import (
    CausalHistoricalResidualIndex,
    HistoricalResidualIndex,
    RetrievalFeatureConfig,
    extract_retrieval_features,
)


def _bundle(sample_count=20, sequence_length=24, channels=2, seed=7):
    rng = np.random.default_rng(seed)
    y_base = rng.normal(size=(sample_count, sequence_length, channels)).astype(np.float32)
    bundle = {
        "x": rng.normal(size=(sample_count, sequence_length, channels)).astype(np.float32),
        "x_mark": rng.normal(size=(sample_count, sequence_length, 4)).astype(np.float32),
        "y_mark": rng.normal(size=(sample_count, sequence_length, 4)).astype(np.float32),
        "y_base": y_base,
    }
    residual = np.arange(sample_count, dtype=np.float32)[:, None, None]
    bundle["y"] = y_base + residual
    return bundle


def test_exact_match_retrieves_own_residual():
    bundle = _bundle()
    config = RetrievalFeatureConfig(tail_len=24, pooled_steps=12)
    index = HistoricalResidualIndex(config).fit(bundle)
    neighbors, scores = index.query_neighbors(bundle, max_k=1)
    np.testing.assert_array_equal(neighbors[:, 0], np.arange(len(bundle["x"])))
    correction = index.aggregate(neighbors, scores, k=1)
    np.testing.assert_allclose(
        correction[:, 0, 0],
        np.arange(len(bundle["x"])),
        rtol=1e-5,
        atol=1e-5,
    )


def test_causal_index_only_returns_fully_observed_residuals():
    bundle = _bundle(sample_count=200)
    config = RetrievalFeatureConfig(tail_len=24, pooled_steps=12)
    index = CausalHistoricalResidualIndex(config).fit_timeline(
        bundle, normalization_indices=np.arange(80)
    )
    query_indices = np.arange(100, 200)
    origins = np.arange(200)
    neighbors, _ = index.query_causal(
        query_indices, origins, min_lag=24, max_k=16
    )
    assert np.all(origins[neighbors] <= origins[query_indices, None] - 24)


def test_feature_width_is_bounded_for_high_dimensional_series():
    bundle = _bundle(channels=100)
    config = RetrievalFeatureConfig(
        tail_len=24,
        pooled_steps=12,
        include_calendar=False,
        max_channels=8,
    )

    features = extract_retrieval_features(bundle, config)

    assert features.shape[0] == len(bundle["x"])
    assert features.shape[1] < 1000
