from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from scripts.run_native_ts_rag_baseline import (
    assert_train_only_indices,
    exact_l2_top_k,
    load_ts_rag_model_module,
    select_safe_released_candidates,
    split_boundaries,
)


def test_ett_hour_split_starts_test_context_before_test_targets() -> None:
    boundaries = split_boundaries("ett_hour", 14_400, 512)

    assert boundaries == {
        "train_end": 8_640,
        "test_context_start": 11_008,
        "test_end": 14_400,
    }


def test_retrieval_future_must_end_inside_training() -> None:
    accepted = assert_train_only_indices(
        np.array([[0, 10], [8_063, 4_000]], dtype=np.int64),
        train_end=8_640,
        context_length=512,
        prediction_length=64,
    )

    assert accepted["latest_retrieved_future_end"] == 8_639
    with pytest.raises(ValueError, match="crosses the training boundary"):
        assert_train_only_indices(
            np.array([[8_065]], dtype=np.int64),
            train_end=8_640,
            context_length=512,
            prediction_length=64,
        )


def test_safe_candidate_selection_marks_short_rows_for_exact_search() -> None:
    indices = np.array([[0, 9, 4], [8, 9, 6]], dtype=np.int64)
    distances = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    selected, selected_distances, fallback = (
        select_safe_released_candidates(
            indices, distances, maximum_start=7, top_k=2
        )
    )

    np.testing.assert_array_equal(selected[0], [0, 4])
    np.testing.assert_allclose(selected_distances[0], [0.1, 0.3])
    np.testing.assert_array_equal(fallback, [1])


def test_exact_l2_top_k_returns_sorted_candidate_indices() -> None:
    candidates = np.array([[0.0], [3.0], [1.0], [2.0]], dtype=np.float32)

    indices, distances = exact_l2_top_k(
        np.array([[1.1]], dtype=np.float32), candidates, top_k=3
    )

    np.testing.assert_array_equal(indices, [[2, 3, 0]])
    np.testing.assert_allclose(
        distances, [[0.01, 0.81, 1.21]], atol=1e-6
    )


def test_load_ts_rag_models_in_isolated_namespace(tmp_path) -> None:
    model_root = tmp_path / "TS-RAG" / "models"
    model_root.mkdir(parents=True)
    (model_root / "base.py").write_text("VALUE = 7\n", encoding="utf-8")
    (model_root / "ChronosBolt.py").write_text(
        "from .base import VALUE\n",
        encoding="utf-8",
    )

    module = load_ts_rag_model_module(tmp_path)

    assert module.VALUE == 7
    assert module.__name__ == (
        "_timeraf_native_ts_rag_models.ChronosBolt"
    )


def test_ts_rag_asset_digests_are_well_formed() -> None:
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "docs"
            / "native_ts_rag_assets.json"
        ).read_text(encoding="utf-8")
    )
    records = manifest["checkpoint_files"] + manifest["data_files"]

    assert records
    assert all(
        re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
        for record in records
    )
