import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import scripts.run_native_raf_baseline as raf_runner
from scripts.run_native_raf_baseline import (
    filter_rows_by_cell_ids,
    generate_sample_forecasts_loop_interchanged,
    loop_interchanged_best_matches,
    memory_safe_best_matches,
    seed_everything,
)


class IdentityEmbeddingPipeline:
    def embed(self, values):
        return values.unsqueeze(-1), None


class RecordingPipeline:
    def __init__(self):
        self.embed_inputs = []
        self.predict_inputs = []

    def embed(self, values):
        self.embed_inputs.append(values.detach().cpu().clone())
        return values.unsqueeze(-1), None

    def predict(
        self,
        context,
        prediction_length,
        num_samples,
        **_predict_kwargs,
    ):
        values = torch.stack(context)
        self.predict_inputs.append(values.clone())
        noise = torch.rand(
            (len(values), num_samples, prediction_length),
            dtype=values.dtype,
        )
        return values.sum(dim=1)[:, None, None] + noise


find_best_matches_full_series_batch = memory_safe_best_matches


def fake_batcher(values, batch_size):
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def fake_augment_time_series(
    train_df,
    pipeline,
    context_tensor_matrix,
    prediction_length,
    top_n,
):
    test_length = len(context_tensor_matrix[0])
    matches = find_best_matches_full_series_batch(
        train_df,
        context_tensor_matrix,
        test_length,
        prediction_length,
        pipeline,
        top_n,
    )
    augmented = []
    for query_index, context in enumerate(context_tensor_matrix):
        match = torch.as_tensor(matches[query_index * top_n])
        augmented.append(torch.cat((match, context)))
    return augmented, [None] * len(augmented)


def fake_generate_sample_forecasts(
    train_df,
    augment,
    top_n,
    test_data_input,
    pipeline,
    prediction_length,
    batch_size,
    num_samples,
    **predict_kwargs,
):
    forecasts = []
    for batch in fake_batcher(test_data_input, batch_size):
        context = [torch.tensor(entry["target"]) for entry in batch]
        if augment:
            context, _ = fake_augment_time_series(
                train_df,
                pipeline,
                context,
                prediction_length,
                top_n,
            )
        forecasts.append(
            pipeline.predict(
                context,
                prediction_length=prediction_length,
                num_samples=num_samples,
                **predict_kwargs,
            )
        )
    return torch.cat(forecasts)


FAKE_RUN_CHRONOS = SimpleNamespace(
    augment_time_series=fake_augment_time_series,
    batcher=fake_batcher,
    generate_sample_forecasts=fake_generate_sample_forecasts,
)


def test_streaming_raf_retrieval_matches_full_distance():
    train = [
        {"target": np.array([0.0, 1.0, 2.0, 9.0, 9.0])},
        {"target": np.array([5.0, 6.0, 7.0, 8.0, 9.0])},
    ]
    queries = [
        torch.tensor([0.0, 1.0]),
        torch.tensor([6.0, 7.0]),
    ]

    matches = memory_safe_best_matches(
        train,
        queries,
        test_length=2,
        prediction_length=1,
        pipeline=IdentityEmbeddingPipeline(),
        top_n=1,
        candidate_batch_size=2,
    )

    np.testing.assert_allclose(matches[0], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(matches[1], [6.0, 7.0, 8.0])


def test_loop_interchange_matches_reference_and_embeds_candidates_once():
    train = [
        {"target": np.array([0.0, 1.0, 2.0, 9.0, 9.0])},
        {"target": np.array([5.0, 6.0, 7.0, 8.0, 9.0])},
    ]
    query_batches = [
        [torch.tensor([0.0, 1.0]), torch.tensor([6.0, 7.0])],
        [torch.tensor([5.0, 6.0])],
    ]
    candidate_batch_size = 4
    candidate_count = 6
    candidate_batch_count = (
        candidate_count + candidate_batch_size - 1
    ) // candidate_batch_size

    reference_pipeline = RecordingPipeline()
    reference_matches = [
        memory_safe_best_matches(
            train,
            batch,
            test_length=2,
            prediction_length=1,
            pipeline=reference_pipeline,
            top_n=1,
            candidate_batch_size=candidate_batch_size,
        )
        for batch in query_batches
    ]
    accelerated_pipeline = RecordingPipeline()
    accelerated_matches = loop_interchanged_best_matches(
        train,
        query_batches,
        test_length=2,
        prediction_length=1,
        pipeline=accelerated_pipeline,
        top_n=1,
        candidate_batch_size=candidate_batch_size,
    )

    for reference_batch, accelerated_batch in zip(
        reference_matches, accelerated_matches
    ):
        for reference, accelerated in zip(reference_batch, accelerated_batch):
            np.testing.assert_array_equal(reference, accelerated)

    query_batch_count = len(query_batches)
    reference_candidate_calls = len(reference_pipeline.embed_inputs) - query_batch_count
    accelerated_candidate_calls = (
        len(accelerated_pipeline.embed_inputs) - query_batch_count
    )
    assert reference_candidate_calls == (candidate_batch_count * query_batch_count)
    assert accelerated_candidate_calls == candidate_batch_count
    for observed, expected in zip(
        accelerated_pipeline.embed_inputs[:query_batch_count],
        query_batches,
    ):
        assert torch.equal(observed, torch.stack(expected))
    reference_candidate_inputs = reference_pipeline.embed_inputs[
        1 : 1 + candidate_batch_count
    ]
    accelerated_candidate_inputs = accelerated_pipeline.embed_inputs[
        query_batch_count:
    ]
    for reference_input, accelerated_input in zip(
        reference_candidate_inputs,
        accelerated_candidate_inputs,
    ):
        assert torch.equal(reference_input, accelerated_input)


def test_loop_interchange_preserves_predict_order_results_and_rng():
    train = [
        {"target": np.array([0.0, 1.0, 2.0, 9.0, 9.0])},
        {"target": np.array([5.0, 6.0, 7.0, 8.0, 9.0])},
    ]
    test_entries = [
        {"target": np.array([0.0, 1.0])},
        {"target": np.array([6.0, 7.0])},
        {"target": np.array([5.0, 6.0])},
    ]

    seed_everything(42)
    reference_pipeline = RecordingPipeline()
    reference = fake_generate_sample_forecasts(
        train,
        True,
        top_n=1,
        test_data_input=test_entries,
        pipeline=reference_pipeline,
        prediction_length=1,
        batch_size=2,
        num_samples=3,
    )
    reference_rng_state = torch.get_rng_state()

    seed_everything(42)
    accelerated_pipeline = RecordingPipeline()
    accelerated = generate_sample_forecasts_loop_interchanged(
        FAKE_RUN_CHRONOS,
        train,
        top_n=1,
        test_data_input=test_entries,
        pipeline=accelerated_pipeline,
        prediction_length=1,
        batch_size=2,
        num_samples=3,
    )
    accelerated_rng_state = torch.get_rng_state()

    assert torch.equal(reference, accelerated)
    assert torch.equal(reference_rng_state, accelerated_rng_state)
    assert len(reference_pipeline.predict_inputs) == 2
    assert len(accelerated_pipeline.predict_inputs) == 2
    for reference_input, accelerated_input in zip(
        reference_pipeline.predict_inputs,
        accelerated_pipeline.predict_inputs,
    ):
        assert torch.equal(reference_input, accelerated_input)
    assert find_best_matches_full_series_batch is memory_safe_best_matches


def test_loop_interchange_rejects_candidate_batch_above_512():
    with pytest.raises(ValueError, match="between 1 and 512"):
        loop_interchanged_best_matches(
            [],
            [[torch.tensor([0.0])]],
            test_length=1,
            prediction_length=1,
            pipeline=IdentityEmbeddingPipeline(),
            candidate_batch_size=513,
        )


def test_filter_rows_by_cell_ids_preserves_canonical_order(tmp_path):
    rows = [{"id": "raf/a"}, {"id": "raf/b"}, {"id": "raf/c"}]
    json_path = tmp_path / "cells.json"
    json_path.write_text(json.dumps(["raf/c", "raf/a"]), encoding="utf-8")
    jsonl_path = tmp_path / "cells.jsonl"
    jsonl_path.write_text('"raf/b"\n"raf/a"\n', encoding="utf-8")

    assert [row["id"] for row in filter_rows_by_cell_ids(rows, json_path)] == [
        "raf/a",
        "raf/c",
    ]
    assert [row["id"] for row in filter_rows_by_cell_ids(rows, jsonl_path)] == [
        "raf/a",
        "raf/b",
    ]


def test_main_filters_rows_before_worker_assignment(tmp_path, monkeypatch):
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text("{}", encoding="utf-8")
    cell_ids_path = tmp_path / "cells.json"
    cell_ids_path.write_text('["raf/c", "raf/a"]', encoding="utf-8")
    args = SimpleNamespace(
        protocol=protocol_path,
        cell_ids_path=cell_ids_path,
        max_cells=0,
        worker_count=6,
        worker_gpu_ids=(2, 3, 4, 5, 6, 7),
        worker_id=None,
    )
    captured = {}

    monkeypatch.setattr(raf_runner, "parse_args", lambda: args)
    monkeypatch.setattr(
        raf_runner,
        "raf_rows",
        lambda _protocol: [
            {"id": "raf/a"},
            {"id": "raf/b"},
            {"id": "raf/c"},
        ],
    )

    def capture_parent(received_args, _protocol, rows):
        captured["worker_count"] = received_args.worker_count
        captured["worker_gpu_ids"] = received_args.worker_gpu_ids
        captured["rows"] = rows
        return 0

    monkeypatch.setattr(raf_runner, "run_parent", capture_parent)

    assert raf_runner.main() == 0
    assert captured["worker_count"] == 2
    assert captured["worker_gpu_ids"] == (2, 3)
    assert [row["id"] for row in captured["rows"]] == ["raf/a", "raf/c"]


@pytest.mark.parametrize(
    "payload, expected_error",
    [
        ("[]", "selected no RAF rows"),
        ('[""]', "must not be empty"),
        ('["raf/a", "raf/a"]', "duplicate cell ID"),
        ('["raf/missing"]', "unknown RAF cell ID"),
    ],
)
def test_filter_rows_by_cell_ids_rejects_invalid_input(
    tmp_path, payload, expected_error
):
    path = tmp_path / "cells.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match=expected_error):
        filter_rows_by_cell_ids([{"id": "raf/a"}], path)
