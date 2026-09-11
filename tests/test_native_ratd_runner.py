from __future__ import annotations

import numpy as np
import pytest

from scripts.run_native_ratd_baseline import (
    assert_train_only_references,
    repaired_impute,
    split_starts,
)


def test_released_code_split_uses_every_eval_origin() -> None:
    split = split_starts(26_304, 96, 168)

    assert split["train_end"] == 18_149
    assert len(split["train"]) == 17_886
    assert len(split["validation"]) == 2_463
    assert len(split["test"]) == 5_093
    assert split["test"][0] + 96 == 21_044
    assert np.all(np.diff(split["validation"]) == 1)
    assert np.all(np.diff(split["test"]) == 1)


def test_reference_futures_are_inside_training() -> None:
    audit = assert_train_only_references(
        np.array([[0, 10, 17_885]], dtype=np.int64),
        train_end=18_149,
        context_length=96,
        prediction_length=168,
    )

    assert audit["latest_reference_future_end"] == 18_149
    with pytest.raises(ValueError, match="crosses training boundary"):
        assert_train_only_references(
            np.array([[17_886, 0, 1]], dtype=np.int64),
            train_end=18_149,
            context_length=96,
            prediction_length=168,
        )


def test_reverse_diffusion_receives_and_uses_reference() -> None:
    torch = pytest.importorskip("torch")

    class ReferenceAwareDiffusion:
        def __call__(
            self, model_input, side_info, diffusion_step, reference=None
        ):
            if reference is None:
                return torch.zeros_like(model_input[:, 0])
            return torch.ones_like(model_input[:, 0]) * reference.mean()

    class FakeModel:
        device = torch.device("cpu")
        num_steps = 1
        alpha_hat = np.array([0.9], dtype=np.float32)
        alpha = np.array([0.9], dtype=np.float32)
        beta = np.array([0.1], dtype=np.float32)
        diffmodel = ReferenceAwareDiffusion()

    observed = torch.zeros(1, 1, 2)
    cond_mask = torch.zeros_like(observed)
    side_info = torch.zeros(1, 1, 1, 2)
    torch.manual_seed(7)
    without_reference = repaired_impute(
        FakeModel(), observed, cond_mask, side_info, 1, reference=None
    )
    torch.manual_seed(7)
    with_reference = repaired_impute(
        FakeModel(),
        observed,
        cond_mask,
        side_info,
        1,
        reference=torch.ones(1, 1, 3),
    )

    assert not torch.equal(without_reference, with_reference)
