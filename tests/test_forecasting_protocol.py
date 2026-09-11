from types import SimpleNamespace

import numpy as np
from sklearn.preprocessing import StandardScaler

from data_provider.data_loader import Dataset_PEMS
from ts_rag.evaluation import compute_pems_metrics
from ts_rag.exp_wrapper import ExpLongTermForecastRAG


def test_pems_loader_matches_timefuse_split_and_test_stride(tmp_path):
    data = np.arange(1200 * 4, dtype=np.float64).reshape(1200, 4, 1)
    np.savez(tmp_path / "PEMS03.npz", data=data)
    args = SimpleNamespace()
    kwargs = {
        "args": args,
        "root_path": str(tmp_path),
        "data_path": "PEMS03.npz",
        "size": [96, 12, 12],
    }

    train = Dataset_PEMS(flag="train", **kwargs)
    val = Dataset_PEMS(flag="val", **kwargs)
    test = Dataset_PEMS(flag="test", **kwargs)

    assert len(train) == 613
    assert len(val) == 133
    assert len(test) == 11
    assert train[0][0].shape == (96, 4)
    assert train[0][1].shape == (24, 4)
    np.testing.assert_allclose(train.data_x.mean(axis=0), 0.0, atol=1e-12)


def test_ms_bundle_uses_target_channel_and_inverts_with_target_scaler():
    exp = ExpLongTermForecastRAG.__new__(ExpLongTermForecastRAG)
    exp.args = SimpleNamespace(features="MS")
    values = np.arange(30, dtype=np.float64).reshape(10, 3)
    scaler = StandardScaler().fit(values)
    dataset = SimpleNamespace(
        scaler=scaler,
        inverse_transform=scaler.inverse_transform,
    )
    scaled_target = np.array([[[-1.0], [0.0], [1.0]]])

    assert exp._output_slice() == slice(-1, None)
    inverted = exp._inverse_output(dataset, scaled_target)
    expected = scaled_target * scaler.scale_[-1] + scaler.mean_[-1]
    np.testing.assert_allclose(inverted, expected)


def test_pems_mape_matches_timemixer_outlier_mask():
    true = np.array([1.0, 0.0, 1.0, 10.0])
    pred = np.array([1.5, 2.0, 8.0, 12.0])

    metrics = compute_pems_metrics(pred, true)

    # Relative errors are [0.5, inf, 7.0, 0.2]; TimeMixer masks > 5.
    assert metrics["mape"] == 0.175
