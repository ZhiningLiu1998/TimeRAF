import json
import zipfile

from ts_rag.release_checkpoints import build_release_catalog


def _manifest_row(dataset, model, d_model):
    return {
        "id": f"long_term/{dataset}/{model}/96",
        "dataset": dataset,
        "model": model,
        "pred_len": 96,
        "args": {"d_model": d_model},
    }


def test_release_catalog_only_enables_matching_model_dimensions(tmp_path):
    archive = tmp_path / "release.zip"
    members = {
        "TimeFuse Download/checkpoints/96_48_96/"
        "ETTh1_DLinear_dmodel32_epoch10/checkpoint.pth": b"one",
        "TimeFuse Download/checkpoints/96_0_96/"
        "ETTh1_TimeMixer_dmodel16_epoch10/checkpoint.pth": b"two",
    }
    with zipfile.ZipFile(archive, "w") as output:
        for name, value in members.items():
            output.writestr(name, value)

    catalog = build_release_catalog(
        archive,
        [
            _manifest_row("ETTh1", "DLinear", 64),
            _manifest_row("ETTh1", "TimeMixer", 16),
        ],
    )

    dlinear = catalog["checkpoints"]["long_term/ETTh1/DLinear/96"]
    time_mixer = catalog["checkpoints"]["long_term/ETTh1/TimeMixer/96"]
    assert catalog["checkpoint_count"] == 2
    assert catalog["config_compatible_count"] == 1
    assert dlinear["reason"] == "d_model_mismatch"
    assert not dlinear["usable"]
    assert time_mixer["config_compatible"]
    assert not time_mixer["usable"]
    assert json.loads(json.dumps(catalog))["release_sha256"]
