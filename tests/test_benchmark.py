from ts_rag.benchmark import (
    DATASETS,
    MODELS,
    all_metrics_improve,
    iter_experiment_manifest,
    parse_tslib_shell,
)
from ts_rag.data_audit import loader_window_counts, split_point_counts


def test_manifest_has_every_paper_cell():
    rows = list(iter_experiment_manifest())

    assert len(DATASETS) == 16
    assert len(MODELS) == 13
    assert len(rows) == 585
    assert len({row["id"] for row in rows}) == len(rows)
    assert sum(row["task_family"] == "long_term" for row in rows) == 364
    assert sum(row["task_family"] == "pems" for row in rows) == 156
    assert sum(row["task_family"] == "epf" for row in rows) == 65


def test_task_specific_metric_protocols():
    rows = list(iter_experiment_manifest())
    pems = next(row for row in rows if row["dataset"] == "PEMS03")
    pems_timemixer = next(
        row
        for row in rows
        if row["dataset"] == "PEMS03" and row["model"] == "TimeMixer"
    )
    epf = next(row for row in rows if row["dataset"] == "NP")
    epf_timemixer = next(
        row for row in rows if row["dataset"] == "NP" and row["model"] == "TimeMixer"
    )
    ettm1 = next(row for row in rows if row["dataset"] == "ETTm1")

    assert pems["metrics"] == ["mae", "rmse", "mape"]
    assert pems["metric_space"] == "inverse_scaled"
    assert pems["test_stride"] == 12
    assert pems["args"]["freq"] == "t"
    assert pems_timemixer["config_source"] == "upstream_model_shell"
    assert pems_timemixer["args"]["d_model"] == 128
    assert pems_timemixer["args"]["e_layers"] == 5
    assert epf["metrics"] == ["mse", "mae"]
    assert epf["args"]["features"] == "MS"
    assert epf["args"]["enc_in"] == 3
    assert epf["args"]["c_out"] == 1
    assert epf_timemixer["args"]["c_out"] == 3
    assert ettm1["args"]["freq"] == "h"


def test_shell_parser_recovers_horizon_specific_config():
    configs = parse_tslib_shell(
        "scripts/long_term_forecast/ETT_script/TimeXer_ETTh1.sh"
    )

    assert configs[(96, 96)]["d_model"] == 256
    assert configs[(96, 192)]["e_layers"] == 2
    assert configs[(96, 336)]["d_ff"] == 1024
    assert configs[(96, 720)]["batch_size"] == 16


def test_shell_parser_preserves_list_arguments():
    configs = parse_tslib_shell(
        "scripts/long_term_forecast/ETT_script/"
        "Nonstationary_Transformer_ETTh1.sh"
    )

    assert configs[(96, 96)]["p_hidden_dims"] == [256, 256]
    assert configs[(96, 96)]["p_hidden_layers"] == 2


def test_acceptance_requires_strict_improvement_for_every_metric():
    baseline = {"mse": 1.0, "mae": 0.5}

    assert all_metrics_improve(baseline, {"mse": 0.9, "mae": 0.49}, ("mse", "mae"))
    assert not all_metrics_improve(
        baseline, {"mse": 0.9, "mae": 0.5}, ("mse", "mae")
    )


def test_data_audit_window_formulas_cover_all_protocol_families():
    etth1 = next(dataset for dataset in DATASETS if dataset.name == "ETTh1")
    weather = next(dataset for dataset in DATASETS if dataset.name == "weather")
    pems03 = next(dataset for dataset in DATASETS if dataset.name == "PEMS03")
    np_epf = next(dataset for dataset in DATASETS if dataset.name == "NP")

    assert split_point_counts(etth1, 17420) == (8640, 2880, 2880)
    assert loader_window_counts(etth1, 17420, 96) == (8449, 2785, 2785)
    assert split_point_counts(weather, 52696) == (36887, 5270, 10539)
    assert loader_window_counts(weather, 52696, 720) == (36072, 4551, 9820)
    assert loader_window_counts(pems03, 26208, 12) == (15617, 5135, 427)
    assert loader_window_counts(np_epf, 52416, 24) == (36500, 5219, 10460)
