from collections import Counter

from ts_rag.appendix_benchmark import (
    APPENDIX_BASELINES,
    iter_appendix_manifest,
)


def test_appendix_manifest_covers_every_reported_baseline_system():
    rows = list(iter_appendix_manifest())

    assert len(APPENDIX_BASELINES) == 6
    assert len(rows) == 96
    assert len({row["id"] for row in rows}) == 96
    assert Counter(row["baseline"] for row in rows) == {
        baseline.name: 16 for baseline in APPENDIX_BASELINES
    }
    assert Counter(row["task_family"] for row in rows) == {
        "long_term": 42,
        "pems": 24,
        "epf": 30,
    }


def test_appendix_manifest_matches_reported_horizons_and_metrics():
    rows = list(iter_appendix_manifest())

    assert all(
        row["pred_len"] == (96 if row["task_family"] == "long_term" else 24)
        for row in rows
    )
    assert all(
        row["metrics"] == ["mae", "rmse", "mape"]
        for row in rows
        if row["task_family"] == "pems"
    )
    assert all(
        row["metrics"] == ["mse", "mae"]
        for row in rows
        if row["task_family"] != "pems"
    )


def test_appendix_manifest_retains_paper_oot_cells():
    rows = list(iter_appendix_manifest())
    unavailable = [
        row for row in rows if not row["paper_status"]["evaluable"]
    ]

    assert len(unavailable) == 2
    assert {
        (row["baseline"], row["dataset"]) for row in unavailable
    } == {
        ("autogluon_high_quality", "electricity"),
        ("autogluon_high_quality", "traffic"),
    }
    assert sum(row["paper_status"]["evaluable"] for row in rows) == 94
