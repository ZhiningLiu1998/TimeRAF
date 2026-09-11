from dataclasses import asdict, dataclass

from ts_rag.benchmark import DATASETS


PAPER_ARXIV_ID = "2505.18442"
PAPER_PDF_SHA256 = (
    "52db1209be3ab5f3721c69fcee84db2b7edf51a416fc5d56488d5704a0c7e423"
)
PAPER_TABLE = "Appendix Table: Comparison with AutoML and Advanced Ensembles"
AUTOGLOUON_TIMESERIES_VERSION = "1.4.0"
TORCH_CHECKPOINT_COMPATIBILITY_ENV = "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"


def autogluon_checkpoint_compatibility(baseline):
    enabled = baseline == "autogluon_high_quality"
    return {
        "enabled": enabled,
        "environment_variable": (
            TORCH_CHECKPOINT_COMPATIBILITY_ENV if enabled else None
        ),
        "environment_value": "1" if enabled else None,
        "scope": (
            "locally_generated_autogluon_predictor_checkpoints"
            if enabled
            else None
        ),
        "trusted_local_checkpoints_only": enabled,
    }


@dataclass(frozen=True)
class AppendixBaselineProtocol:
    name: str
    display_name: str
    category: str
    implementation: str
    requires_base_model_predictions: bool = False


APPENDIX_BASELINES = (
    AppendixBaselineProtocol(
        name="forward_selection",
        display_name="Forward Selection",
        category="advanced_ensemble",
        implementation="validation_greedy_caruana_ensemble",
        requires_base_model_predictions=True,
    ),
    AppendixBaselineProtocol(
        name="portfolio_ensemble",
        display_name="Portfolio Ensemble",
        category="advanced_ensemble",
        implementation="validation_ranked_subset_mean",
        requires_base_model_predictions=True,
    ),
    AppendixBaselineProtocol(
        name="zeroshot_ensemble",
        display_name="ZeroShot Ensemble",
        category="advanced_ensemble",
        implementation="dataset_meta_feature_similarity_weights",
        requires_base_model_predictions=True,
    ),
    AppendixBaselineProtocol(
        name="autogluon_high_quality",
        display_name="AutoGluon (24 models)",
        category="automl_ensemble",
        implementation="autogluon_timeseries_high_quality",
    ),
    AppendixBaselineProtocol(
        name="chronos_bolt_finetuned",
        display_name="Chronos-Bolt-Base Finetuned",
        category="foundation_model",
        implementation="autogluon_chronos_bolt_base_finetuned",
    ),
    AppendixBaselineProtocol(
        name="chronos_bolt_zeroshot",
        display_name="Chronos-Bolt-Base Zeroshot",
        category="foundation_model",
        implementation="autogluon_chronos_bolt_base_zeroshot",
    ),
)


def _paper_status(dataset, baseline):
    if (
        baseline.name == "autogluon_high_quality"
        and dataset.family == "long_term"
        and dataset.name in {"electricity", "traffic"}
    ):
        return {
            "status": "paper_oot",
            "evaluable": False,
            "reason": "Paper reports AutoGluon inference over three hours.",
        }
    return {"status": "reported", "evaluable": True, "reason": None}


def iter_appendix_manifest():
    for dataset in DATASETS:
        pred_len = 96 if dataset.family == "long_term" else 24
        if pred_len not in dataset.pred_lens:
            raise ValueError(
                f"Appendix horizon {pred_len} is absent for {dataset.name}"
            )
        for baseline in APPENDIX_BASELINES:
            paper_status = _paper_status(dataset, baseline)
            yield {
                "id": (
                    f"appendix/{dataset.family}/{dataset.name}/"
                    f"{baseline.name}/{pred_len}"
                ),
                "scope": "appendix_additional_baselines",
                "task_family": dataset.family,
                "dataset": dataset.name,
                "baseline": baseline.name,
                "baseline_display_name": baseline.display_name,
                "baseline_category": baseline.category,
                "implementation": baseline.implementation,
                "requires_base_model_predictions": (
                    baseline.requires_base_model_predictions
                ),
                "seq_len": dataset.seq_len,
                "pred_len": pred_len,
                "metrics": list(dataset.metrics),
                "metric_space": dataset.metric_space,
                "test_stride": dataset.test_stride,
                "paper_split_sizes": list(dataset.paper_split_sizes),
                "protocol": asdict(dataset),
                "paper_status": paper_status,
                "paper_reference": {
                    "arxiv_id": PAPER_ARXIV_ID,
                    "pdf_sha256": PAPER_PDF_SHA256,
                    "table": PAPER_TABLE,
                },
            }
