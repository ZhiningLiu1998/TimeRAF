import re
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path


TIMEFUSE_COMMIT = "978e6c6b9e4f246632c269aa0f9beeb099eabcfc"

MODELS = (
    "TimeXer",
    "TimeMixer",
    "PAttn",
    "iTransformer",
    "TimesNet",
    "PatchTST",
    "DLinear",
    "FreTS",
    "FEDformer",
    "Nonstationary_Transformer",
    "LightTS",
    "Informer",
    "Autoformer",
)


@dataclass(frozen=True)
class DatasetProtocol:
    name: str
    family: str
    data: str
    data_path: str
    input_variates: int
    output_variates: int
    seq_len: int
    label_len: int
    pred_lens: tuple[int, ...]
    features: str
    target: str
    freq: str
    metrics: tuple[str, ...]
    metric_space: str
    paper_split_sizes: tuple[int, int, int]
    test_stride: int = 1


def _long_dataset(name, data, data_path, n_variates, freq, split_sizes):
    return DatasetProtocol(
        name=name,
        family="long_term",
        data=data,
        data_path=data_path,
        input_variates=n_variates,
        output_variates=n_variates,
        seq_len=96,
        label_len=48,
        pred_lens=(96, 192, 336, 720),
        features="M",
        target="OT",
        freq=freq,
        metrics=("mse", "mae"),
        metric_space="normalized",
        paper_split_sizes=split_sizes,
    )


def _pems_dataset(name, n_variates, split_sizes):
    return DatasetProtocol(
        name=name,
        family="pems",
        data="PEMS",
        data_path=f"{name}.npz",
        input_variates=n_variates,
        output_variates=n_variates,
        seq_len=96,
        label_len=12,
        pred_lens=(6, 12, 24),
        features="M",
        target="OT",
        freq="t",
        metrics=("mae", "rmse", "mape"),
        metric_space="inverse_scaled",
        paper_split_sizes=split_sizes,
        test_stride=12,
    )


def _epf_dataset(name):
    return DatasetProtocol(
        name=name,
        family="epf",
        data="custom",
        data_path=f"{name}.csv",
        input_variates=3,
        output_variates=1,
        seq_len=168,
        label_len=48,
        pred_lens=(24,),
        features="MS",
        target="OT",
        freq="h",
        metrics=("mse", "mae"),
        metric_space="normalized",
        paper_split_sizes=(36500, 5219, 10460),
    )


DATASETS = (
    _long_dataset("ETTh1", "ETTh1", "ETTh1.csv", 7, "h", (8545, 2881, 2881)),
    _long_dataset("ETTh2", "ETTh2", "ETTh2.csv", 7, "h", (8545, 2881, 2881)),
    _long_dataset("ETTm1", "ETTm1", "ETTm1.csv", 7, "h", (34465, 11521, 11521)),
    _long_dataset("ETTm2", "ETTm2", "ETTm2.csv", 7, "h", (34465, 11521, 11521)),
    _long_dataset("weather", "custom", "weather.csv", 21, "h", (36792, 5271, 10540)),
    _long_dataset(
        "electricity", "custom", "electricity.csv", 321, "h", (18317, 2633, 5261)
    ),
    _long_dataset("traffic", "custom", "traffic.csv", 862, "h", (12185, 1757, 3509)),
    _pems_dataset("PEMS03", 358, (15617, 5135, 5135)),
    _pems_dataset("PEMS04", 307, (10172, 3375, 3375)),
    _pems_dataset("PEMS07", 883, (16911, 5622, 5622)),
    _pems_dataset("PEMS08", 170, (10690, 3548, 265)),
    *(_epf_dataset(name) for name in ("NP", "PJM", "BE", "FR", "DE")),
)


_PARSER_DEFAULTS = {
    "task_name": "long_term_forecast",
    "is_training": 1,
    "model_id": "test",
    "model": "Autoformer",
    "data": "ETTm1",
    "root_path": "./dataset/ETT-small/",
    "data_path": "ETTh1.csv",
    "features": "M",
    "target": "OT",
    "freq": "h",
    "checkpoints": "./checkpoints/",
    "seq_len": 96,
    "label_len": 48,
    "pred_len": 96,
    "seasonal_patterns": "Monthly",
    "inverse": False,
    "top_k": 5,
    "num_kernels": 6,
    "enc_in": 7,
    "dec_in": 7,
    "c_out": 7,
    "d_model": 512,
    "n_heads": 8,
    "e_layers": 2,
    "d_layers": 1,
    "d_ff": 2048,
    "moving_avg": 25,
    "factor": 3,
    "distil": True,
    "dropout": 0.1,
    "embed": "timeF",
    "activation": "gelu",
    "channel_independence": 1,
    "decomp_method": "moving_avg",
    "use_norm": 1,
    "down_sampling_layers": 3,
    "down_sampling_window": 2,
    "down_sampling_method": "avg",
    "seg_len": 48,
    "num_workers": 64,
    "itr": 1,
    "train_epochs": 10,
    "batch_size": 32,
    "patience": 3,
    "learning_rate": 0.0001,
    "des": "Exp",
    "loss": "MSE",
    "lradj": "type1",
    "use_amp": False,
    "use_gpu": True,
    "gpu": 0,
    "gpu_type": "cuda",
    "use_multi_gpu": False,
    "devices": "0",
    "p_hidden_dims": [128, 128],
    "p_hidden_layers": 2,
    "use_dtw": False,
    "augmentation_ratio": 0,
    "seed": 2021,
    "patch_len": 16,
    "expand": 2,
    "d_conv": 4,
    "num_class": 2,
}

_EPF_OVERRIDES = {
    "NP": {"e_layers": 3, "batch_size": 4},
    "PJM": {"e_layers": 3, "batch_size": 16},
    "BE": {"e_layers": 2, "batch_size": 16},
    "FR": {"e_layers": 2, "batch_size": 16},
    "DE": {"e_layers": 1, "batch_size": 4},
}

_PEMS_TIMEMIXER_OVERRIDES = {
    "label_len": 0,
    "e_layers": 5,
    "use_norm": 0,
    "channel_independence": 0,
    "d_model": 128,
    "d_ff": 256,
    "batch_size": 32,
    "learning_rate": 0.003,
    "patience": 10,
    "down_sampling_layers": 1,
    "down_sampling_window": 2,
    "down_sampling_method": "avg",
}

_LIST_ARGUMENTS = {"p_hidden_dims"}


def _coerce(value):
    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _replace_shell_vars(value, variables):
    pattern = re.compile(r"\$(?:\{(?P<braced>\w+)\}|(?P<plain>\w+))")

    def replace(match):
        name = match.group("braced") or match.group("plain")
        return str(variables.get(name, match.group(0)))

    return pattern.sub(replace, value)


def parse_tslib_shell(path):
    """Return explicit argument dictionaries keyed by (seq_len, pred_len)."""
    content = Path(path).read_text(encoding="utf-8")
    variables = {}
    for line in content.splitlines():
        match = re.match(r"^\s*(?:export\s+)?(\w+)=([^#]+?)\s*$", line)
        if match:
            variables[match.group(1)] = match.group(2).strip().strip("'\"")

    result = {}
    merged = re.sub(r"\\\s*\n", " ", content)
    for line in merged.splitlines():
        match = re.search(r"\bpython\s+-u\s+run\.py\s+(.*)$", line)
        if not match:
            continue
        command = _replace_shell_vars(match.group(1), variables)
        tokens = shlex.split(command)
        args = {}
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if not token.startswith("--"):
                index += 1
                continue
            key = token[2:].replace("-", "_")
            if index + 1 == len(tokens) or tokens[index + 1].startswith("--"):
                args[key] = True
                index += 1
            else:
                value_end = index + 1
                while (
                    value_end < len(tokens)
                    and not tokens[value_end].startswith("--")
                ):
                    value_end += 1
                values = [
                    _coerce(value)
                    for value in tokens[index + 1 : value_end]
                ]
                args[key] = values if key in _LIST_ARGUMENTS else values[0]
                index = value_end
        if "seq_len" in args and "pred_len" in args:
            result[(int(args["seq_len"]), int(args["pred_len"]))] = args
    return result


def _config_path(dataset, model, scripts_root):
    scripts_root = Path(scripts_root)
    if dataset.startswith("ETT"):
        relative = Path("ETT_script") / f"{model}_{dataset}.sh"
    elif dataset == "electricity":
        relative = Path("ECL_script") / f"{model}.sh"
    else:
        relative = Path(f"{dataset.title()}_script") / f"{model}.sh"
    path = scripts_root / relative
    return path if path.exists() else None


def _inferred_dimension(n_variates):
    value = 1
    while value < n_variates:
        value *= 2
    return min(max(value, 32), 512)


def build_experiment_args(dataset, model, pred_len, scripts_root):
    args = dict(_PARSER_DEFAULTS)
    config_path = None
    if dataset.family == "long_term":
        config_path = _config_path(dataset.name, model, scripts_root)
    parsed = {}
    if config_path is not None:
        parsed = parse_tslib_shell(config_path).get((dataset.seq_len, pred_len), {})

    if parsed:
        args.update(parsed)
        config_source = "official_shell"
        config_reference = str(config_path)
    else:
        dimension = _inferred_dimension(dataset.input_variates)
        args.update({"d_model": dimension, "d_ff": dimension})
        config_source = "published_fallback"
        config_reference = "TimeFuse/load_configs.py:get_forecast_exp_args"

    root = "./dataset/timefuse/long_term_forecast/"
    if dataset.family == "pems":
        root = "./dataset/timefuse/short_term_forecast/PEMS/"
    elif dataset.family == "epf":
        root = "./dataset/timefuse/short_term_forecast/EPF/"

    args.update(
        {
            "task_name": "long_term_forecast",
            "is_training": 1,
            "model": model,
            "model_id": f"{dataset.name}_{dataset.seq_len}_{pred_len}",
            "data": dataset.data,
            "root_path": root,
            "data_path": dataset.data_path,
            "features": dataset.features,
            "target": dataset.target,
            "freq": dataset.freq,
            "seq_len": dataset.seq_len,
            "label_len": pred_len if dataset.family == "pems" else dataset.label_len,
            "pred_len": pred_len,
            "enc_in": dataset.input_variates,
            "dec_in": dataset.input_variates,
            "c_out": dataset.output_variates,
            "train_epochs": 10,
        }
    )
    if dataset.family == "epf":
        args.update(
            {
                "d_model": 512,
                "d_ff": 512,
                "patch_len": 24,
                **_EPF_OVERRIDES[dataset.name],
            }
        )
        if model == "TimeMixer":
            args["c_out"] = dataset.input_variates
        config_reference = (
            "Time-Series-Library/scripts/exogenous_forecast/EPF/TimeXer.sh "
            "+ TimeFuse fallback"
        )
    if dataset.family == "pems" and model == "TimeMixer":
        args.update(_PEMS_TIMEMIXER_OVERRIDES)
        config_source = "upstream_model_shell"
        config_reference = (
            "kwuking/TimeMixer/scripts/short_term_forecast/PEMS/TimeMixer.sh"
        )
    if model == "TimeMixer":
        args["label_len"] = 0
    if model == "Nonstationary_Transformer":
        args["learning_rate"] = max(float(args["learning_rate"]), 0.001)
    return args, config_source, config_reference


def iter_experiment_manifest(scripts_root="./scripts/long_term_forecast"):
    for dataset in DATASETS:
        for pred_len in dataset.pred_lens:
            for model in MODELS:
                args, config_source, config_reference = build_experiment_args(
                    dataset, model, pred_len, scripts_root
                )
                yield {
                    "id": f"{dataset.family}/{dataset.name}/{model}/{pred_len}",
                    "task_family": dataset.family,
                    "dataset": dataset.name,
                    "model": model,
                    "seq_len": dataset.seq_len,
                    "pred_len": pred_len,
                    "metrics": list(dataset.metrics),
                    "metric_space": dataset.metric_space,
                    "metric_reference": (
                        "kwuking/TimeMixer/utils/metrics.py"
                        if dataset.family == "pems"
                        else "TimeFuse paper Appendix A.2"
                    ),
                    "test_stride": dataset.test_stride,
                    "paper_split_sizes": list(dataset.paper_split_sizes),
                    "protocol": asdict(dataset),
                    "config_source": config_source,
                    "config_reference": config_reference,
                    "timefuse_commit": TIMEFUSE_COMMIT,
                    "args": args,
                }


def all_metrics_improve(baseline, corrected, metrics):
    return all(float(corrected[name]) < float(baseline[name]) for name in metrics)
