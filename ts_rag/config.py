import argparse
import os
import random
from copy import deepcopy

import numpy as np
import torch


DATASET_PRESETS = {
    "ETTh1": {
        "data": "ETTh1",
        "root_path": "./dataset/ETT-small/",
        "data_path": "ETTh1.csv",
        "freq": "h",
        "target": "OT",
        "enc_in": 7,
        "dec_in": 7,
        "c_out": 7,
        "d_model": 16,
        "d_ff": 32,
        "e_layers": 2,
        "channel_independence": 1,
        "down_sampling_layers": 3,
        "down_sampling_window": 2,
        "learning_rate": 0.01,
        "batch_size": 128,
        "train_epochs": 10,
        "patience": 10,
    },
    "ETTh2": {
        "data": "ETTh2",
        "root_path": "./dataset/ETT-small/",
        "data_path": "ETTh2.csv",
        "freq": "h",
        "target": "OT",
        "enc_in": 7,
        "dec_in": 7,
        "c_out": 7,
        "d_model": 16,
        "d_ff": 32,
        "e_layers": 2,
        "channel_independence": 1,
        "down_sampling_layers": 3,
        "down_sampling_window": 2,
        "learning_rate": 0.01,
        "batch_size": 32,
        "train_epochs": 10,
        "patience": 10,
    },
    "ETTm1": {
        "data": "ETTm1",
        "root_path": "./dataset/ETT-small/",
        "data_path": "ETTm1.csv",
        "freq": "t",
        "target": "OT",
        "enc_in": 7,
        "dec_in": 7,
        "c_out": 7,
        "d_model": 16,
        "d_ff": 32,
        "e_layers": 2,
        "channel_independence": 1,
        "down_sampling_layers": 3,
        "down_sampling_window": 2,
        "learning_rate": 0.01,
        "batch_size": 16,
        "train_epochs": 10,
        "patience": 10,
    },
    "ETTm2": {
        "data": "ETTm2",
        "root_path": "./dataset/ETT-small/",
        "data_path": "ETTm2.csv",
        "freq": "t",
        "target": "OT",
        "enc_in": 7,
        "dec_in": 7,
        "c_out": 7,
        "d_model": 32,
        "d_ff": 32,
        "e_layers": 2,
        "channel_independence": 1,
        "down_sampling_layers": 3,
        "down_sampling_window": 2,
        "learning_rate": 0.01,
        "batch_size": 128,
        "train_epochs": 10,
        "patience": 10,
    },
    "Weather": {
        "data": "custom",
        "root_path": "./dataset/weather/",
        "data_path": "weather.csv",
        "freq": "h",
        "target": "OT",
        "enc_in": 21,
        "dec_in": 21,
        "c_out": 21,
        "d_model": 16,
        "d_ff": 32,
        "e_layers": 3,
        "channel_independence": 1,
        "down_sampling_layers": 3,
        "down_sampling_window": 2,
        "learning_rate": 0.01,
        "batch_size": 128,
        "train_epochs": 20,
        "patience": 10,
    },
    "Electricity": {
        "data": "custom",
        "root_path": "./dataset/electricity/",
        "data_path": "electricity.csv",
        "freq": "h",
        "target": "OT",
        "enc_in": 321,
        "dec_in": 321,
        "c_out": 321,
        "d_model": 16,
        "d_ff": 32,
        "e_layers": 3,
        "channel_independence": 1,
        "down_sampling_layers": 3,
        "down_sampling_window": 2,
        "learning_rate": 0.01,
        "batch_size": 32,
        "train_epochs": 20,
        "patience": 10,
    },
    "Traffic": {
        "data": "custom",
        "root_path": "./dataset/traffic/",
        "data_path": "traffic.csv",
        "freq": "h",
        "target": "OT",
        "enc_in": 862,
        "dec_in": 862,
        "c_out": 862,
        "d_model": 32,
        "d_ff": 64,
        "e_layers": 3,
        "channel_independence": 1,
        "down_sampling_layers": 3,
        "down_sampling_window": 2,
        "learning_rate": 0.01,
        "batch_size": 8,
        "train_epochs": 10,
        "patience": 5,
    },
}


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def get_dataset_preset(name):
    if name not in DATASET_PRESETS:
        raise KeyError(f"Unsupported dataset preset: {name}")
    return deepcopy(DATASET_PRESETS[name])


def add_common_args(parser):
    parser.add_argument("--dataset", type=str, default="ETTh1", choices=list(DATASET_PRESETS.keys()))
    parser.add_argument("--task_name", type=str, default="long_term_forecast")
    parser.add_argument("--is_training", type=int, default=1)
    parser.add_argument("--model", type=str, default="TimeMixer")
    parser.add_argument("--model_id", type=str, default="ts_rag")
    parser.add_argument("--des", type=str, default="ts_rag")
    parser.add_argument("--features", type=str, default="M")
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=0)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--root_path", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--freq", type=str, default=None)
    parser.add_argument("--enc_in", type=int, default=None)
    parser.add_argument("--dec_in", type=int, default=None)
    parser.add_argument("--c_out", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--d_ff", type=int, default=None)
    parser.add_argument("--e_layers", type=int, default=None)
    parser.add_argument("--d_layers", type=int, default=1)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--moving_avg", type=int, default=25)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--channel_independence", type=int, default=None)
    parser.add_argument("--decomp_method", type=str, default="moving_avg")
    parser.add_argument("--use_norm", type=int, default=1)
    parser.add_argument("--down_sampling_layers", type=int, default=None)
    parser.add_argument("--down_sampling_window", type=int, default=None)
    parser.add_argument("--down_sampling_method", type=str, default="avg")
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--d_conv", type=int, default=4)
    parser.add_argument("--distil", type=str2bool, default=True)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--train_epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--loss", type=str, default="MSE")
    parser.add_argument("--lradj", type=str, default="type1")
    parser.add_argument("--seasonal_patterns", type=str, default="Monthly")
    parser.add_argument("--inverse", type=str2bool, default=False)
    parser.add_argument("--use_dtw", type=str2bool, default=False)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--augmentation_ratio", type=int, default=0)
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/")
    parser.add_argument("--output_dir", type=str, default="./ts_rag_outputs")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpu_type", type=str, default="cuda")
    parser.add_argument("--use_multi_gpu", type=str2bool, default=False)
    parser.add_argument("--devices", type=str, default="0,1,2,3")
    parser.add_argument("--use_amp", type=str2bool, default=False)
    parser.add_argument("--train_model", type=str2bool, default=True)
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--save_predictions", type=str2bool, default=True)
    parser.add_argument("--save_memory_bank", type=str2bool, default=True)


def add_rag_args(parser):
    parser.add_argument("--residual_mode", type=str, default="moving_avg", choices=["moving_avg"])
    parser.add_argument("--residual_smooth_kernel", type=int, default=5)
    parser.add_argument("--event_patch_len", type=int, default=24)
    parser.add_argument("--event_top_k", type=int, default=3)
    parser.add_argument("--event_min_distance", type=int, default=12)
    parser.add_argument("--event_score_percentile", type=float, default=90.0)
    parser.add_argument("--active_var_percentile", type=float, default=75.0)
    parser.add_argument("--memory_event_limit", type=int, default=0)
    parser.add_argument("--retrieval_top_k", type=int, default=5)
    parser.add_argument("--retrieval_metric", type=str, default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument("--mask_lambda", type=float, default=0.1)
    parser.add_argument("--correction_mode", type=str, default="future_residual", choices=["future_residual", "future_target"])
    parser.add_argument("--aggregation", type=str, default="similarity_weighted", choices=["uniform", "similarity_weighted"])
    parser.add_argument("--blend_alpha", type=float, default=1.0)
    parser.add_argument("--event_split_percentile", type=float, default=75.0)
    parser.add_argument("--raw_retrieval_correction", type=str2bool, default=True)
    parser.add_argument("--memory_bank_path", type=str, default=None)
    parser.add_argument("--prediction_dump_path", type=str, default=None)
    parser.add_argument("--case_count", type=int, default=5)
    parser.add_argument("--viz_variables", type=int, nargs="+", default=[0, 1, 2])


def build_parser(description):
    parser = argparse.ArgumentParser(description=description)
    add_common_args(parser)
    add_rag_args(parser)
    return parser


def apply_dataset_preset(args):
    preset = get_dataset_preset(args.dataset)
    for key, value in preset.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def prepare_device(args):
    args.use_gpu = bool(args.use_gpu and torch.cuda.is_available())
    if args.use_gpu:
        args.device = torch.device(f"cuda:{args.gpu}")
    else:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            args.device = torch.device("mps")
            args.gpu_type = "mps"
        else:
            args.device = torch.device("cpu")
            args.gpu_type = "cpu"
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        args.device_ids = [int(item) for item in args.devices.split(",")]
        args.gpu = args.device_ids[0]
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_args(args):
    args = apply_dataset_preset(args)
    args = prepare_device(args)
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    return args


def build_setting(args, ii=0):
    return "{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}".format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.expand,
        args.d_conv,
        args.factor,
        args.embed,
        args.distil,
        args.des,
        ii,
    )
