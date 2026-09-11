import os

import matplotlib.pyplot as plt
import numpy as np


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_dataset_samples(bundle, dataset_name, variables, save_path, max_samples=3):
    _ensure_dir(os.path.dirname(save_path))
    sample_count = min(max_samples, bundle["x"].shape[0])
    fig, axes = plt.subplots(sample_count, 1, figsize=(12, 3 * sample_count), squeeze=False)
    for i in range(sample_count):
        ax = axes[i, 0]
        for dim in variables:
            ax.plot(bundle["x"][i, :, dim], label=f"x_var{dim}")
        ax.set_title(f"{dataset_name} sample {i}")
        ax.legend(loc="upper right", ncol=min(4, len(variables)))
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_detected_event(raw_seq, residual_seq, event, variables, save_path):
    _ensure_dir(os.path.dirname(save_path))
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), squeeze=False)
    for dim in variables:
        axes[0, 0].plot(raw_seq[:, dim], label=f"raw_var{dim}")
        axes[1, 0].plot(residual_seq[:, dim], label=f"res_var{dim}")
    axes[0, 0].axvspan(event.start, event.end - 1, color="tab:red", alpha=0.15)
    axes[1, 0].axvspan(event.start, event.end - 1, color="tab:red", alpha=0.15)
    axes[0, 0].set_title("Raw signal with detected event patch")
    axes[1, 0].set_title("Residual signal with detected event patch")
    axes[0, 0].legend(loc="upper right", ncol=min(4, len(variables)))
    axes[1, 0].legend(loc="upper right", ncol=min(4, len(variables)))
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_case_study(case, variables, save_path):
    _ensure_dir(os.path.dirname(save_path))
    x = case["x"]
    y = case["y"]
    y_base = case["y_base"]
    y_corr = case["y_corr"]
    query = case["query_event"]
    retrieved = case["retrieved"]

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.4, 1.0, 1.2])

    ax0 = fig.add_subplot(gs[0, 0])
    horizon_offset = x.shape[0]
    for dim in variables:
        ax0.plot(np.arange(x.shape[0]), x[:, dim], label=f"input_var{dim}")
        ax0.plot(np.arange(horizon_offset, horizon_offset + y.shape[0]), y[:, dim], linestyle="--", label=f"gt_var{dim}")
        ax0.plot(np.arange(horizon_offset, horizon_offset + y.shape[0]), y_base[:, dim], linestyle=":", label=f"base_var{dim}")
        ax0.plot(np.arange(horizon_offset, horizon_offset + y.shape[0]), y_corr[:, dim], label=f"corr_var{dim}")
    ax0.set_title("Input, baseline, corrected forecast, and ground truth")
    ax0.legend(loc="upper right", ncol=min(4, len(variables)))

    ax1 = fig.add_subplot(gs[1, 0])
    for dim in variables:
        ax1.plot(query["E_raw"][:, dim], label=f"query_raw_var{dim}")
        ax1.plot(query["E_res"][:, dim], linestyle="--", label=f"query_res_var{dim}")
    ax1.set_title("Query event patch")
    ax1.legend(loc="upper right", ncol=min(4, len(variables)))

    ax2 = fig.add_subplot(gs[2, 0])
    for rank, item in enumerate(retrieved[: min(3, len(retrieved))]):
        for dim in variables[:1]:
            ax2.plot(item["E_res"][:, dim], label=f"top{rank+1}_res_var{dim}_score{item['score']:.2f}")
    ax2.set_title("Top retrieved residual events")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
