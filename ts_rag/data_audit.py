import csv
import hashlib
from pathlib import Path

import numpy as np

from ts_rag.benchmark import DatasetProtocol


def sha256_file(path, block_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _csv_shape(path):
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="replace", newline="") as source:
        header = next(csv.reader(source))

    newline_count = 0
    last_byte = b""
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            newline_count += block.count(b"\n")
            last_byte = block[-1:]
    line_count = newline_count + int(bool(last_byte) and last_byte != b"\n")
    return {
        "container": "csv",
        "raw_shape": [line_count - 1, len(header)],
        "time_column": header[0],
        "value_columns": len(header) - 1,
    }


def _npz_shape(path):
    with np.load(path, allow_pickle=False) as archive:
        data = archive["data"]
        nonfinite = 0
        for start in range(0, len(data), 1024):
            nonfinite += int(np.size(data[start : start + 1024]) - np.isfinite(data[start : start + 1024]).sum())
        return {
            "container": "npz",
            "archive_keys": list(archive.files),
            "raw_shape": list(data.shape),
            "dtype": str(data.dtype),
            "selected_signal_shape": [int(data.shape[0]), int(data.shape[1])],
            "selected_signal_index": 0,
            "nonfinite_values": nonfinite,
        }


def split_point_counts(protocol, raw_points):
    if protocol.family == "long_term" and protocol.data in {"ETTh1", "ETTh2"}:
        return (12 * 30 * 24, 4 * 30 * 24, 4 * 30 * 24)
    if protocol.family == "long_term" and protocol.data in {"ETTm1", "ETTm2"}:
        return (12 * 30 * 24 * 4, 4 * 30 * 24 * 4, 4 * 30 * 24 * 4)

    num_train = int(raw_points * (0.6 if protocol.family == "pems" else 0.7))
    if protocol.family == "pems":
        val_end = int(raw_points * 0.8)
        return (num_train, val_end - num_train, raw_points - val_end)

    num_test = int(raw_points * 0.2)
    return (num_train, raw_points - num_train - num_test, num_test)


def loader_window_counts(protocol, raw_points, pred_len):
    train_points, val_points, test_points = split_point_counts(protocol, raw_points)
    if protocol.family == "pems":
        windows = (
            train_points - protocol.seq_len - pred_len + 1,
            val_points - protocol.seq_len - pred_len + 1,
            test_points - protocol.seq_len - pred_len + 1,
        )
        return (windows[0], windows[1], windows[2] // protocol.test_stride)

    return (
        train_points - protocol.seq_len - pred_len + 1,
        val_points - pred_len + 1,
        test_points - pred_len + 1,
    )


def _paper_size_interpretation(protocol, raw_points):
    train_points, val_points, test_points = split_point_counts(protocol, raw_points)
    if protocol.family == "long_term":
        derived = (
            train_points - protocol.seq_len + 1,
            val_points + 1,
            test_points + 1,
        )
        semantics = "input windows before requiring a future horizon"
    elif protocol.family == "epf":
        derived = loader_window_counts(protocol, raw_points, protocol.pred_lens[0])
        semantics = "forecast windows at the single published horizon"
    else:
        derived = None
        semantics = (
            "not consistently derivable from the released files and loader; "
            "retain as a paper claim only"
        )
    return {
        "reported": list(protocol.paper_split_sizes),
        "semantics": semantics,
        "derived": None if derived is None else list(derived),
        "matches_derived": None if derived is None else tuple(derived) == protocol.paper_split_sizes,
    }


def dataset_path(protocol, dataset_root):
    dataset_root = Path(dataset_root)
    subdirectory = {
        "long_term": "long_term_forecast",
        "pems": "short_term_forecast/PEMS",
        "epf": "short_term_forecast/EPF",
    }[protocol.family]
    return dataset_root / subdirectory / protocol.data_path


def audit_dataset(protocol: DatasetProtocol, dataset_root):
    path = dataset_path(protocol, dataset_root)
    shape = _npz_shape(path) if path.suffix == ".npz" else _csv_shape(path)
    raw_points = int(shape["raw_shape"][0])
    observed_variates = (
        int(shape["selected_signal_shape"][1])
        if shape["container"] == "npz"
        else int(shape["value_columns"])
    )
    return {
        "dataset": protocol.name,
        "task_family": protocol.family,
        "relative_path": str(path.relative_to(dataset_root)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        **shape,
        "protocol_input_variates": protocol.input_variates,
        "observed_input_variates": observed_variates,
        "input_variates_match": observed_variates == protocol.input_variates,
        "split_points": dict(
            zip(("train", "val", "test"), split_point_counts(protocol, raw_points))
        ),
        "loader_windows_by_horizon": {
            str(pred_len): dict(
                zip(
                    ("train", "val", "test"),
                    loader_window_counts(protocol, raw_points, pred_len),
                )
            )
            for pred_len in protocol.pred_lens
        },
        "paper_size_interpretation": _paper_size_interpretation(protocol, raw_points),
    }


def build_data_audit(protocols, dataset_root):
    rows = [audit_dataset(protocol, dataset_root) for protocol in protocols]
    return {
        "schema_version": 1,
        "dataset_root": str(Path(dataset_root)),
        "dataset_count": len(rows),
        "all_files_present": len(rows) == len(protocols),
        "all_input_variates_match": all(row["input_variates_match"] for row in rows),
        "datasets": rows,
    }
