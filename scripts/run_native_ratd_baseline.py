#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def save_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def verify_assets(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = Path(manifest["asset_root"])
    verified = []
    for record in manifest["files"]:
        path = root / record["path"]
        stat = path.stat()
        if stat.st_size != record["size_bytes"]:
            raise ValueError(f"Size mismatch for {path}")
        digest = sha256_file(path)
        if digest != record["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {path}")
        verified.append(
            {
                "path": str(path),
                "size_bytes": stat.st_size,
                "sha256": digest,
            }
        )
    return {"manifest": str(manifest_path), "files": verified}


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_starts(
    length: int, context_length: int, prediction_length: int
) -> dict[str, np.ndarray | int]:
    train_end = (
        int(length * 0.7) - prediction_length - context_length + 1
    )
    validation_length = int(length * 0.1)
    test_length = int(length * 0.2)
    train_starts = np.arange(
        0,
        train_end - context_length - prediction_length + 1,
        dtype=np.int64,
    )
    validation_first = train_end - context_length
    validation_end = train_end + validation_length
    validation_starts = np.arange(
        validation_first,
        validation_end - context_length - prediction_length + 1,
        dtype=np.int64,
    )
    test_first = length - test_length - context_length
    test_starts = np.arange(
        test_first,
        length - context_length - prediction_length + 1,
        dtype=np.int64,
    )
    return {
        "train_end": train_end,
        "train": train_starts,
        "validation": validation_starts,
        "test": test_starts,
    }


def assert_train_only_references(
    references: np.ndarray,
    train_end: int,
    context_length: int,
    prediction_length: int,
) -> dict:
    if references.ndim != 2 or references.shape[1] != 3:
        raise ValueError("RATD references must have shape [queries, 3]")
    minimum = int(references.min())
    maximum = int(references.max())
    if minimum < 0:
        raise ValueError("RATD reference index is negative")
    latest_end = maximum + context_length + prediction_length
    if latest_end > train_end:
        raise ValueError(
            "RATD reference future crosses training boundary: "
            f"{latest_end} > {train_end}"
        )
    return {
        "minimum_reference_start": minimum,
        "maximum_reference_start": maximum,
        "latest_reference_future_end": latest_end,
        "train_end": train_end,
    }


def load_electricity(path: Path) -> np.ndarray:
    frame = pd.read_csv(path, index_col="date", parse_dates=True)
    values = frame.to_numpy(dtype=np.float32)
    if values.shape != (26_304, 321):
        raise ValueError(
            f"Unexpected reconstructed Electricity shape: {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("Reconstructed Electricity data are non-finite")
    return values


def tcn_encode(model, values: np.ndarray, starts: np.ndarray, batch_size: int):
    import torch

    outputs = []
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(starts), batch_size):
            batch_starts = starts[offset : offset + batch_size]
            batch = np.stack(
                [values[start : start + 96].T for start in batch_starts]
            )
            tensor = torch.from_numpy(batch).float().to(device)
            embedded = model.drop(model.encoder(tensor))
            encoded = model.tcn(embedded.transpose(1, 2))
            outputs.append(encoded[:, :, -1].float().cpu())
    return torch.cat(outputs, dim=0)


def exact_gpu_top3(queries, candidates, batch_size: int = 512) -> np.ndarray:
    import torch

    device = torch.device("cuda:0")
    candidate_tensor = candidates.to(device)
    candidate_norm = candidate_tensor.square().sum(dim=1).unsqueeze(0)
    results = []
    for offset in range(0, len(queries), batch_size):
        query = queries[offset : offset + batch_size].to(device)
        distances = (
            query.square().sum(dim=1, keepdim=True)
            + candidate_norm
            - 2.0 * query @ candidate_tensor.T
        )
        results.append(
            torch.topk(
                distances, k=3, dim=1, largest=False, sorted=True
            ).indices.cpu()
        )
    return torch.cat(results, dim=0).numpy().astype(np.int64)


def build_reference_map(
    data_path: Path,
    tcn_checkpoint: Path,
    tcn_source_root: Path,
    output_path: Path,
    seed: int,
) -> dict:
    import torch

    seed_everything(seed)
    values = load_electricity(data_path)
    splits = split_starts(len(values), 96, 168)
    tcn_train_end = int(len(values) * 0.7)
    means = values[:tcn_train_end].mean(axis=0, dtype=np.float64)
    scales = values[:tcn_train_end].std(axis=0, dtype=np.float64)
    scales[scales == 0] = 1.0
    standardized = ((values - means) / scales).astype(np.float32)

    sys.path.insert(0, str(tcn_source_root))
    model = torch.load(
        tcn_checkpoint,
        map_location="cuda:0",
        weights_only=False,
    )
    model.to("cuda:0").eval()
    train_starts = splits["train"]
    candidate_embeddings = tcn_encode(
        model, standardized, train_starts, batch_size=16
    )
    references = {}
    for split in ("train", "validation", "test"):
        starts = splits[split]
        if split == "train":
            query_embeddings = candidate_embeddings
        else:
            query_embeddings = tcn_encode(
                model, standardized, starts, batch_size=16
            )
        selected = exact_gpu_top3(query_embeddings, candidate_embeddings)
        references[split] = train_starts[selected]

    audits = {
        split: assert_train_only_references(
            references[split], int(splits["train_end"]), 96, 168
        )
        for split in references
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        train_starts=splits["train"],
        train_references=references["train"],
        validation_starts=splits["validation"],
        validation_references=references["validation"],
        test_starts=splits["test"],
        test_references=references["test"],
    )
    metadata = {
        "schema_version": 1,
        "seed": seed,
        "context_length": 96,
        "prediction_length": 168,
        "top_k": 3,
        "train_end": int(splits["train_end"]),
        "query_counts": {
            split: int(len(splits[split]))
            for split in ("train", "validation", "test")
        },
        "audits": audits,
        "data_path": str(data_path),
        "data_sha256": sha256_file(data_path),
        "tcn_checkpoint": str(tcn_checkpoint),
        "tcn_checkpoint_sha256": sha256_file(tcn_checkpoint),
        "tcn_embedding": (
            "Serialized Linear(96,96), dropout, and TemporalConvNet; "
            "the final temporal hidden state is the 400-dimensional embedding."
        ),
        "reference_map": str(output_path),
        "reference_map_sha256": sha256_file(output_path),
    }
    save_json_atomic(output_path.with_suffix(".json"), metadata)
    return metadata


def repaired_rma_forward(self, x, cond_info, reference, return_attn=False):
    import torch
    from einops import rearrange, repeat

    batch, channels, features, length = x.shape
    heads = self.heads
    x = self.norm(x)
    cond_info = self.norm(cond_info)
    reference = self.context_norm(reference)
    reference = repeat(
        reference, "b n c -> (b f) n c", f=channels
    )
    query = self.y_to_q(x.reshape(batch * channels, features, length))
    key = self.cond_to_k(
        torch.cat(
            (
                x.reshape(batch * channels, features, length),
                cond_info.reshape(batch * channels, features, length),
                reference,
            ),
            dim=-1,
        )
    )
    value = self.ref_to_v(
        torch.cat(
            (
                x.reshape(batch * channels, features, length),
                reference,
            ),
            dim=-1,
        )
    )
    query, key, value = map(
        lambda tensor: rearrange(
            tensor, "b n (h d) -> b h n d", h=heads
        ),
        (query, key, value),
    )
    similarity = torch.einsum(
        "b h i d, b h j d -> b h i j", key, value
    ) * self.scale
    attention = self.talking_heads(
        self.dropout(similarity.softmax(dim=-1))
    )
    context_attention = self.context_talking_heads(
        self.context_dropout(similarity.softmax(dim=-2))
    )
    output = torch.einsum(
        "b h i j, b h j d -> b h i d", attention, value
    )
    context_output = torch.einsum(
        "b h j i, b h j d -> b h i d", context_attention, key
    )
    output = rearrange(output, "b h n d -> b n (h d)")
    context_output = rearrange(
        context_output, "b h n d -> b n (h d)"
    )
    output = self.to_out(output)
    if return_attn:
        return output, context_output, attention, context_attention
    return output


def repaired_process_data(self, batch):
    import torch

    observed_data = batch["observed_data"].to(self.device).float()
    observed_mask = batch["observed_mask"].to(self.device).float()
    observed_tp = batch["timepoints"].to(self.device).float()
    gt_mask = batch["gt_mask"].to(self.device).float()
    feature_id = batch["feature_id"].to(self.device).long()
    reference = (
        batch["reference"].to(self.device).float().permute(0, 2, 1)
        if self.use_reference
        else None
    )
    observed_data = observed_data.permute(0, 2, 1)
    observed_mask = observed_mask.permute(0, 2, 1)
    gt_mask = gt_mask.permute(0, 2, 1)
    cut_length = torch.zeros(len(observed_data)).long().to(self.device)
    return (
        observed_data,
        observed_mask,
        observed_tp,
        gt_mask,
        observed_mask,
        cut_length,
        feature_id,
        reference,
    )


def repaired_get_side_info(self, observed_tp, cond_mask, feature_id=None):
    import torch

    batch, features, length = cond_mask.shape
    time_embed = self.time_embedding(
        observed_tp, self.emb_time_dim
    ).unsqueeze(2).expand(-1, -1, features, -1)
    if feature_id is None:
        feature_id = (
            torch.arange(features, device=self.device)
            .unsqueeze(0)
            .expand(batch, -1)
        )
    feature_embed = (
        self.embed_layer(feature_id).unsqueeze(1).expand(-1, length, -1, -1)
    )
    side_info = torch.cat((time_embed, feature_embed), dim=-1).permute(
        0, 3, 2, 1
    )
    if not self.is_unconditional:
        side_info = torch.cat((side_info, cond_mask.unsqueeze(1)), dim=1)
    return side_info


def sample_features_with_reference(
    self,
    observed_data,
    observed_mask,
    feature_id,
    gt_mask,
    reference,
):
    import torch

    selected_data = []
    selected_mask = []
    selected_ids = []
    selected_gt = []
    selected_reference = []
    for batch_index in range(len(observed_data)):
        indices = np.arange(self.target_dim_base)
        np.random.shuffle(indices)
        indices = torch.as_tensor(
            indices[: self.num_sample_features],
            device=observed_data.device,
        )
        selected_data.append(observed_data[batch_index, indices])
        selected_mask.append(observed_mask[batch_index, indices])
        selected_ids.append(feature_id[batch_index, indices])
        selected_gt.append(gt_mask[batch_index, indices])
        if reference is not None:
            selected_reference.append(reference[batch_index, indices])
    self.target_dim = self.num_sample_features
    return (
        torch.stack(selected_data),
        torch.stack(selected_mask),
        torch.stack(selected_ids),
        torch.stack(selected_gt),
        torch.stack(selected_reference) if reference is not None else None,
    )


def repaired_forward(self, batch, is_train=1):
    (
        observed_data,
        observed_mask,
        observed_tp,
        gt_mask,
        _,
        _,
        feature_id,
        reference,
    ) = self.process_data(batch)
    if is_train == 1 and self.target_dim_base > self.num_sample_features:
        (
            observed_data,
            observed_mask,
            feature_id,
            gt_mask,
            reference,
        ) = self.sample_features_with_reference(
            observed_data,
            observed_mask,
            feature_id,
            gt_mask,
            reference,
        )
    else:
        self.target_dim = self.target_dim_base
    cond_mask = self.get_test_pattern_mask(observed_mask, gt_mask)
    side_info = self.get_side_info(observed_tp, cond_mask, feature_id)
    loss_function = self.calc_loss if is_train == 1 else self.calc_loss_valid
    return loss_function(
        observed_data,
        cond_mask,
        observed_mask,
        side_info,
        is_train,
        reference=reference,
    )


def repaired_impute(
    self,
    observed_data,
    cond_mask,
    side_info,
    n_samples,
    reference=None,
):
    import torch

    batch, features, length = observed_data.shape
    samples = torch.zeros(
        batch, n_samples, features, length, device=self.device
    )
    for sample_index in range(n_samples):
        current = torch.randn_like(observed_data)
        for step in range(self.num_steps - 1, -1, -1):
            cond_obs = (cond_mask * observed_data).unsqueeze(1)
            noisy_target = ((1 - cond_mask) * current).unsqueeze(1)
            model_input = torch.cat((cond_obs, noisy_target), dim=1)
            diffusion_step = torch.full(
                (batch,), step, device=self.device, dtype=torch.long
            )
            predicted = self.diffmodel(
                model_input,
                side_info,
                diffusion_step,
                reference=reference,
            )
            coefficient_1 = 1 / self.alpha_hat[step] ** 0.5
            coefficient_2 = (
                (1 - self.alpha_hat[step]) / (1 - self.alpha[step]) ** 0.5
            )
            current = coefficient_1 * (
                current - coefficient_2 * predicted
            )
            if step > 0:
                sigma = (
                    (1 - self.alpha[step - 1])
                    / (1 - self.alpha[step])
                    * self.beta[step]
                ) ** 0.5
                current += sigma * torch.randn_like(current)
        samples[:, sample_index] = current.detach()
    return samples


def repaired_evaluate(self, batch, n_samples):
    (
        observed_data,
        observed_mask,
        observed_tp,
        gt_mask,
        _,
        _,
        feature_id,
        reference,
    ) = self.process_data(batch)
    self.target_dim = self.target_dim_base
    with __import__("torch").no_grad():
        cond_mask = gt_mask
        target_mask = observed_mask * (1 - gt_mask)
        side_info = self.get_side_info(
            observed_tp, cond_mask, feature_id
        )
        samples = self.impute(
            observed_data,
            cond_mask,
            side_info,
            n_samples,
            reference=reference,
        )
    return samples, observed_data, target_mask


def install_repairs(upstream_root: Path):
    sys.path.insert(0, str(upstream_root))
    import diff_models  # type: ignore
    import main_model  # type: ignore

    diff_models.ReferenceModulatedCrossAttention.forward = (
        repaired_rma_forward
    )
    model_class = main_model.RATD_Forecasting
    model_class.process_data = repaired_process_data
    model_class.get_side_info = repaired_get_side_info
    model_class.sample_features_with_reference = (
        sample_features_with_reference
    )
    model_class.forward = repaired_forward
    model_class.impute = repaired_impute
    model_class.evaluate = repaired_evaluate
    return model_class


def make_dataset_class():
    import torch

    class ElectricityDataset(torch.utils.data.Dataset):
        def __init__(
            self,
            values: np.ndarray,
            starts: np.ndarray,
            references: np.ndarray,
            train_end: int,
            use_reference: bool,
        ):
            means = values[:train_end].mean(axis=0, dtype=np.float64)
            scales = values[:train_end].std(axis=0, dtype=np.float64)
            scales[scales == 0] = 1.0
            self.values = ((values - means) / scales).astype(np.float32)
            self.starts = starts
            self.references = references
            self.use_reference = use_reference

        def __len__(self):
            return len(self.starts)

        def __getitem__(self, index):
            start = int(self.starts[index])
            observed = self.values[start : start + 264]
            target_mask = np.ones_like(observed, dtype=np.float32)
            target_mask[-168:] = 0.0
            if self.use_reference:
                reference = np.concatenate(
                    [
                        self.values[
                            ref_start + 96 : ref_start + 96 + 168
                        ]
                        for ref_start in self.references[index]
                    ],
                    axis=0,
                )
            else:
                reference = np.zeros((504, 321), dtype=np.float32)
            return {
                "observed_data": observed,
                "observed_mask": np.ones_like(
                    observed, dtype=np.float32
                ),
                "gt_mask": target_mask,
                "timepoints": np.arange(264, dtype=np.float32),
                "feature_id": np.arange(321, dtype=np.int64),
                "reference": reference,
            }

    return ElectricityDataset


def load_reference_map(path: Path) -> dict:
    with np.load(path) as bundle:
        return {key: bundle[key] for key in bundle.files}


def save_training_checkpoint(
    path: Path,
    epoch: int,
    model,
    optimizer,
    scheduler,
    generator,
    identity: dict,
) -> None:
    import torch

    payload = {
        "identity": identity,
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(),
        "loader_rng": generator.get_state(),
    }
    temporary = path.with_suffix(f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train_and_evaluate(args: argparse.Namespace, use_reference: bool) -> dict:
    import torch
    import yaml

    seed_everything(args.seed)
    model_class = install_repairs(args.upstream_root)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config["train"]["epochs"] = args.epochs
    config["model"]["use_reference"] = use_reference
    config["model"]["is_unconditional"] = False
    config["diffusion"]["h_size"] = 96
    config["diffusion"]["ref_size"] = 168
    if int(config["diffusion"]["num_steps"]) != args.diffusion_steps:
        raise ValueError("Diffusion-step configuration drift")

    values = load_electricity(args.data)
    reference_map = load_reference_map(args.reference_map)
    split = split_starts(len(values), 96, 168)
    dataset_class = make_dataset_class()
    train_dataset = dataset_class(
        values,
        reference_map["train_starts"],
        reference_map["train_references"],
        int(split["train_end"]),
        use_reference,
    )
    test_dataset = dataset_class(
        values,
        reference_map["test_starts"],
        reference_map["test_references"],
        int(split["train_end"]),
        use_reference,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0
    )
    device = torch.device("cuda:0")
    model = model_class(config, device, 321).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=1e-6,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[
            int(0.75 * args.epochs),
            int(0.9 * args.epochs),
        ],
        gamma=0.1,
    )
    method = "RATD" if use_reference else "CSDI"
    artifact = args.output_root / method.lower()
    artifact.mkdir(parents=True, exist_ok=True)
    checkpoint = artifact / "training_checkpoint.pt"
    protocol_sha256 = sha256_file(args.protocol)
    checkpoint_identity = {
        "method": method,
        "source_revision": args.source_revision,
        "upstream_commit": args.upstream_commit,
        "protocol_sha256": protocol_sha256,
        "config_sha256": sha256_file(args.config),
        "data_sha256": sha256_file(args.data),
        "reference_map_sha256": sha256_file(args.reference_map),
        "seed": args.seed,
        "epochs": args.epochs,
        "diffusion_steps": args.diffusion_steps,
        "samples": args.samples,
    }
    start_epoch = 0
    if checkpoint.exists():
        state = torch.load(
            checkpoint, map_location=device, weights_only=False
        )
        if state.get("identity") != checkpoint_identity:
            raise ValueError(
                f"{method} checkpoint identity does not match this run"
            )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        generator.set_state(state["loader_rng"])
        start_epoch = int(state["epoch"])

    started = time.time()
    save_json_atomic(
        artifact / "status.json",
        {
            "status": "running",
            "method": method,
            "pid": os.getpid(),
            "started_unix": started,
            "resumed_epoch": start_epoch,
        },
    )
    for epoch in range(start_epoch, args.epochs):
        model.train()
        loss_total = 0.0
        batches = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model(batch)
            if not torch.isfinite(loss):
                raise ValueError(
                    f"{method} non-finite training loss at epoch {epoch}"
                )
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach().item())
            batches += 1
        scheduler.step()
        save_training_checkpoint(
            checkpoint,
            epoch + 1,
            model,
            optimizer,
            scheduler,
            generator,
            checkpoint_identity,
        )
        save_json_atomic(
            artifact / "status.json",
            {
                "status": "running",
                "method": method,
                "pid": os.getpid(),
                "completed_epochs": epoch + 1,
                "epochs": args.epochs,
                "last_epoch_mean_loss": loss_total / batches,
            },
        )

    model.eval()
    squared = 0.0
    absolute = 0.0
    points = 0
    with torch.inference_mode():
        for batch in test_loader:
            samples, target, target_mask = model.evaluate(
                batch, args.samples
            )
            median = samples.median(dim=1).values
            error = (median - target) * target_mask
            squared += float(error.square().sum().item())
            absolute += float(error.abs().sum().item())
            points += int(target_mask.sum().item())
    mse = squared / points
    metrics = {"rmse": float(np.sqrt(mse)), "mae": absolute / points}
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError(f"{method} produced non-finite metrics")
    finished = time.time()
    result = {
        "schema_version": 1,
        "status": "completed",
        "method": method,
        "use_reference": use_reference,
        "backbone": "CSDI",
        "dataset": "electricity",
        "context_length": 96,
        "prediction_length": 168,
        "seed": args.seed,
        "epochs": args.epochs,
        "diffusion_steps": args.diffusion_steps,
        "samples": args.samples,
        "test_origins": len(test_dataset),
        "test_points": points,
        "metrics": metrics,
        "protocol_sha256": protocol_sha256,
        "source_revision": args.source_revision,
        "upstream_commit": args.upstream_commit,
        "reference_map_sha256": sha256_file(args.reference_map),
        "started_unix": started,
        "finished_unix": finished,
        "elapsed_seconds": finished - started,
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_device": "cuda:0",
    }
    save_json_atomic(artifact / "result.json", result)
    save_json_atomic(
        artifact / "status.json",
        {
            "status": "completed",
            "method": method,
            "finished_unix": finished,
        },
    )
    return result


def query_gpu_processes() -> list[dict]:
    output = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    rows = []
    for line in output.splitlines():
        if line.strip():
            pid, uuid, memory = [
                item.strip() for item in line.split(",", 2)
            ]
            rows.append(
                {
                    "pid": int(pid),
                    "gpu_uuid": uuid,
                    "used_memory_mib": int(memory),
                }
            )
    return rows


def query_gpu_uuids() -> dict[str, int]:
    output = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        uuid.strip(): int(index)
        for index, uuid in (
            line.split(",", 1) for line in output.splitlines() if line.strip()
        )
    }


def terminate(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_parent(args: argparse.Namespace) -> int:
    import torch

    protocol_sha256 = sha256_file(args.protocol)
    visible_gpus = torch.cuda.device_count()
    if visible_gpus != args.reserved_gpus_per_host:
        raise RuntimeError(
            "Visible GPU count does not match reserved topology: "
            f"{visible_gpus} != {args.reserved_gpus_per_host}"
        )
    if args.reserved_gpus_per_host < 2:
        raise ValueError("The paired RATD run requires at least two GPUs")
    args.output_root.mkdir(parents=True, exist_ok=True)
    logs = args.output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    verification = verify_assets(args.asset_manifest)
    save_json_atomic(
        args.output_root / "asset_verification.json", verification
    )
    reference_metadata_path = args.reference_map.with_suffix(".json")
    if args.reference_map.exists() and reference_metadata_path.exists():
        reference_metadata = json.loads(
            reference_metadata_path.read_text(encoding="utf-8")
        )
        if (
            reference_metadata["reference_map_sha256"]
            != sha256_file(args.reference_map)
        ):
            raise ValueError("RATD reference map hash drift")
    else:
        reference_metadata = build_reference_map(
            args.data,
            args.tcn_checkpoint,
            args.tcn_source_root,
            args.reference_map,
            args.seed,
        )
    if reference_metadata.get("query_counts") != {
        "train": 17_886,
        "validation": 2_463,
        "test": 5_093,
    }:
        raise ValueError("RATD reference map uses the wrong split origins")

    uuid_to_gpu = query_gpu_uuids()
    processes = {}
    log_files = {}
    observations = []
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--upstream-root",
        str(args.upstream_root),
        "--config",
        str(args.config),
        "--data",
        str(args.data),
        "--reference-map",
        str(args.reference_map),
        "--output-root",
        str(args.output_root),
        "--protocol",
        str(args.protocol),
        "--source-revision",
        args.source_revision,
        "--upstream-commit",
        args.upstream_commit,
        "--seed",
        str(args.seed),
        "--epochs",
        str(args.epochs),
        "--diffusion-steps",
        str(args.diffusion_steps),
        "--samples",
        str(args.samples),
    ]
    try:
        for gpu_id, method in enumerate(("ratd", "csdi")):
            log_file = (logs / f"{method}.log").open(
                "a", encoding="utf-8"
            )
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            environment["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(
                command + ["--method", method],
                cwd=args.upstream_root,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            processes[method] = process
            log_files[method] = log_file
        while any(process.poll() is None for process in processes.values()):
            observations.append(
                {
                    "recorded_unix": time.time(),
                    "processes": [
                        {
                            **record,
                            "physical_gpu": uuid_to_gpu.get(
                                record["gpu_uuid"]
                            ),
                        }
                        for record in query_gpu_processes()
                    ],
                }
            )
            time.sleep(args.poll_seconds)
    finally:
        for process in processes.values():
            if process.poll() is None:
                terminate(process)
        for log_file in log_files.values():
            log_file.close()

    topology = {
        "schema_version": 1,
        "instance_type": args.instance_type,
        "instance_count": 1,
        "reserved_gpus_per_host": args.reserved_gpus_per_host,
        "processes_per_host": 2,
        "worker_gpu_ids": [0, 1],
        "total_gpus": args.reserved_gpus_per_host,
        "world_size": 2,
        "inactive_reserved_gpus": args.reserved_gpus_per_host - 2,
        "inactive_reason": (
            "The paired experiment has exactly two independently trained "
            "systems, RATD and its no-reference CSDI base."
        ),
        "child_device": "cuda:0",
        "worker_pids": {
            method: process.pid for method, process in processes.items()
        },
        "worker_returncodes": {
            method: process.returncode
            for method, process in processes.items()
        },
        "observations": observations,
    }
    expected_bindings = {
        process.pid: gpu_id
        for gpu_id, process in enumerate(processes.values())
    }
    observed_bindings = {
        pid: sorted(
            {
                int(record["physical_gpu"])
                for observation in observations
                for record in observation["processes"]
                if record["pid"] == pid
                and record["physical_gpu"] is not None
            }
        )
        for pid in expected_bindings
    }
    topology_valid = all(
        observed_bindings[pid] == [expected_gpu]
        for pid, expected_gpu in expected_bindings.items()
    )
    topology["expected_bindings"] = expected_bindings
    topology["observed_bindings"] = observed_bindings
    topology["topology_valid"] = topology_valid
    topology_path = args.output_root / "startup_topology.json"
    save_json_atomic(topology_path, topology)
    results = {}
    failures = []
    for method in ("ratd", "csdi"):
        path = args.output_root / method / "result.json"
        if path.exists():
            results[method] = json.loads(path.read_text(encoding="utf-8"))
        else:
            failures.append(method)
    result_errors = []
    for method, result in results.items():
        metrics = result.get("metrics", {})
        if (
            result.get("status") != "completed"
            or result.get("test_origins") != 5_093
            or result.get("protocol_sha256") != protocol_sha256
            or set(metrics) != {"rmse", "mae"}
            or not all(np.isfinite(value) for value in metrics.values())
        ):
            result_errors.append(method)
    all_completed = (
        not failures
        and not result_errors
        and topology_valid
        and all(process.returncode == 0 for process in processes.values())
        and all(
            result.get("status") == "completed"
            for result in results.values()
        )
    )
    summary = {
        "schema_version": 1,
        "method": "RATD",
        "base_method": "CSDI",
        "expected_pairs": 1,
        "completed_pairs": int(all_completed),
        "failed_or_missing_pairs": 0 if all_completed else 1,
        "all_completed": all_completed,
        "protocol_sha256": protocol_sha256,
        "source_revision": args.source_revision,
        "upstream_commit": args.upstream_commit,
        "reference_metadata": reference_metadata,
        "topology_path": str(topology_path),
        "topology_valid": topology_valid,
        "results": results,
        "failures": failures + result_errors,
    }
    if all_completed:
        summary["pair"] = {
            "cell_id": "ratd/electricity/c96/h168",
            "base": results["csdi"]["metrics"],
            "retrieval": results["ratd"]["metrics"],
            "delta": {
                metric: results["ratd"]["metrics"][metric]
                - results["csdi"]["metrics"][metric]
                for metric in ("rmse", "mae")
            },
        }
    save_json_atomic(args.output_root / "summary.json", summary)
    return 0 if all_completed else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-manifest", type=Path)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--tcn-checkpoint", type=Path)
    parser.add_argument("--tcn-source-root", type=Path)
    parser.add_argument("--reference-map", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("docs/native_retrieval_baseline_protocol.json"),
    )
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--upstream-commit", required=True)
    parser.add_argument(
        "--instance-type", default="ml.g5.12xlarge"
    )
    parser.add_argument(
        "--reserved-gpus-per-host", type=int, default=4
    )
    parser.add_argument("--method", choices=("ratd", "csdi"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--diffusion-steps", type=int, default=50)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.method:
        try:
            train_and_evaluate(args, use_reference=args.method == "ratd")
        except BaseException as error:
            save_json_atomic(
                args.output_root / args.method / "status.json",
                {
                    "status": "failed",
                    "method": args.method.upper(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "finished_unix": time.time(),
                },
            )
            raise
        return 0
    required = (
        "asset_manifest",
        "tcn_checkpoint",
        "tcn_source_root",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise ValueError(f"Parent execution requires: {', '.join(missing)}")
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
