import gc
import hashlib
import importlib
import re
import zipfile
from pathlib import Path
from types import SimpleNamespace

import torch


_CHECKPOINT_PATTERN = re.compile(
    r"^TimeFuse Download/checkpoints/(96_(?:48|0)_96)/"
    r"(.+)_(TimeXer|TimeMixer|PAttn|iTransformer|TimesNet|PatchTST|"
    r"DLinear|FreTS|FEDformer)_dmodel(\d+)_epoch10/checkpoint[.]pth$"
)


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_release_catalog(archive_path, manifest):
    manifest_index = {
        (row["dataset"], row["model"], row["pred_len"]): row
        for row in manifest
    }
    checkpoints = {}
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            match = _CHECKPOINT_PATTERN.match(info.filename)
            if not match:
                continue
            setting, dataset, model, d_model = match.groups()
            cell = manifest_index.get((dataset, model, 96))
            if cell is None:
                continue
            d_model = int(d_model)
            manifest_d_model = int(cell["args"]["d_model"])
            config_compatible = d_model == manifest_d_model
            relative_path = str(
                Path(setting)
                / f"{dataset}_{model}_dmodel{d_model}_epoch10"
                / "checkpoint.pth"
            )
            checkpoints[cell["id"]] = {
                "archive_member": info.filename,
                "relative_path": relative_path,
                "file_size": info.file_size,
                "compressed_size": info.compress_size,
                "release_d_model": d_model,
                "manifest_d_model": manifest_d_model,
                "config_compatible": config_compatible,
                "usable": False,
                "reason": None if config_compatible else "d_model_mismatch",
            }
    return {
        "release_archive": str(archive_path),
        "release_sha256": sha256_file(archive_path),
        "checkpoint_count": len(checkpoints),
        "config_compatible_count": sum(
            row["config_compatible"] for row in checkpoints.values()
        ),
        "usable_count": 0,
        "checkpoints": dict(sorted(checkpoints.items())),
    }


def extract_config_compatible(archive_path, catalog, output_root):
    output_root = Path(output_root)
    extracted = 0
    with zipfile.ZipFile(archive_path) as archive:
        for record in catalog["checkpoints"].values():
            if not record["config_compatible"]:
                continue
            destination = output_root / record["relative_path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(record["archive_member"]) as source:
                with destination.open("wb") as output:
                    while chunk := source.read(8 * 1024 * 1024):
                        output.write(chunk)
            if destination.stat().st_size != record["file_size"]:
                raise IOError(f"Size mismatch after extracting {destination}")
            extracted += 1
    return extracted


def audit_extracted_checkpoints(catalog, manifest, checkpoint_root):
    manifest_index = {row["id"]: row for row in manifest}
    checkpoint_root = Path(checkpoint_root)
    audited = dict(catalog)
    audited["checkpoints"] = {
        cell_id: dict(record)
        for cell_id, record in catalog["checkpoints"].items()
    }
    usable = 0
    for cell_id, record in audited["checkpoints"].items():
        record["usable"] = False
        if not record["config_compatible"]:
            continue
        checkpoint_path = checkpoint_root / record["relative_path"]
        if not checkpoint_path.is_file():
            record["reason"] = "checkpoint_missing"
            continue
        try:
            cell = manifest_index[cell_id]
            module = importlib.import_module(f"models.{cell['model']}")
            model = module.Model(SimpleNamespace(**cell["args"])).float()
            state_dict = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(state_dict, strict=True)
            record["usable"] = True
            record["reason"] = None
            usable += 1
        except Exception as error:
            record["reason"] = f"{type(error).__name__}: {error}"
        finally:
            if "model" in locals():
                del model
            if "state_dict" in locals():
                del state_dict
            gc.collect()
    audited["usable_count"] = usable
    return audited
