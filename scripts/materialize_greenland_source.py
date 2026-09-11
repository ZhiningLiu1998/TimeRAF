"""Materialize a fixed Git commit without creating a local branch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
from pathlib import Path, PurePosixPath


def _save_json_atomic(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _safe_extract_tar(archive_path, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:*") as archive:
        members = archive.getmembers()
        for member in members:
            member_path = PurePosixPath(member.name)
            if (
                member_path.is_absolute()
                or ".." in member_path.parts
                or member.issym()
                or member.islnk()
                or member.isdev()
            ):
                raise ValueError(f"Unsafe archive member: {member.name}")
            resolved = (destination / member.name).resolve()
            if resolved != destination and destination not in resolved.parents:
                raise ValueError(
                    f"Archive member escapes destination: {member.name}"
                )
        archive.extractall(destination, members=members, filter="data")


def _below(root, value, field):
    root = Path(root).resolve()
    path = Path(value).resolve()
    if path == root or root not in path.parents:
        raise ValueError(f"{field} must live below artifact-root")
    return path


def _filesystem_type(path):
    return subprocess.run(
        ["findmnt", "-T", str(path), "-o", "FSTYPE", "-n"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _run_git(arguments, *, git_dir=None, cwd=None):
    command = ["git"]
    if git_dir is not None:
        command.extend(["--git-dir", str(git_dir)])
    command.extend(arguments)
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _refs(git_dir):
    payload = _run_git(
        ["for-each-ref", "--format=%(refname)"],
        git_dir=git_dir,
    )
    return [line for line in payload.splitlines() if line]


def _verify_workspace(destination, revision, bundle_sha256):
    destination = Path(destination)
    receipt_path = destination / "materialization_receipt.json"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Existing source workspace has no valid receipt: {destination}"
        ) from error
    git_dir = destination / "repository.git"
    source = destination / "source"
    if (
        receipt.get("revision") != revision
        or receipt.get("bundle_sha256") != bundle_sha256
        or not git_dir.is_dir()
        or not source.is_dir()
    ):
        raise ValueError("Existing source workspace identity drifted")
    actual = _run_git(
        ["rev-parse", "--verify", f"{revision}^{{commit}}"],
        git_dir=git_dir,
    )
    refs = _refs(git_dir)
    if actual != revision or refs:
        raise ValueError("Existing source workspace created Git refs or drifted")
    git_pointer = source / ".git"
    if git_pointer.read_text(encoding="utf-8") != "gitdir: ../repository.git\n":
        raise ValueError("Existing source workspace Git pointer drifted")
    return receipt


def materialize_source(
    bundle,
    bundle_ref,
    revision,
    artifact_root,
    destination,
    *,
    required_filesystem=None,
):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a full lowercase Git commit SHA")
    artifact_root = Path(artifact_root).resolve()
    if not artifact_root.is_dir():
        raise ValueError("artifact-root must be an existing directory")
    if required_filesystem and _filesystem_type(
        artifact_root
    ) != required_filesystem:
        raise ValueError(
            f"artifact-root must use {required_filesystem} durable storage"
        )
    bundle = _below(artifact_root, bundle, "bundle")
    destination = _below(artifact_root, destination, "destination")
    if not bundle.is_file() or bundle.is_symlink():
        raise ValueError("bundle must be a regular file")
    bundle_sha256 = _sha256(bundle)
    if destination.exists():
        return _verify_workspace(destination, revision, bundle_sha256)

    stage = destination.with_name(
        f".{destination.name}.stage-{os.getpid()}"
    )
    if stage.exists():
        raise ValueError(f"Source staging path already exists: {stage}")
    stage.mkdir(parents=True)
    git_dir = stage / "repository.git"
    source = stage / "source"
    archive = stage / "source.tar"
    try:
        _run_git(["init", "--bare", "--quiet", str(git_dir)])
        _run_git(
            ["fetch", "--quiet", str(bundle), bundle_ref],
            git_dir=git_dir,
        )
        fetched = _run_git(
            ["rev-parse", "--verify", "FETCH_HEAD^{commit}"],
            git_dir=git_dir,
        )
        if fetched != revision:
            raise ValueError(
                f"Bundle ref resolved to {fetched}, expected {revision}"
            )
        if _refs(git_dir):
            raise ValueError("Bundle fetch unexpectedly created Git refs")
        _run_git(
            [
                "archive",
                "--format=tar",
                "--output",
                str(archive),
                revision,
            ],
            git_dir=git_dir,
        )
        _safe_extract_tar(archive, source)
        archive.unlink()
        (source / ".git").write_text(
            "gitdir: ../repository.git\n",
            encoding="utf-8",
        )
        resolved = _run_git(
            ["rev-parse", "--verify", f"{revision}^{{commit}}"],
            cwd=source,
        )
        if resolved != revision:
            raise ValueError("Materialized source cannot resolve its revision")
        receipt = {
            "schema_version": 1,
            "revision": revision,
            "bundle": str(bundle),
            "bundle_ref": bundle_ref,
            "bundle_sha256": bundle_sha256,
            "artifact_root": str(artifact_root),
            "workspace": str(destination),
            "source": str(destination / "source"),
            "git_dir": str(destination / "repository.git"),
            "git_refs": [],
            "branch_created": False,
        }
        _save_json_atomic(receipt, stage / "materialization_receipt.json")
        stage.replace(destination)
        return receipt
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Materialize a fixed Greenland source commit without refs"
    )
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--bundle-ref", default="HEAD")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--required-filesystem", default="nfs4")
    args = parser.parse_args(argv)

    receipt = materialize_source(
        args.bundle,
        args.bundle_ref,
        args.revision,
        args.artifact_root,
        args.destination,
        required_filesystem=args.required_filesystem,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
