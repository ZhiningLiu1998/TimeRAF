import os
import subprocess
from pathlib import Path

from scripts import materialize_greenland_source


def _git(git_dir, arguments, *, payload=None, env=None):
    return subprocess.run(
        ["git", "--git-dir", str(git_dir), *arguments],
        input=payload,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    ).stdout.strip()


def _bundle_fixture(tmp_path):
    repository = tmp_path / "fixture.git"
    subprocess.run(
        ["git", "init", "--bare", "--quiet", str(repository)],
        check=True,
    )
    blob = _git(
        repository,
        ["hash-object", "-w", "--stdin"],
        payload="fixed source\n",
    )
    tree = _git(
        repository,
        ["mktree"],
        payload=f"100644 blob {blob}\tREADME.md\n",
    )
    environment = {
        **os.environ,
        "GIT_AUTHOR_NAME": "TimeRAF Test",
        "GIT_AUTHOR_EMAIL": "timeraf-test@example.com",
        "GIT_COMMITTER_NAME": "TimeRAF Test",
        "GIT_COMMITTER_EMAIL": "timeraf-test@example.com",
    }
    revision = _git(
        repository,
        ["commit-tree", tree],
        payload="fixture\n",
        env=environment,
    )
    _git(
        repository,
        ["update-ref", "refs/tags/fixture", revision],
    )
    bundle = tmp_path / "fixture.bundle"
    _git(
        repository,
        ["bundle", "create", str(bundle), "refs/tags/fixture"],
    )
    return bundle, revision


def test_materialized_greenland_source_has_no_branch_or_tag_refs(tmp_path):
    artifact_root = tmp_path / "efs-root"
    artifact_root.mkdir()
    bundle, revision = _bundle_fixture(artifact_root)
    destination = artifact_root / "operations" / f"source-{revision}"

    receipt = materialize_greenland_source.materialize_source(
        bundle,
        "refs/tags/fixture",
        revision,
        artifact_root,
        destination,
    )

    source = destination / "source"
    git_dir = destination / "repository.git"
    assert receipt["branch_created"] is False
    assert receipt["git_refs"] == []
    assert (source / "README.md").read_text() == "fixed source\n"
    assert _git(
        git_dir,
        ["for-each-ref", "--format=%(refname)"],
    ) == ""
    assert (
        subprocess.run(
            [
                "git",
                "-C",
                str(source),
                "rev-parse",
                "--verify",
                f"{revision}^{{commit}}",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        == revision
    )

    repeated = materialize_greenland_source.materialize_source(
        bundle,
        "refs/tags/fixture",
        revision,
        artifact_root,
        destination,
    )
    assert repeated == receipt
