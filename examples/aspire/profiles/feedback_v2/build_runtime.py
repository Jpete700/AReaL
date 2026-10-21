"""Reconstruct the tested stage2 profile without tracking a duplicate repository."""

import json
import shutil
from pathlib import Path


def build_runtime(project: Path, destination: Path) -> None:
    repository = project.parents[3]
    assert (repository / "areal").is_dir()
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("examples/aspire", "tests/examples/aspire"):
        shutil.copytree(
            repository / name,
            destination / name,
            ignore=shutil.ignore_patterns(
                "profiles", "__pycache__", "*.pyc", ".pytest_cache"
            ),
        )
    manifest = json.loads((project / "overlay_manifest.json").read_text())
    for name in manifest["omit_from_materialized_profile"]:
        path = destination / name
        assert path.resolve().is_relative_to(destination.resolve())
        path.unlink()
    for name in manifest["overrides"]:
        source = project / "overrides" / name
        target = destination / name
        assert target.resolve().is_relative_to(destination.resolve())
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


if __name__ == "__main__":
    profile = Path(__file__).resolve().parent
    build_runtime(profile, profile / "runtime")
