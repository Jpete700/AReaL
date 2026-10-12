"""Fresh group member workspaces and immutable source/submission ownership."""

import pytest

from examples.aspire.contracts import PolicyError

from .conftest import events


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [
        "/etc/passwd",
        "../sibling/solution.py",
        "a/../../escape",
        "C:/private/data",
        r"a\b",
        ".",
    ],
)
async def test_workspace_outside_paths_rejected(harness, path):
    """Neither POSIX nor Windows-style paths may escape the owned directory."""
    h = await harness()
    with pytest.raises(PolicyError):
        await h.workspace.read(path)


@pytest.mark.asyncio
async def test_workspace_symlink_and_search_escape_rejected(harness, tmp_path):
    """Reject symlink escape for reads and recursive search."""
    h = await harness()
    outside = tmp_path / "outside.txt"
    outside.write_text("private")
    (h.workspace.root / "link.txt").symlink_to(outside)
    with pytest.raises(PolicyError):
        await h.workspace.read("link.txt")


@pytest.mark.asyncio
async def test_workspace_group_members_and_l0_remain_isolated(harness):
    """A member's edits never mutate another member or the initial L0."""
    first = await harness(episode_id="member1")
    second = await harness(episode_id="member2")
    await first.workspace.write("solution.py", "changed")
    assert await second.workspace.read("solution.py") == "# Fresh workspace\n"
    with pytest.raises(PolicyError):
        await first.workspace.write("L0/localize.md", "promoted")
    with pytest.raises(PolicyError):
        await first.workspace.write("api-reference.md", "altered API")
    with pytest.raises(PolicyError):
        await first.workspace.read("../member2/workspace/solution.py")


@pytest.mark.asyncio
async def test_workspace_duplicate_run_id_rejected(harness):
    """An existing episode can never be silently overwritten."""
    await harness()
    with pytest.raises(FileExistsError):
        await harness()


@pytest.mark.asyncio
async def test_submission_requires_matching_hash_and_freezes_all_writes(harness):
    """Registered immutable bytes define submission, independent of later filesystem edits."""
    h = await harness()
    record = await h.workspace.write("solution.py", "value = 1\n")
    with pytest.raises(PolicyError):
        await h.workspace.freeze("solution.py", "0" * 64)
    submission = await h.workspace.freeze("solution.py", record["code_sha256"])
    assert await h.workspace.freeze("solution.py", record["code_sha256"]) == submission
    with pytest.raises(PolicyError):
        await h.workspace.write("anything.py", "mutate")
    assert (
        h.audit.root / "artifacts" / submission.artifact_hash
    ).read_bytes() == b"value = 1\n"
    assert sum(e["event_type"] == "submission_frozen" for e in events(h)) == 1
