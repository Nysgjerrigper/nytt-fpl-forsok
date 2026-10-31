"""Exercise CI's actual shell filter against real, isolated Git histories."""
import os
from pathlib import Path
import subprocess

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / ".github/scripts/ci-changes.sh"
ZERO_SHA = "0" * 40


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.PIPE).strip()


def _commit(root: Path, changes: dict[str, str | None]) -> str:
    for name, content in changes.items():
        path = root / name
        if content is None:
            path.unlink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        _git(root, "add", "--", name)
    _git(root, "-c", "user.name=CI", "-c", "user.email=ci@example.invalid", "commit", "-m", "fixture")
    return _git(root, "rev-parse", "HEAD")


@pytest.fixture
def history(tmp_path):
    _git(tmp_path, "init", "-b", "main")
    base = _commit(tmp_path, {"README.md": "original\n", "code.py": "x = 1\n", "requirements.txt": "numpy\n"})
    _git(tmp_path, "update-ref", "refs/remotes/origin/main", base)
    _git(tmp_path, "checkout", "-b", "feature/test")
    return tmp_path, base


def _filter(root: Path, base: str, *, ref: str = "refs/heads/feature/test") -> subprocess.CompletedProcess:
    output = root / "github-output"
    output.unlink(missing_ok=True)
    env = {**os.environ, "BASE_SHA": base, "HEAD_SHA": _git(root, "rev-parse", "HEAD"),
           "GITHUB_OUTPUT": str(output), "GITHUB_REF": ref, "DEFAULT_BRANCH": "main"}
    return subprocess.run(["bash", str(SCRIPT)], cwd=root, env=env, text=True, capture_output=True)


@pytest.mark.parametrize("new_branch,changes,expected", [
    (True, {"README.md": "updated\n"}, False),
    (False, {"docs/note.md": "note\n"}, False),
    (True, {"code.py": "x = 2\n"}, True),
    (True, {"requirements.txt": "pandas\n"}, True),
    (True, {"README.md": "updated\n", "code.py": "x = 2\n"}, True),
    (True, {"code.py": None}, True),
    (False, {}, False),
    (True, {}, False),
])
def test_classifies_real_event_diffs(history, new_branch, changes, expected):
    root, base = history
    if changes:
        _commit(root, changes)
    result = _filter(root, ZERO_SHA if new_branch else base)
    assert result.returncode == 0, result.stderr
    assert (root / "github-output").read_text() == f"run_tests={str(expected).lower()}\n"


def test_initial_default_branch_keeps_tests_enabled(history):
    root, _ = history
    result = _filter(root, ZERO_SHA, ref="refs/heads/main")
    assert result.returncode == 0
    assert (root / "github-output").read_text() == "run_tests=true\n"


def test_missing_comparison_base_fails_closed(history):
    root, _ = history
    _git(root, "update-ref", "-d", "refs/remotes/origin/main")
    result = _filter(root, ZERO_SHA)
    assert result.returncode != 0
    assert not (root / "github-output").exists()


def test_code_renamed_to_markdown_still_runs_tests(history):
    root, _ = history
    _git(root, "mv", "code.py", "renamed.md")
    _commit(root, {"renamed.md": "x = 1\n"})
    result = _filter(root, ZERO_SHA)
    assert result.returncode == 0, result.stderr
    assert (root / "github-output").read_text() == "run_tests=true\n"
