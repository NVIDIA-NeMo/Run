"""Regression test for cherry-pick workflow trigger."""
import pathlib

CHERRY_YML = pathlib.Path(".github/workflows/cherry-pick-release-commit.yml")


def test_cherry_pick_triggers_on_release_branches():
    content = CHERRY_YML.read_text()
    assert "branches:\n      - main" not in content, (
        "cherry-pick workflow should not trigger only on main"
    )
    assert "[rv]" in content, (
        "cherry-pick workflow should trigger on release branches"
    )
