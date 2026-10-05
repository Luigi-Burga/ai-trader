from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.phase1
def test_phase1_artifacts_are_not_inside_app():
    forbidden = [
        PROJECT_ROOT / "app" / "pytest.ini",
        PROJECT_ROOT / "app" / "test_runner.ps1",
        PROJECT_ROOT / "app" / "requirements-test.txt",
    ]

    existing = [str(p) for p in forbidden if p.exists()]
    assert not existing, (
        "Phase 1 infrastructure must remain outside production app/: "
        f"{existing}"
    )


@pytest.mark.phase1
def test_no_phase1_runner_inside_app():
    app_files = list((PROJECT_ROOT / "app").rglob("test_runner*.ps1"))
    assert not app_files, f"Test runners found inside app/: {app_files}"
