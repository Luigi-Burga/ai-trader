from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.phase1
def test_project_root_exists():
    assert PROJECT_ROOT.exists()
    assert (PROJECT_ROOT / "app").is_dir()


@pytest.mark.phase1
def test_production_entrypoint_exists():
    assert (PROJECT_ROOT / "app" / "main.py").is_file()


@pytest.mark.phase1
def test_python_package_is_present():
    assert (PROJECT_ROOT / "app" / "__init__.py").is_file()


@pytest.mark.phase1
def test_runtime_requirements_exist():
    assert (PROJECT_ROOT / "requirements.txt").is_file()


@pytest.mark.phase1
def test_core_production_areas_are_present():
    expected_dirs = [
        "app/ai",
        "app/alerts",
        "app/config",
        "app/data",
        "app/portfolio",
        "app/scanners",
        "app/strategies",
        "app/utils",
    ]

    missing = [
        item for item in expected_dirs
        if not (PROJECT_ROOT / item).is_dir()
    ]

    assert not missing, f"Core production directories missing: {missing}"


@pytest.mark.phase1
def test_at_least_one_versioned_ai_engine_exists():
    ai_dir = PROJECT_ROOT / "app" / "ai"

    candidates = list(ai_dir.glob("characteristic_curve*.py"))
    assert candidates, "No Characteristic Curve module found under app/ai/."


@pytest.mark.phase1
def test_test_directory_is_outside_production():
    tests_dir = PROJECT_ROOT / "tests"
    app_dir = PROJECT_ROOT / "app"

    assert tests_dir != app_dir
    assert tests_dir.parent == PROJECT_ROOT
