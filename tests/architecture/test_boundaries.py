"""Enforce package boundaries independently of heavyweight model dependencies."""

import importlib.util
from pathlib import Path


def test_architecture():
    path = Path(__file__).resolve().parents[2] / "scripts" / "check_architecture.py"
    spec = importlib.util.spec_from_file_location("architecture_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.violations() == []
