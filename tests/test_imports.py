"""Lightweight import checks (PyTorch required for solver import)."""
import importlib.util


def test_solver_source_exists():
    assert importlib.util.find_spec('pd_acoustics.simulate') is not None


def test_visualization_source_exists():
    assert importlib.util.find_spec('pd_acoustics.visualize') is not None
