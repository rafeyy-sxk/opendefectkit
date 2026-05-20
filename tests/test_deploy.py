"""Tests for opendefectkit.deploy — no onnx/ultralytics/torch required."""
import tempfile
from pathlib import Path

import pytest

from opendefectkit.deploy import (
    DEVICE_PROFILES,
    DeviceProfile,
    EdgeOptimizer,
    ONNXExporter,
    OptimizationResult,
)


# ---------------------------------------------------------------------------
# DeviceProfile / DEVICE_PROFILES
# ---------------------------------------------------------------------------


def test_device_profiles_exist():
    """DEVICE_PROFILES contains all 5 expected device keys."""
    expected = {"jetson_nano", "jetson_orin", "raspberry_pi_4", "intel_nuc", "generic_x86"}
    assert expected == set(DEVICE_PROFILES.keys())


def test_device_profile_fields():
    """jetson_nano profile has correct target_fps and supports_gpu."""
    profile = DEVICE_PROFILES["jetson_nano"]
    assert isinstance(profile, DeviceProfile)
    assert profile.target_fps == 30.0
    assert profile.supports_gpu is True


# ---------------------------------------------------------------------------
# EdgeOptimizer.profile_device
# ---------------------------------------------------------------------------


def test_profile_device_valid():
    """profile_device with a known device returns self for chaining."""
    optimizer = EdgeOptimizer("x.onnx")
    result = optimizer.profile_device("jetson_nano")
    assert result is optimizer


def test_profile_device_invalid_raises():
    """profile_device with an unknown device raises ValueError."""
    optimizer = EdgeOptimizer("x.onnx")
    with pytest.raises(ValueError, match="Unknown device"):
        optimizer.profile_device("invalid_device")


# ---------------------------------------------------------------------------
# EdgeOptimizer.optimize
# ---------------------------------------------------------------------------


def test_optimize_no_onnx_file():
    """optimize() on a non-existent file returns size=0 and non-empty recommendations."""
    optimizer = EdgeOptimizer("nonexistent_model.onnx")
    optimizer.profile_device("jetson_nano")
    result = optimizer.optimize()
    assert isinstance(result, OptimizationResult)
    assert result.original_size_mb == 0.0
    assert len(result.recommendations) > 0


def test_optimize_with_profile():
    """raspberry_pi_4 profile includes 'quantization' in recommendations."""
    optimizer = EdgeOptimizer("nonexistent_model.onnx")
    optimizer.profile_device("raspberry_pi_4")
    result = optimizer.optimize()
    combined = " ".join(result.recommendations).lower()
    assert "quantization" in combined


# ---------------------------------------------------------------------------
# EdgeOptimizer.generate_deployment_package
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def test_generate_deployment_package_creates_dir(tmp_dir):
    """generate_deployment_package creates the output directory."""
    pkg_dir = tmp_dir / "pkg"
    optimizer = EdgeOptimizer("x.onnx")
    optimizer.generate_deployment_package(str(pkg_dir))
    assert pkg_dir.exists() and pkg_dir.is_dir()


def test_generate_deployment_package_creates_zip(tmp_dir):
    """deployment_package.zip exists after generate_deployment_package."""
    pkg_dir = tmp_dir / "pkg"
    optimizer = EdgeOptimizer("x.onnx")
    optimizer.generate_deployment_package(str(pkg_dir))
    assert (pkg_dir / "deployment_package.zip").exists()


def test_generate_deployment_package_inference_script(tmp_dir):
    """run_inference.py is created in the output directory."""
    pkg_dir = tmp_dir / "pkg"
    optimizer = EdgeOptimizer("x.onnx")
    optimizer.generate_deployment_package(str(pkg_dir), include_inference_script=True)
    assert (pkg_dir / "run_inference.py").exists()


def test_generate_deployment_package_requirements(tmp_dir):
    """requirements.txt exists and contains 'onnxruntime'."""
    pkg_dir = tmp_dir / "pkg"
    optimizer = EdgeOptimizer("x.onnx")
    optimizer.generate_deployment_package(str(pkg_dir), include_requirements=True)
    req_path = pkg_dir / "requirements.txt"
    assert req_path.exists()
    assert "onnxruntime" in req_path.read_text()


# ---------------------------------------------------------------------------
# ONNXExporter — import guard
# ---------------------------------------------------------------------------


def test_onnx_exporter_import_error(monkeypatch):
    """ONNXExporter.export raises ImportError when ultralytics is not installed."""
    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "ultralytics":
            raise ImportError("No module named 'ultralytics'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    exporter = ONNXExporter("x.pt")
    with pytest.raises(ImportError, match="ultralytics"):
        exporter.export("out.onnx")
