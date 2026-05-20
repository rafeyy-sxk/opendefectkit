"""Tests for the opendefectkit.benchmark module.

These tests do NOT require ultralytics or torch.
"""

import os
import pathlib

import pytest

from opendefectkit.benchmark import (
    BenchmarkResult,
    ClassMetrics,
    DefectBenchmark,
    IndustryReportGenerator,
    ModelComparison,
)
from opendefectkit.benchmark.metrics import _compute_edge_readiness_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_result() -> BenchmarkResult:
    """Build a fake BenchmarkResult for testing."""
    return BenchmarkResult(
        model_path="fake/model.pt",
        dataset_path="fake/dataset",
        map50=0.82,
        map50_95=0.61,
        per_class=[
            ClassMetrics("crack", precision=0.88, recall=0.79, f1=0.83, ap50=0.84),
            ClassMetrics("rust", precision=0.71, recall=0.65, f1=0.68, ap50=0.72),
        ],
        false_positive_rate=0.12,
        false_negative_rate=0.21,
        fps_cpu=8.5,
        fps_gpu=None,
        model_size_mb=6.2,
        edge_readiness_score=0.0,  # computed separately in tests
    )


# ---------------------------------------------------------------------------
# Part A — BenchmarkResult dataclass
# ---------------------------------------------------------------------------


def test_benchmark_result_fields() -> None:
    """BenchmarkResult can be instantiated and all fields are accessible."""
    r = make_result()
    assert r.model_path == "fake/model.pt"
    assert r.dataset_path == "fake/dataset"
    assert r.map50 == pytest.approx(0.82)
    assert r.map50_95 == pytest.approx(0.61)
    assert len(r.per_class) == 2
    assert r.per_class[0].label == "crack"
    assert r.false_positive_rate == pytest.approx(0.12)
    assert r.false_negative_rate == pytest.approx(0.21)
    assert r.fps_cpu == pytest.approx(8.5)
    assert r.fps_gpu is None
    assert r.model_size_mb == pytest.approx(6.2)


# ---------------------------------------------------------------------------
# Edge readiness scoring
# ---------------------------------------------------------------------------


def test_edge_readiness_score_small_fast_accurate() -> None:
    """Small (6 MB), fast (8.5 fps), accurate (0.82 mAP) model scores > 50."""
    score = _compute_edge_readiness_score(
        model_size_mb=6.2,
        fps_cpu=8.5,
        map50=0.82,
    )
    # size<10 → 30, fps>5 → 20, map50>0.8 → 40  = 90
    assert score > 50
    assert score == pytest.approx(90.0)


def test_edge_readiness_score_large_slow() -> None:
    """Large (200 MB), slow (2 fps), poor (0.5 mAP) model scores 0."""
    score = _compute_edge_readiness_score(
        model_size_mb=200.0,
        fps_cpu=2.0,
        map50=0.5,
    )
    assert score == pytest.approx(0.0)


def test_edge_readiness_score_medium_model() -> None:
    """Model between 10 and 50 MB gets +20 for size."""
    score = _compute_edge_readiness_score(
        model_size_mb=30.0,
        fps_cpu=None,
        map50=0.0,
    )
    assert score == pytest.approx(20.0)


def test_edge_readiness_score_fast_fps() -> None:
    """fps > 10 gives +30; only top tier applies."""
    score = _compute_edge_readiness_score(
        model_size_mb=200.0,
        fps_cpu=15.0,
        map50=0.0,
    )
    assert score == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# HTML report from BenchmarkResult
# ---------------------------------------------------------------------------


def test_save_html_creates_file(tmp_path: pathlib.Path) -> None:
    """save_html writes an HTML file that exists and contains 'mAP'."""
    r = make_result()
    out = str(tmp_path / "report.html")
    r.save_html(out)
    assert os.path.isfile(out)
    with open(out, encoding="utf-8") as fh:
        content = fh.read()
    assert "mAP" in content


def test_save_html_contains_model_path(tmp_path: pathlib.Path) -> None:
    """The HTML report references the model path."""
    r = make_result()
    out = str(tmp_path / "report.html")
    r.save_html(out)
    with open(out, encoding="utf-8") as fh:
        content = fh.read()
    assert "fake/model.pt" in content


def test_save_html_contains_per_class(tmp_path: pathlib.Path) -> None:
    """The HTML report includes per-class labels."""
    r = make_result()
    out = str(tmp_path / "report.html")
    r.save_html(out)
    with open(out, encoding="utf-8") as fh:
        content = fh.read()
    assert "crack" in content
    assert "rust" in content


# ---------------------------------------------------------------------------
# IndustryReportGenerator — HTML
# ---------------------------------------------------------------------------


def test_industry_report_generate_html(tmp_path: pathlib.Path) -> None:
    """generate_html creates an HTML file at the requested path."""
    r = make_result()
    out = str(tmp_path / "industry_report.html")
    gen = IndustryReportGenerator(r, "ACME", "Line 1")
    gen.generate_html(out)
    assert os.path.isfile(out)


def test_industry_report_html_contains_company(tmp_path: pathlib.Path) -> None:
    """The generated HTML includes the company name."""
    r = make_result()
    out = str(tmp_path / "industry_report.html")
    gen = IndustryReportGenerator(r, "ACME", "Line 1")
    gen.generate_html(out)
    with open(out, encoding="utf-8") as fh:
        content = fh.read()
    assert "ACME" in content


def test_industry_report_html_contains_inspection_line(tmp_path: pathlib.Path) -> None:
    """The generated HTML includes the inspection line."""
    r = make_result()
    out = str(tmp_path / "industry_report.html")
    gen = IndustryReportGenerator(r, "ACME", "Line 1")
    gen.generate_html(out)
    with open(out, encoding="utf-8") as fh:
        content = fh.read()
    assert "Line 1" in content


def test_industry_report_recommendation_high_fnr(tmp_path: pathlib.Path) -> None:
    """FNR > 0.2 triggers HIGH RISK recommendation."""
    r = make_result()
    r.false_negative_rate = 0.3  # above threshold
    out = str(tmp_path / "high_fnr.html")
    gen = IndustryReportGenerator(r, "ACME", "Line 1")
    gen.generate_html(out)
    with open(out, encoding="utf-8") as fh:
        content = fh.read()
    assert "RISK" in content


def test_industry_report_recommendation_production_ready(tmp_path: pathlib.Path) -> None:
    """mAP50 > 0.85 and low rates → PRODUCTION READY."""
    r = make_result()
    r.map50 = 0.90
    r.false_negative_rate = 0.05
    r.false_positive_rate = 0.05
    out = str(tmp_path / "prod_ready.html")
    gen = IndustryReportGenerator(r, "ACME", "Line 1")
    gen.generate_html(out)
    with open(out, encoding="utf-8") as fh:
        content = fh.read()
    assert "PRODUCTION READY" in content


def test_industry_report_recommendation_high_fpr(tmp_path: pathlib.Path) -> None:
    """FPR > 0.3 (and FNR <= 0.2) triggers WARNING."""
    r = make_result()
    r.false_negative_rate = 0.10  # below threshold
    r.false_positive_rate = 0.40  # above threshold
    out = str(tmp_path / "high_fpr.html")
    gen = IndustryReportGenerator(r, "ACME", "Line 1")
    gen.generate_html(out)
    with open(out, encoding="utf-8") as fh:
        content = fh.read()
    assert "WARNING" in content


# ---------------------------------------------------------------------------
# DefectBenchmark / ModelComparison — ImportError when ultralytics missing
# ---------------------------------------------------------------------------


def test_defect_benchmark_import_error() -> None:
    """DefectBenchmark.run() raises ImportError when ultralytics is not installed."""
    pytest.importorskip(
        "ultralytics",
        reason="ultralytics IS installed — skipping import-error test",
        # Invert: only run this test when ultralytics is NOT available.
    )
    # If importorskip didn't skip, ultralytics is available — skip explicitly.
    pytest.skip("ultralytics is installed; cannot test ImportError path")


def test_defect_benchmark_import_error_real() -> None:
    """Raise ImportError when ultralytics is absent (mocked via monkeypatching)."""
    import sys

    # Temporarily hide ultralytics from the import system
    original = sys.modules.get("ultralytics", None)
    sys.modules["ultralytics"] = None  # type: ignore[assignment]
    try:
        bench = DefectBenchmark("x", "y")
        with pytest.raises(ImportError, match="ultralytics"):
            bench.run()
    finally:
        if original is None:
            del sys.modules["ultralytics"]
        else:
            sys.modules["ultralytics"] = original


def test_model_comparison_run_import_error() -> None:
    """ModelComparison.run() raises ImportError when ultralytics is absent."""
    import sys

    original = sys.modules.get("ultralytics", None)
    sys.modules["ultralytics"] = None  # type: ignore[assignment]
    try:
        comp = ModelComparison(models={"m1": "x.pt"}, test_dataset="ds")
        with pytest.raises(ImportError, match="ultralytics"):
            comp.run()
    finally:
        if original is None:
            del sys.modules["ultralytics"]
        else:
            sys.modules["ultralytics"] = original


# ---------------------------------------------------------------------------
# ClassMetrics
# ---------------------------------------------------------------------------


def test_class_metrics_fields() -> None:
    """ClassMetrics stores all fields correctly."""
    cm = ClassMetrics(label="scratch", precision=0.9, recall=0.85, f1=0.87, ap50=0.88)
    assert cm.label == "scratch"
    assert cm.precision == pytest.approx(0.9)
    assert cm.recall == pytest.approx(0.85)
    assert cm.f1 == pytest.approx(0.87)
    assert cm.ap50 == pytest.approx(0.88)
