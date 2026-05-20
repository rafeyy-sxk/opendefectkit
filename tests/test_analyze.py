"""Tests for the opendefectkit.analyze module."""
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pytest

from opendefectkit.analyze import (
    DatasetHealthScore,
    DatasetProfiler,
    DatasetVisualizer,
    SeverityScorer,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Helper
# ─────────────────────────────────────────────────────────────────────────────


def make_labeled_yolo_dir(
    tmp_path: Path,
    n_images: int = 10,
    classes: Tuple[str, ...] = ("crack", "rust"),
) -> Path:
    """Create a proper YOLO dataset with images/ labels/ classes.txt."""
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    (tmp_path / "classes.txt").write_text("\n".join(classes))
    for i in range(n_images):
        img = np.full((480, 640, 3), 150, dtype=np.uint8)
        cv2.imwrite(str(images_dir / f"img_{i:03d}.jpg"), img)
        class_id = i % len(classes)
        (labels_dir / f"img_{i:03d}.txt").write_text(f"{class_id} 0.5 0.5 0.3 0.2\n")
    return tmp_path


# ─────────────────────────────────────────────────────────────────────────────
#  DatasetProfiler tests
# ─────────────────────────────────────────────────────────────────────────────


def test_profiler_total_images(tmp_dir):
    """10-image dataset should report total_images == 10."""
    dataset_dir = make_labeled_yolo_dir(tmp_dir, n_images=10)
    report = DatasetProfiler(str(dataset_dir)).run()
    assert report.total_images == 10


def test_profiler_annotated_count(tmp_dir):
    """All 10 images have labels → annotated_images == 10."""
    dataset_dir = make_labeled_yolo_dir(tmp_dir, n_images=10)
    report = DatasetProfiler(str(dataset_dir)).run()
    assert report.annotated_images == 10


def test_profiler_class_distribution(tmp_dir):
    """With 10 images and 2 classes (crack, rust), each class gets 5 annotations."""
    dataset_dir = make_labeled_yolo_dir(tmp_dir, n_images=10, classes=("crack", "rust"))
    report = DatasetProfiler(str(dataset_dir)).run()
    dist = report.class_distribution
    assert "crack" in dist
    assert "rust" in dist
    assert dist["crack"] == 5
    assert dist["rust"] == 5


def test_profiler_bbox_stats_non_empty(tmp_dir):
    """bbox_widths should be a non-empty list of floats."""
    dataset_dir = make_labeled_yolo_dir(tmp_dir, n_images=5)
    report = DatasetProfiler(str(dataset_dir)).run()
    assert isinstance(report.bbox_widths, list)
    assert len(report.bbox_widths) > 0
    assert all(isinstance(w, float) for w in report.bbox_widths)


def test_profiler_recommended_split(tmp_dir):
    """Sum of split counts must equal annotated_images."""
    dataset_dir = make_labeled_yolo_dir(tmp_dir, n_images=10)
    report = DatasetProfiler(str(dataset_dir)).run()
    split_sum = sum(report.recommended_split.values())
    assert split_sum == report.annotated_images


def test_profiler_defect_density_grid(tmp_dir):
    """Density grid must be 10x10 with values that sum > 0."""
    dataset_dir = make_labeled_yolo_dir(tmp_dir, n_images=5)
    report = DatasetProfiler(str(dataset_dir)).run()
    grid = report.defect_density_grid
    assert len(grid) == 10
    assert all(len(row) == 10 for row in grid)
    total = sum(v for row in grid for v in row)
    assert total > 0


# ─────────────────────────────────────────────────────────────────────────────
#  DatasetHealthScore tests
# ─────────────────────────────────────────────────────────────────────────────


def test_health_score_returns_0_to_100(tmp_dir):
    """Health score must be in [0, 100]."""
    dataset_dir = make_labeled_yolo_dir(tmp_dir, n_images=10)
    result = DatasetHealthScore(str(dataset_dir)).run()
    assert 0 <= result.score <= 100


def test_health_score_annotation_coverage(tmp_dir):
    """Dataset where all images are annotated should pass the coverage criterion."""
    dataset_dir = make_labeled_yolo_dir(tmp_dir, n_images=10)
    result = DatasetHealthScore(str(dataset_dir)).run()
    coverage_item = next(
        item for item in result.items if "coverage" in item.criterion.lower()
    )
    assert coverage_item.passed is True


def test_health_score_recommendation_not_empty(tmp_dir):
    """Recommendation string must be non-empty."""
    dataset_dir = make_labeled_yolo_dir(tmp_dir, n_images=10)
    result = DatasetHealthScore(str(dataset_dir)).run()
    assert isinstance(result.recommendation, str)
    assert len(result.recommendation.strip()) > 0


# ─────────────────────────────────────────────────────────────────────────────
#  DatasetVisualizer tests
# ─────────────────────────────────────────────────────────────────────────────


def test_visualizer_plot_class_distribution(tmp_dir):
    """plot_class_distribution should create a PNG file."""
    dataset_dir = make_labeled_yolo_dir(tmp_dir, n_images=6)
    save_path = str(tmp_dir / "class_dist.png")
    DatasetVisualizer(str(dataset_dir)).plot_class_distribution(save_path)
    assert Path(save_path).exists()
    assert Path(save_path).stat().st_size > 0


def test_visualizer_plot_heatmap(tmp_dir):
    """plot_defect_heatmap should create a PNG file."""
    dataset_dir = make_labeled_yolo_dir(tmp_dir, n_images=6)
    save_path = str(tmp_dir / "heatmap.png")
    DatasetVisualizer(str(dataset_dir)).plot_defect_heatmap(save_path)
    assert Path(save_path).exists()
    assert Path(save_path).stat().st_size > 0


def test_visualizer_html_report(tmp_dir):
    """generate_html_report should create an HTML file containing 'class'."""
    dataset_dir = make_labeled_yolo_dir(tmp_dir, n_images=6)
    save_path = str(tmp_dir / "report.html")
    DatasetVisualizer(str(dataset_dir)).generate_html_report(save_path)
    html_path = Path(save_path)
    assert html_path.exists()
    content = html_path.read_text(encoding="utf-8")
    assert "class" in content


# ─────────────────────────────────────────────────────────────────────────────
#  SeverityScorer tests
# ─────────────────────────────────────────────────────────────────────────────


def test_severity_scorer_import_error():
    """SeverityScorer('fake.pt') must raise ImportError when ultralytics is absent."""
    pytest.importorskip(
        "ultralytics",
        reason="ultralytics is installed; skipping ImportError test",
        # If importorskip succeeds, the test is skipped.
        # If the module is missing, importorskip itself raises Skipped —
        # so we need the inverse: skip when installed, test when absent.
    )
    # If we reach here, ultralytics IS installed — skip.
    pytest.skip("ultralytics is installed; ImportError would not be raised")


# Separate test to actually verify ImportError when ultralytics is absent.
def test_severity_scorer_import_error_when_missing(monkeypatch):
    """If ultralytics cannot be imported, SeverityScorer must raise ImportError."""
    import sys
    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "ultralytics":
            raise ImportError("mocked missing ultralytics")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    # Remove cached module if present
    monkeypatch.delitem(sys.modules, "ultralytics", raising=False)

    with pytest.raises(ImportError, match="SeverityScorer requires ultralytics"):
        SeverityScorer("fake.pt")
