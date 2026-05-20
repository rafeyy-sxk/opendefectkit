"""Tests for the opendefectkit.validate module."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from opendefectkit.validate import (
    AnnotationFixer,
    AnnotationValidator,
    IssueType,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_yolo_dataset(
    tmp_path: Path,
    images_with_labels: List[Tuple[str, int, int, List[str]]],
    classes: List[str] = None,
) -> Path:
    """Create a synthetic YOLO dataset on disk.

    images_with_labels: list of (filename, width, height, label_lines)
    """
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    if classes is not None:
        (labels_dir / "classes.txt").write_text("\n".join(classes))

    for fname, w, h, lines in images_with_labels:
        img = np.full((h, w, 3), 150, dtype=np.uint8)
        cv2.imwrite(str(images_dir / fname), img)
        label_file = labels_dir / (Path(fname).stem + ".txt")
        label_file.write_text("\n".join(lines))

    return tmp_path


# ---------------------------------------------------------------------------
# Checker tests
# ---------------------------------------------------------------------------

def test_validator_no_issues(tmp_path):
    """A clean YOLO dataset with valid boxes produces 0 issues."""
    classes = ["crack", "rust"]
    dataset = make_yolo_dataset(
        tmp_path,
        [
            ("img_000.jpg", 640, 480, ["0 0.5 0.5 0.3 0.25"]),
            ("img_001.jpg", 640, 480, ["1 0.2 0.3 0.15 0.1"]),
        ],
        classes=classes,
    )
    validator = AnnotationValidator(str(dataset))
    report = validator.check()
    assert not report.has_issues(), f"Expected no issues but got: {report.issues}"


def test_validator_detects_oob_box(tmp_path):
    """Box with x_max > image_width triggers OOB_BOX."""
    classes = ["crack"]
    # cx=0.95, w=0.2 → x_max = (0.95 + 0.1) * 640 = 672 > 640
    dataset = make_yolo_dataset(
        tmp_path,
        [("img.jpg", 640, 480, ["0 0.95 0.5 0.2 0.2"])],
        classes=classes,
    )
    report = AnnotationValidator(str(dataset)).check()
    types = [i.issue_type for i in report.issues]
    assert IssueType.OOB_BOX in types, f"Expected OOB_BOX in {types}"


def test_validator_detects_zero_size_box(tmp_path):
    """Box with w=0 in YOLO format triggers ZERO_SIZE_BOX."""
    classes = ["crack"]
    # bw = 0 → x_min == x_max
    dataset = make_yolo_dataset(
        tmp_path,
        [("img.jpg", 640, 480, ["0 0.5 0.5 0.0 0.2"])],
        classes=classes,
    )
    report = AnnotationValidator(str(dataset)).check()
    types = [i.issue_type for i in report.issues]
    # Zero width should trigger ZERO_SIZE_BOX and/or NEGATIVE_DIMENSION
    assert any(
        t in types for t in (IssueType.ZERO_SIZE_BOX, IssueType.NEGATIVE_DIMENSION)
    ), f"Expected zero-size or negative-dimension issue in {types}"


def test_validator_detects_duplicate(tmp_path):
    """Two identical boxes on the same image trigger DUPLICATE_ANNOTATION."""
    classes = ["crack"]
    line = "0 0.5 0.5 0.3 0.25"
    dataset = make_yolo_dataset(
        tmp_path,
        [("img.jpg", 640, 480, [line, line])],
        classes=classes,
    )
    report = AnnotationValidator(str(dataset)).check()
    types = [i.issue_type for i in report.issues]
    assert IssueType.DUPLICATE_ANNOTATION in types, f"Expected DUPLICATE_ANNOTATION in {types}"


def test_validator_detects_missing_annotation(tmp_path):
    """Image without a corresponding label file triggers MISSING_ANNOTATION_FILE."""
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    # Create image but no label file
    img = np.full((480, 640, 3), 150, dtype=np.uint8)
    cv2.imwrite(str(images_dir / "orphan.jpg"), img)

    # Create one image WITH a label so directory structure is valid YOLO
    cv2.imwrite(str(images_dir / "paired.jpg"), img)
    (labels_dir / "paired.txt").write_text("0 0.5 0.5 0.3 0.25")
    (labels_dir / "classes.txt").write_text("crack")

    report = AnnotationValidator(str(tmp_path)).check()
    types = [i.issue_type for i in report.issues]
    assert IssueType.MISSING_ANNOTATION_FILE in types, f"Expected MISSING_ANNOTATION_FILE in {types}"


def test_validator_inconsistent_class_names(tmp_path):
    """Classes with same name but different casing trigger INCONSISTENT_CLASS_NAME."""
    # We simulate two label files using class index 0 —
    # classes.txt has "crack" but we'll put a second classes variant to test
    # the checker. The checker looks at actual label strings resolved from class IDs.
    # We use two separate classes in classes.txt that are case-variants of each other.
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    img = np.full((480, 640, 3), 150, dtype=np.uint8)
    cv2.imwrite(str(images_dir / "img_a.jpg"), img)
    cv2.imwrite(str(images_dir / "img_b.jpg"), img)

    # classes: index 0 = "crack", index 1 = "Crack" (case variant)
    (labels_dir / "classes.txt").write_text("crack\nCrack")
    # img_a uses class 0 ("crack"), img_b uses class 1 ("Crack")
    (labels_dir / "img_a.txt").write_text("0 0.5 0.5 0.3 0.25")
    (labels_dir / "img_b.txt").write_text("1 0.5 0.5 0.3 0.25")

    report = AnnotationValidator(str(tmp_path)).check()
    types = [i.issue_type for i in report.issues]
    assert IssueType.INCONSISTENT_CLASS_NAME in types, (
        f"Expected INCONSISTENT_CLASS_NAME in {types}"
    )


# ---------------------------------------------------------------------------
# Fixer tests
# ---------------------------------------------------------------------------

def test_fixer_clips_oob_boxes(tmp_path):
    """OOB box is clipped to image bounds in the output label file."""
    classes = ["crack"]
    src = tmp_path / "src"
    out = tmp_path / "out"

    # cx=0.95, bw=0.2 → x_max = 672 > 640 (oob)
    make_yolo_dataset(src, [("img.jpg", 640, 480, ["0 0.95 0.5 0.2 0.2"])], classes=classes)

    AnnotationFixer(str(src)).fix_all(str(out), clip_oob_boxes=True, remove_tiny_boxes=False)

    label = (out / "labels" / "img.txt").read_text().strip()
    assert label != "", "Expected a fixed label line in output"
    parts = label.split()
    assert len(parts) == 5
    cx, bw_n = float(parts[1]), float(parts[3])
    # x_max normalised must be <= 1.0
    x_max_n = cx + bw_n / 2
    assert x_max_n <= 1.0 + 1e-6, f"x_max_normalised={x_max_n} still out of bounds"


def test_fixer_removes_tiny_boxes(tmp_path):
    """Boxes smaller than min_box_size are removed from the output."""
    classes = ["crack"]
    src = tmp_path / "src"
    out = tmp_path / "out"

    # bw = 0.004 * 640 = 2.56 px → tiny (< 5 px)
    make_yolo_dataset(src, [("img.jpg", 640, 480, ["0 0.5 0.5 0.004 0.5"])], classes=classes)

    AnnotationFixer(str(src)).fix_all(str(out), remove_tiny_boxes=True, min_box_size=5)

    label = (out / "labels" / "img.txt").read_text().strip()
    assert label == "", f"Expected empty label file after tiny box removal, got: {label!r}"


def test_fixer_normalizes_class_names(tmp_path):
    """'Crack' is normalized to 'crack' in the output."""
    classes = ["Crack"]  # mixed-case original
    src = tmp_path / "src"
    out = tmp_path / "out"

    make_yolo_dataset(src, [("img.jpg", 640, 480, ["0 0.5 0.5 0.3 0.25"])], classes=classes)

    AnnotationFixer(str(src)).fix_all(str(out), normalize_class_names=True)

    classes_out = (out / "labels" / "classes.txt").read_text().strip().splitlines()
    assert "crack" in [c.strip() for c in classes_out], (
        f"Expected 'crack' in output classes.txt, got: {classes_out}"
    )


def test_fixer_writes_fix_report(tmp_path):
    """fix_report.json is written to output_dir after fix_all."""
    classes = ["crack"]
    src = tmp_path / "src"
    out = tmp_path / "out"

    make_yolo_dataset(src, [("img.jpg", 640, 480, ["0 0.5 0.5 0.3 0.25"])], classes=classes)

    AnnotationFixer(str(src)).fix_all(str(out))

    report_path = out / "fix_report.json"
    assert report_path.exists(), "fix_report.json was not created"
    data = json.loads(report_path.read_text())
    assert "total_images_processed" in data
    assert "fixes_applied" in data
    assert data["total_images_processed"] == 1


def test_fixer_does_not_modify_originals(tmp_path):
    """Original label files are unchanged after fix_all."""
    classes = ["crack"]
    src = tmp_path / "src"
    out = tmp_path / "out"
    original_label = "0 0.5 0.5 0.3 0.25"

    make_yolo_dataset(src, [("img.jpg", 640, 480, [original_label])], classes=classes)

    original_content = (src / "labels" / "img.txt").read_text()

    AnnotationFixer(str(src)).fix_all(str(out))

    after_content = (src / "labels" / "img.txt").read_text()
    assert original_content == after_content, (
        "Original label file was modified by fix_all!"
    )
