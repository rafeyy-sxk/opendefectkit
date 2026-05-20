"""Dataset profiling, health scoring, and severity scoring for OpenDefectKit."""
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def _find_images(directory: Path) -> List[Path]:
    """Return all image files under a directory (non-recursive)."""
    return [p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]


def _find_all_images(root: Path) -> List[Path]:
    """Return all image files recursively under root."""
    return [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]


def _load_classes(dataset_dir: Path) -> Optional[List[str]]:
    """Load class names from classes.txt if present."""
    classes_file = dataset_dir / "classes.txt"
    if classes_file.exists():
        lines = classes_file.read_text().strip().splitlines()
        return [ln.strip() for ln in lines if ln.strip()]
    return None


@dataclass
class ProfileReport:
    """Full profiling report for a YOLO-format dataset."""

    total_images: int
    annotated_images: int
    unannotated_images: int
    class_distribution: Dict[str, int]
    bbox_widths: List[float]
    bbox_heights: List[float]
    image_widths: List[int]
    image_heights: List[int]
    duplicate_count: int
    low_quality_images: List[str]
    recommended_split: Dict[str, int]
    defect_density_grid: List[List[float]]

    def summary(self) -> str:
        """Return a rich-formatted text summary of the profile report."""
        lines = [
            "=" * 60,
            "  OpenDefectKit — Dataset Profile Summary",
            "=" * 60,
            f"  Total images      : {self.total_images}",
            f"  Annotated         : {self.annotated_images}",
            f"  Unannotated       : {self.unannotated_images}",
            f"  Duplicates        : {self.duplicate_count}",
            f"  Low-quality imgs  : {len(self.low_quality_images)}",
            "",
            "  Class distribution:",
        ]
        for cls, cnt in self.class_distribution.items():
            lines.append(f"    {cls:20s} : {cnt}")
        lines += [
            "",
            "  Recommended split:",
            f"    Train : {self.recommended_split.get('train', 0)}",
            f"    Val   : {self.recommended_split.get('val', 0)}",
            f"    Test  : {self.recommended_split.get('test', 0)}",
            "=" * 60,
        ]
        return "\n".join(lines)


class DatasetProfiler:
    """Profiles a YOLO-format dataset directory."""

    def __init__(self, dataset_dir: str) -> None:
        """Initialise with the root directory of the dataset."""
        self.dataset_dir = Path(dataset_dir)

    def _detect_layout(self) -> Tuple[Path, Optional[Path]]:
        """Return (images_dir, labels_dir) based on layout detection."""
        images_sub = self.dataset_dir / "images"
        labels_sub = self.dataset_dir / "labels"
        if images_sub.exists() and labels_sub.exists():
            return images_sub, labels_sub
        # flat layout: images and .txt files live in the same dir
        return self.dataset_dir, None

    def _label_path(self, image_path: Path, labels_dir: Optional[Path]) -> Path:
        """Return the expected label file path for an image."""
        if labels_dir is not None:
            return labels_dir / (image_path.stem + ".txt")
        return image_path.with_suffix(".txt")

    def run(self) -> ProfileReport:
        """Run the full profiling pipeline and return a ProfileReport."""
        images_dir, labels_dir = self._detect_layout()
        classes = _load_classes(self.dataset_dir)

        image_files = _find_images(images_dir)
        total_images = len(image_files)

        # ── Annotation analysis ──────────────────────────────────────────────
        annotated_images = 0
        unannotated_images = 0
        class_counts: Dict[int, int] = {}
        bbox_widths: List[float] = []
        bbox_heights: List[float] = []
        image_widths: List[int] = []
        image_heights: List[int] = []
        density_grid = np.zeros((10, 10), dtype=float)

        for img_path in image_files:
            try:
                with Image.open(img_path) as pil_img:
                    img_w, img_h = pil_img.size
            except Exception:
                img_w, img_h = 640, 480

            image_widths.append(img_w)
            image_heights.append(img_h)

            lbl_path = self._label_path(img_path, labels_dir)
            if lbl_path.exists():
                lines = [ln.strip() for ln in lbl_path.read_text().splitlines() if ln.strip()]
                if lines:
                    annotated_images += 1
                    for line in lines:
                        parts = line.split()
                        if len(parts) < 5:
                            continue
                        class_id = int(parts[0])
                        cx, cy, bw, bh = (float(p) for p in parts[1:5])
                        class_counts[class_id] = class_counts.get(class_id, 0) + 1
                        bbox_widths.append(bw * img_w)
                        bbox_heights.append(bh * img_h)
                        # density grid
                        gx = min(int(cx * 10), 9)
                        gy = min(int(cy * 10), 9)
                        density_grid[gy][gx] += 1
                else:
                    unannotated_images += 1
            else:
                unannotated_images += 1

        # Normalise density grid
        total_boxes = density_grid.sum()
        if total_boxes > 0:
            density_grid = density_grid / total_boxes
        defect_density_grid: List[List[float]] = density_grid.tolist()

        # Build class distribution with names
        if classes:
            class_distribution = {
                classes[cid] if cid < len(classes) else f"class_{cid}": cnt
                for cid, cnt in class_counts.items()
            }
        else:
            class_distribution = {f"class_{cid}": cnt for cid, cnt in class_counts.items()}

        # ── Duplicate detection ──────────────────────────────────────────────
        hashes: Dict[str, List[Path]] = {}
        for img_path in image_files:
            md5 = hashlib.md5(img_path.read_bytes()).hexdigest()
            hashes.setdefault(md5, []).append(img_path)
        duplicate_count = sum(len(v) - 1 for v in hashes.values() if len(v) > 1)

        # ── Low quality detection ────────────────────────────────────────────
        low_quality_images: List[str] = []
        for img_path in image_files:
            img_cv = cv2.imread(str(img_path))
            if img_cv is None:
                continue
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            mean_val = float(gray.mean())
            if laplacian_var < 100 or mean_val < 40:
                low_quality_images.append(str(img_path))

        # ── Recommended split ────────────────────────────────────────────────
        n = annotated_images
        test_n = max(1, int(n * 0.10))
        val_n = max(1, int(n * 0.20))
        train_n = max(1, n - val_n - test_n)
        recommended_split = {"train": train_n, "val": val_n, "test": test_n}

        return ProfileReport(
            total_images=total_images,
            annotated_images=annotated_images,
            unannotated_images=unannotated_images,
            class_distribution=class_distribution,
            bbox_widths=bbox_widths,
            bbox_heights=bbox_heights,
            image_widths=image_widths,
            image_heights=image_heights,
            duplicate_count=duplicate_count,
            low_quality_images=low_quality_images,
            recommended_split=recommended_split,
            defect_density_grid=defect_density_grid,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Health Score
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class HealthCheckItem:
    """A single health-check criterion result."""

    criterion: str
    passed: Optional[bool]
    detail: str
    weight: int


@dataclass
class HealthScoreResult:
    """Aggregated health score result."""

    score: int
    items: List[HealthCheckItem]
    recommendation: str

    def __str__(self) -> str:
        """Return formatted score with checkmarks."""
        lines = [f"Dataset Health Score: {self.score}/100", ""]
        for item in self.items:
            if item.passed is True:
                mark = "[PASS]"
            elif item.passed is False:
                mark = "[FAIL]"
            else:
                mark = "[WARN]"
            lines.append(f"  {mark} ({item.weight:2d}pt) {item.criterion}")
            lines.append(f"         {item.detail}")
        lines += ["", f"Recommendation: {self.recommendation}"]
        return "\n".join(lines)


class DatasetHealthScore:
    """Computes a 0-100 health score for a YOLO-format dataset."""

    def __init__(self, dataset_dir: str) -> None:
        """Initialise with the root directory of the dataset."""
        self.dataset_dir = Path(dataset_dir)

    def run(self) -> HealthScoreResult:
        """Run all health checks and return a HealthScoreResult."""
        profiler = DatasetProfiler(str(self.dataset_dir))
        report = profiler.run()

        items: List[HealthCheckItem] = []
        failed_criteria: List[str] = []

        # 1. Min samples per class ≥ 100 (weight=20)
        if report.class_distribution:
            min_count = min(report.class_distribution.values())
            passed_1 = min_count >= 100
            detail_1 = (
                f"Min samples: {min_count} (need ≥ 100)"
                if not passed_1
                else f"Min samples: {min_count} ✓"
            )
        else:
            passed_1 = False
            detail_1 = "No annotations found"
        items.append(HealthCheckItem("Min samples per class ≥ 100", passed_1, detail_1, 20))
        if not passed_1:
            failed_criteria.append("increase samples per class to at least 100")

        # 2. Class imbalance ratio < 10:1 (weight=20)
        if len(report.class_distribution) >= 2:
            counts = list(report.class_distribution.values())
            ratio = max(counts) / min(counts)
            passed_2 = ratio < 10
            detail_2 = f"Imbalance ratio: {ratio:.1f}:1 (need < 10:1)"
        elif len(report.class_distribution) == 1:
            passed_2 = True
            detail_2 = "Single class dataset — ratio check N/A"
        else:
            passed_2 = False
            detail_2 = "No annotations found"
        items.append(HealthCheckItem("Class imbalance ratio < 10:1", passed_2, detail_2, 20))
        if not passed_2:
            failed_criteria.append("balance class distribution (ratio is ≥ 10:1)")

        # 3. Image quality < 5% low-quality (weight=15)
        if report.total_images > 0:
            lq_pct = len(report.low_quality_images) / report.total_images
            passed_3 = lq_pct < 0.05
            detail_3 = f"Low-quality: {len(report.low_quality_images)}/{report.total_images} ({lq_pct*100:.1f}%)"
        else:
            passed_3 = False
            detail_3 = "No images found"
        items.append(HealthCheckItem("Image quality (blur/dark < 5%)", passed_3, detail_3, 15))
        if not passed_3:
            failed_criteria.append("remove or retake blurry/dark images (> 5% are low quality)")

        # 4. Validation split exists (weight=15)
        val_dir = self.dataset_dir / "val"
        valid_dir = self.dataset_dir / "valid"
        has_val = (val_dir.exists() and any(val_dir.rglob("*"))) or (
            valid_dir.exists() and any(valid_dir.rglob("*"))
        )
        passed_4 = has_val
        detail_4 = "val/ or valid/ directory found" if has_val else "No val/ or valid/ subdirectory"
        items.append(HealthCheckItem("Validation split exists", passed_4, detail_4, 15))
        if not passed_4:
            failed_criteria.append("create a val/ or valid/ subdirectory with validation images")

        # 5. Duplicate images < 1% (weight=10)
        if report.total_images > 0:
            dup_pct = report.duplicate_count / report.total_images
            passed_5 = dup_pct < 0.01
            detail_5 = f"Duplicates: {report.duplicate_count}/{report.total_images} ({dup_pct*100:.1f}%)"
        else:
            passed_5 = False
            detail_5 = "No images found"
        items.append(HealthCheckItem("Duplicate images < 1%", passed_5, detail_5, 10))
        if not passed_5:
            failed_criteria.append("remove duplicate images (≥ 1% are duplicates)")

        # 6. Annotation coverage > 95% (weight=20)
        if report.total_images > 0:
            coverage = report.annotated_images / report.total_images
            passed_6 = coverage > 0.95
            detail_6 = f"Coverage: {report.annotated_images}/{report.total_images} ({coverage*100:.1f}%)"
        else:
            passed_6 = False
            detail_6 = "No images found"
        items.append(HealthCheckItem("Annotation coverage > 95%", passed_6, detail_6, 20))
        if not passed_6:
            failed_criteria.append("annotate more images (coverage is ≤ 95%)")

        score = sum(item.weight for item in items if item.passed is True)

        if failed_criteria:
            recommendation = "To improve dataset quality: " + "; ".join(failed_criteria) + "."
        else:
            recommendation = "Dataset looks healthy — no critical issues found."

        return HealthScoreResult(score=score, items=items, recommendation=recommendation)


# ─────────────────────────────────────────────────────────────────────────────
#  Severity Scorer
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DefectResult:
    """A single detected defect with area and confidence."""

    defect_type: str
    area_pct: float
    confidence: float


@dataclass
class SeverityResult:
    """Severity assessment for a single image."""

    severity: str
    severity_score: float
    defects: List[DefectResult]
    recommended_action: str


class SeverityScorer:
    """Scores defect severity using a YOLOv8 model (requires ultralytics)."""

    def __init__(self, model_path: str) -> None:
        """Load the YOLOv8 model; raises ImportError if ultralytics is absent."""
        try:
            from ultralytics import YOLO  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "SeverityScorer requires ultralytics. "
                "Install with: pip install opendefectkit[benchmark]"
            ) from exc

        from ultralytics import YOLO

        self.model = YOLO(model_path)

    def score_image(self, image_path: str) -> SeverityResult:
        """Run inference on a single image and return a SeverityResult."""
        with Image.open(image_path) as pil_img:
            img_w, img_h = pil_img.size

        results = self.model(image_path)
        defects: List[DefectResult] = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                coords = box.xyxy[0].tolist()
                x1, y1, x2, y2 = coords
                bw = x2 - x1
                bh = y2 - y1
                area_pct = (bw * bh) / (img_w * img_h)
                confidence = float(box.conf[0])
                cls_id = int(box.cls[0])
                names = result.names or {}
                defect_type = names.get(cls_id, f"class_{cls_id}")
                defects.append(DefectResult(defect_type=defect_type, area_pct=area_pct, confidence=confidence))

        severity_score = min(1.0, sum(d.area_pct * d.confidence for d in defects))

        if severity_score < 0.2:
            severity = "low"
            recommended_action = "flag_for_review"
        elif severity_score < 0.5:
            severity = "medium"
            recommended_action = "flag_for_review"
        elif severity_score < 0.8:
            severity = "high"
            recommended_action = "quarantine"
        else:
            severity = "critical"
            recommended_action = "remove_from_production"

        return SeverityResult(
            severity=severity,
            severity_score=severity_score,
            defects=defects,
            recommended_action=recommended_action,
        )
