"""Annotation validation logic for OpenDefectKit."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console
from rich.table import Table

from opendefectkit.convert.base import Annotation, BoundingBox

console = Console()

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


class IssueType(Enum):
    """Categories of annotation quality issues."""

    OOB_BOX = "out_of_bounds_box"
    ZERO_SIZE_BOX = "zero_size_box"
    NEGATIVE_DIMENSION = "negative_dimension"
    DUPLICATE_ANNOTATION = "duplicate_annotation"
    INCONSISTENT_CLASS_NAME = "inconsistent_class_name"
    MISSING_ANNOTATION_FILE = "missing_annotation_file"
    CORRUPTED_FILE = "corrupted_file"


@dataclass
class Issue:
    """A single validation issue found in an annotation."""

    issue_type: IssueType
    image_path: str
    detail: str
    box_index: Optional[int] = None


@dataclass
class IssueReport:
    """Collection of issues found during validation."""

    issues: List[Issue] = field(default_factory=list)

    def has_issues(self) -> bool:
        """Return True if any issues were found."""
        return len(self.issues) > 0

    def count_by_type(self) -> Dict[str, int]:
        """Return a dict mapping issue type value to count."""
        counts: Dict[str, int] = {}
        for issue in self.issues:
            key = issue.issue_type.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def print_report(self) -> None:
        """Print a rich-formatted summary of all issues."""
        if not self.issues:
            console.print("[bold green]No issues found.[/bold green]")
            return

        counts = self.count_by_type()
        table = Table(title="Annotation Validation Report", show_lines=True)
        table.add_column("Issue Type", style="cyan")
        table.add_column("Count", justify="right", style="magenta")
        for issue_type, count in counts.items():
            table.add_row(issue_type, str(count))
        console.print(table)

        console.print(f"\n[bold red]Total issues: {len(self.issues)}[/bold red]")
        for issue in self.issues[:20]:
            loc = f"{issue.image_path}"
            if issue.box_index is not None:
                loc += f" (box {issue.box_index})"
            console.print(f"  [{issue.issue_type.value}] {loc}: {issue.detail}")
        if len(self.issues) > 20:
            console.print(f"  ... and {len(self.issues) - 20} more issues.")


def _compute_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """Compute intersection-over-union between two bounding boxes."""
    ix1 = max(box1.x_min, box2.x_min)
    iy1 = max(box1.y_min, box2.y_min)
    ix2 = min(box1.x_max, box2.x_max)
    iy2 = min(box1.y_max, box2.y_max)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area1 = (box1.x_max - box1.x_min) * (box1.y_max - box1.y_min)
    area2 = (box2.x_max - box2.x_min) * (box2.y_max - box2.y_min)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def _get_image_size(image_path: Path):
    """Return (width, height) for an image file using PIL."""
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            return img.width, img.height
    except Exception:
        try:
            import cv2
            img = cv2.imread(str(image_path))
            if img is not None:
                h, w = img.shape[:2]
                return w, h
        except Exception:
            pass
    return None, None


def _parse_yolo_label(
    label_file: Path,
    class_names: List[str],
    image_width: int,
    image_height: int,
) -> Optional[List[BoundingBox]]:
    """Parse a YOLO .txt label file; return None if corrupted."""
    try:
        text = label_file.read_text().strip()
        boxes: List[BoundingBox] = []
        if not text:
            return boxes
        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                return None  # malformed
            class_id = int(parts[0])
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x_min = (cx - bw / 2) * image_width
            y_min = (cy - bh / 2) * image_height
            x_max = (cx + bw / 2) * image_width
            y_max = (cy + bh / 2) * image_height
            label = class_names[class_id] if class_id < len(class_names) else str(class_id)
            boxes.append(BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, label=label))
        return boxes
    except Exception:
        return None


class AnnotationValidator:
    """Validates annotation files in a YOLO dataset directory."""

    def __init__(self, dataset_dir: str, annotation_format: str = "auto") -> None:
        """Initialise validator with path to dataset root."""
        self.dataset_dir = Path(dataset_dir)
        self.annotation_format = annotation_format

    def _find_images_and_labels_dirs(self):
        """Return (images_dir, labels_dir) paths, either as subdirs or same dir."""
        images_dir = self.dataset_dir / "images"
        labels_dir = self.dataset_dir / "labels"
        if images_dir.is_dir() and labels_dir.is_dir():
            return images_dir, labels_dir
        # Fallback: look for images + txt files in the dataset root
        return self.dataset_dir, self.dataset_dir

    def _find_classes(self, labels_dir: Path) -> List[str]:
        """Discover class names from classes.txt or obj.names."""
        for search_dir in (self.dataset_dir, labels_dir):
            for name in ("classes.txt", "obj.names"):
                p = search_dir / name
                if p.exists():
                    return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
        return []

    def check(self) -> IssueReport:
        """Run all checks on the dataset and return an IssueReport."""
        issues: List[Issue] = []
        images_dir, labels_dir = self._find_images_and_labels_dirs()
        class_names = self._find_classes(labels_dir)

        # Collect image files
        image_files = {
            f.stem: f
            for f in images_dir.iterdir()
            if f.suffix.lower() in _IMAGE_EXTENSIONS
        }

        # Track all labels seen for inconsistent-class-name check
        all_labels_lower: Dict[str, List[str]] = {}  # lowercase -> [original forms seen]

        # --- Check 6: Missing annotation files ---
        label_stems = {f.stem for f in labels_dir.glob("*.txt")}
        for stem, img_file in image_files.items():
            if stem not in label_stems:
                issues.append(Issue(
                    issue_type=IssueType.MISSING_ANNOTATION_FILE,
                    image_path=str(img_file),
                    detail=f"No label file found for {img_file.name}",
                ))

        annotations: List[Annotation] = []

        for label_file in sorted(labels_dir.glob("*.txt")):
            if label_file.name in ("classes.txt", "obj.names"):
                continue

            stem = label_file.stem
            img_file = image_files.get(stem)

            # Resolve image size
            if img_file is not None:
                size = _get_image_size(img_file)
                w, h = size if size[0] is not None else (640, 640)
                img_name = str(img_file)
            else:
                w, h = 640, 640
                img_name = stem + ".jpg"

            # --- Check 7: Corrupted files ---
            boxes = _parse_yolo_label(label_file, class_names, w, h)
            if boxes is None:
                issues.append(Issue(
                    issue_type=IssueType.CORRUPTED_FILE,
                    image_path=img_name,
                    detail=f"Could not parse label file: {label_file.name}",
                ))
                continue

            ann = Annotation(image_path=img_name, image_width=w, image_height=h, boxes=boxes)
            annotations.append(ann)

            for idx, box in enumerate(boxes):
                # Collect labels for inconsistency check
                lk = box.label.lower()
                if lk not in all_labels_lower:
                    all_labels_lower[lk] = []
                if box.label not in all_labels_lower[lk]:
                    all_labels_lower[lk].append(box.label)

                # --- Check 1: Out-of-bounds ---
                oob = (
                    box.x_min < 0
                    or box.y_min < 0
                    or box.x_max > w
                    or box.y_max > h
                )
                if oob:
                    issues.append(Issue(
                        issue_type=IssueType.OOB_BOX,
                        image_path=img_name,
                        detail=(
                            f"Box [{box.x_min:.1f},{box.y_min:.1f},"
                            f"{box.x_max:.1f},{box.y_max:.1f}] "
                            f"exceeds image bounds ({w}x{h})"
                        ),
                        box_index=idx,
                    ))

                # --- Check 3: Negative dimensions (x_min >= x_max etc.) ---
                if box.x_min >= box.x_max or box.y_min >= box.y_max:
                    issues.append(Issue(
                        issue_type=IssueType.NEGATIVE_DIMENSION,
                        image_path=img_name,
                        detail=(
                            f"Box has non-positive dimension: "
                            f"w={box.x_max - box.x_min:.2f}, h={box.y_max - box.y_min:.2f}"
                        ),
                        box_index=idx,
                    ))

                # --- Check 2: Zero-size box ---
                bw = box.x_max - box.x_min
                bh = box.y_max - box.y_min
                if bw <= 0 or bh <= 0:
                    issues.append(Issue(
                        issue_type=IssueType.ZERO_SIZE_BOX,
                        image_path=img_name,
                        detail=f"Box has zero or negative size: w={bw:.2f}, h={bh:.2f}",
                        box_index=idx,
                    ))

            # --- Check 4: Duplicate annotations ---
            for (i, b1), (j, b2) in itertools.combinations(enumerate(boxes), 2):
                if b1.label == b2.label and _compute_iou(b1, b2) > 0.95:
                    issues.append(Issue(
                        issue_type=IssueType.DUPLICATE_ANNOTATION,
                        image_path=img_name,
                        detail=f"Boxes {i} and {j} are near-identical (IoU>0.95, label='{b1.label}')",
                        box_index=i,
                    ))

        # --- Check 5: Inconsistent class names ---
        for lower_key, variants in all_labels_lower.items():
            if len(variants) > 1:
                # Find the first image that uses each variant for attribution
                for label_file in sorted(labels_dir.glob("*.txt")):
                    if label_file.name in ("classes.txt", "obj.names"):
                        continue
                    stem = label_file.stem
                    img_file = image_files.get(stem)
                    img_name = str(img_file) if img_file else stem + ".jpg"
                    w = annotations[0].image_width if annotations else 640
                    h = annotations[0].image_height if annotations else 640
                    # Find one image that uses a non-canonical variant
                    try:
                        text = label_file.read_text().strip()
                    except Exception:
                        continue
                    for line in text.splitlines():
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        try:
                            class_id = int(parts[0])
                        except ValueError:
                            continue
                        label = class_names[class_id] if class_id < len(class_names) else str(class_id)
                        if label.lower() == lower_key and label != variants[0]:
                            issues.append(Issue(
                                issue_type=IssueType.INCONSISTENT_CLASS_NAME,
                                image_path=img_name,
                                detail=(
                                    f"Class '{label}' is a case variant of '{variants[0]}' "
                                    f"(all variants: {variants})"
                                ),
                            ))
                            break
                break  # only report once per lower-key group

        return IssueReport(issues=issues)
