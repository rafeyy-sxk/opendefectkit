"""Annotation fixing utilities for OpenDefectKit."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from opendefectkit.convert.base import BoundingBox
from opendefectkit.validate.checker import _compute_iou, _get_image_size, _parse_yolo_label

console = Console()

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


class AnnotationFixer:
    """Reads YOLO annotations, applies quality fixes, and writes to an output directory."""

    def __init__(self, dataset_dir: str) -> None:
        """Initialise fixer with path to dataset root."""
        self.dataset_dir = Path(dataset_dir)

    def _find_images_and_labels_dirs(self):
        """Return (images_dir, labels_dir) from dataset layout."""
        images_dir = self.dataset_dir / "images"
        labels_dir = self.dataset_dir / "labels"
        if images_dir.is_dir() and labels_dir.is_dir():
            return images_dir, labels_dir
        return self.dataset_dir, self.dataset_dir

    def _find_classes(self, labels_dir: Path) -> List[str]:
        """Discover class names from classes.txt or obj.names."""
        for search_dir in (self.dataset_dir, labels_dir):
            for name in ("classes.txt", "obj.names"):
                p = search_dir / name
                if p.exists():
                    return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
        return []

    def _clip_box(self, box: BoundingBox, width: int, height: int) -> BoundingBox:
        """Clamp box coordinates to image bounds."""
        return BoundingBox(
            x_min=max(0.0, min(box.x_min, width)),
            y_min=max(0.0, min(box.y_min, height)),
            x_max=max(0.0, min(box.x_max, width)),
            y_max=max(0.0, min(box.y_max, height)),
            label=box.label,
            confidence=box.confidence,
        )

    def fix_all(
        self,
        output_dir: str,
        normalize_class_names: bool = True,
        clip_oob_boxes: bool = True,
        remove_duplicates: bool = True,
        remove_tiny_boxes: bool = True,
        min_box_size: int = 5,
    ) -> None:
        """Apply fixes and write corrected YOLO labels to output_dir.

        Never modifies the original annotation files.
        Writes fix_report.json and a labels/ directory with fixed .txt files.
        """
        out_path = Path(output_dir)
        out_labels = out_path / "labels"
        out_labels.mkdir(parents=True, exist_ok=True)

        images_dir, labels_dir = self._find_images_and_labels_dirs()
        class_names = self._find_classes(labels_dir)

        image_files: Dict[str, Path] = {
            f.stem: f
            for f in images_dir.iterdir()
            if f.suffix.lower() in _IMAGE_EXTENSIONS
        }

        label_files = [
            f for f in sorted(labels_dir.glob("*.txt"))
            if f.name not in ("classes.txt", "obj.names")
        ]

        fix_counts: Dict[str, int] = {
            "class_names_normalized": 0,
            "oob_boxes_clipped": 0,
            "duplicates_removed": 0,
            "tiny_boxes_removed": 0,
        }

        # Build a mapping of normalised class names for consistent output
        normalised_classes: List[str] = (
            [c.lower() for c in class_names] if normalize_class_names else list(class_names)
        )
        name_to_id: Dict[str, int] = {name: idx for idx, name in enumerate(normalised_classes)}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
        ) as progress:
            task = progress.add_task("Fixing annotations...", total=len(label_files))

            for label_file in label_files:
                stem = label_file.stem
                img_file: Optional[Path] = image_files.get(stem)

                if img_file is not None:
                    w, h = _get_image_size(img_file)
                    if w is None:
                        w, h = 640, 640
                else:
                    w, h = 640, 640

                boxes = _parse_yolo_label(label_file, class_names, w, h)
                if boxes is None:
                    console.print(
                        f"[yellow]Warning:[/yellow] Could not parse {label_file.name}, skipping."
                    )
                    progress.advance(task)
                    continue

                fixed_boxes: List[BoundingBox] = []

                for box in boxes:
                    # Normalise class name
                    original_label = box.label
                    label = box.label.lower() if normalize_class_names else box.label
                    if label != original_label:
                        fix_counts["class_names_normalized"] += 1

                    # Clip OOB
                    was_oob = (
                        box.x_min < 0
                        or box.y_min < 0
                        or box.x_max > w
                        or box.y_max > h
                    )
                    clipped = BoundingBox(
                        x_min=box.x_min,
                        y_min=box.y_min,
                        x_max=box.x_max,
                        y_max=box.y_max,
                        label=label,
                        confidence=box.confidence,
                    )
                    if clip_oob_boxes and was_oob:
                        clipped = self._clip_box(clipped, w, h)
                        fix_counts["oob_boxes_clipped"] += 1

                    # Tiny box check (after clipping)
                    bw = clipped.x_max - clipped.x_min
                    bh = clipped.y_max - clipped.y_min
                    if remove_tiny_boxes and (bw < min_box_size or bh < min_box_size):
                        fix_counts["tiny_boxes_removed"] += 1
                        continue

                    fixed_boxes.append(clipped)

                # Duplicate removal
                if remove_duplicates:
                    kept: List[BoundingBox] = []
                    for box in fixed_boxes:
                        is_dup = False
                        for existing in kept:
                            if existing.label == box.label and _compute_iou(existing, box) > 0.95:
                                is_dup = True
                                fix_counts["duplicates_removed"] += 1
                                break
                        if not is_dup:
                            kept.append(box)
                    fixed_boxes = kept

                # Write fixed label file
                lines: List[str] = []
                for box in fixed_boxes:
                    label = box.label
                    if label not in name_to_id:
                        name_to_id[label] = len(name_to_id)
                        normalised_classes.append(label)
                    class_id = name_to_id[label]
                    cx = (box.x_min + box.x_max) / 2 / w
                    cy = (box.y_min + box.y_max) / 2 / h
                    bw_n = (box.x_max - box.x_min) / w
                    bh_n = (box.y_max - box.y_min) / h
                    lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw_n:.6f} {bh_n:.6f}")

                (out_labels / label_file.name).write_text("\n".join(lines))
                progress.advance(task)

        # Write classes.txt to output labels dir
        (out_labels / "classes.txt").write_text("\n".join(normalised_classes))

        # Write fix report
        report = {
            "total_images_processed": len(label_files),
            "fixes_applied": fix_counts,
        }
        (out_path / "fix_report.json").write_text(json.dumps(report, indent=2))
        console.print(f"[bold green]Fix report written to {out_path / 'fix_report.json'}[/bold green]")
