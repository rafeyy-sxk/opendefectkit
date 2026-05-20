"""DefectPipeline — orchestrator for end-to-end OpenDefectKit workflows."""
import shutil
from pathlib import Path
from typing import List, Optional

from rich.console import Console

console = Console()

VALID_STEPS: List[str] = [
    "detect_format",
    "validate_annotations",
    "fix_annotations",
    "convert_to_yolo",
    "profile_dataset",
    "augment_defects",
    "generate_splits",
    "export_report",
]

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def _find_annotated_images(dataset_dir: Path) -> List[Path]:
    """Return image paths that have a corresponding YOLO .txt label file."""
    annotated: List[Path] = []
    images_dir = dataset_dir / "images" if (dataset_dir / "images").exists() else dataset_dir
    labels_dir = dataset_dir / "labels" if (dataset_dir / "labels").exists() else dataset_dir

    for img in images_dir.rglob("*"):
        if img.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue
        lbl = (labels_dir / img.stem).with_suffix(".txt")
        if lbl.exists() and lbl.stat().st_size > 0:
            annotated.append(img)
    return annotated


class DefectPipeline:
    """End-to-end pipeline for processing defect detection datasets."""

    def __init__(self, raw_data_dir: str, output_dir: str) -> None:
        """Initialise pipeline with source and output directories."""
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self._completed_steps: List[str] = []
        # Internal state shared across steps
        self._fixed_dir: Optional[Path] = None
        self._detected_format: Optional[str] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, steps: Optional[List[str]] = None) -> None:
        """Execute pipeline steps in order. Skips already-completed steps."""
        steps_to_run = steps if steps is not None else list(VALID_STEPS)

        unknown = [s for s in steps_to_run if s not in VALID_STEPS]
        if unknown:
            raise ValueError(
                f"Unknown pipeline step(s): {unknown}. Valid steps: {VALID_STEPS}"
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)

        for step in steps_to_run:
            if step in self._completed_steps:
                console.print(f"[dim]Skipping (already done): {step}[/dim]")
                continue

            sentinel = self.output_dir / f".{step}_done"
            if sentinel.exists():
                console.print(f"[dim]Skipping (sentinel found): {step}[/dim]")
                self._completed_steps.append(step)
                continue

            console.print(f"[bold green]Running step: {step}[/bold green]")
            dispatch = getattr(self, f"_step_{step}")
            dispatch()
            sentinel.touch()
            self._completed_steps.append(step)

    # ── Step implementations ──────────────────────────────────────────────────

    def _step_detect_format(self) -> None:
        """Detect the annotation format of the raw data directory."""
        from opendefectkit.convert import detect_format

        fmt = detect_format(str(self.raw_data_dir))
        self._detected_format = fmt
        console.print(f"  Detected format: [bold cyan]{fmt}[/bold cyan]")

    def _step_validate_annotations(self) -> None:
        """Run annotation validation and print the issue report."""
        from opendefectkit.validate import AnnotationValidator

        report = AnnotationValidator(str(self.raw_data_dir)).check()
        report.print_report()

    def _step_fix_annotations(self) -> None:
        """Run annotation fixer and store the fixed dataset path."""
        from opendefectkit.validate import AnnotationFixer

        fixed_dir = self.output_dir / "fixed"
        AnnotationFixer(str(self.raw_data_dir)).fix_all(str(fixed_dir))
        self._fixed_dir = fixed_dir
        console.print(f"  Fixed annotations saved to: {fixed_dir}")

    def _step_convert_to_yolo(self) -> None:
        """Convert annotations to YOLO format unless source is already YOLO."""
        from opendefectkit.convert import auto_detect_and_convert

        source = self._fixed_dir if self._fixed_dir is not None else self.raw_data_dir
        fmt = self._detected_format

        if fmt == "yolo":
            console.print("  Source already in YOLO format — skipping conversion.")
            return

        yolo_dir = self.output_dir / "yolo"
        auto_detect_and_convert(str(source), "yolo", str(yolo_dir))
        console.print(f"  YOLO dataset written to: {yolo_dir}")

    def _step_profile_dataset(self) -> None:
        """Run DatasetProfiler and print the summary."""
        from opendefectkit.analyze import DatasetProfiler

        report = DatasetProfiler(str(self.raw_data_dir)).run()
        console.print(report.summary())

    def _step_augment_defects(self) -> None:
        """Generate synthetic defects for each class if dataset is small."""
        from opendefectkit.augment import SyntheticDefectGenerator

        annotated = _find_annotated_images(self.raw_data_dir)
        if len(annotated) >= 100:
            console.print(
                f"  {len(annotated)} annotated images found — no augmentation needed."
            )
            return

        console.print(
            f"  Only {len(annotated)} annotated images — generating synthetic defects."
        )

        # Discover class names from classes.txt
        classes_file = self.raw_data_dir / "classes.txt"
        classes: List[str] = []
        if classes_file.exists():
            classes = [ln.strip() for ln in classes_file.read_text().splitlines() if ln.strip()]

        if not classes:
            classes = ["crack"]  # sensible default for industrial CV
            console.print("  No classes.txt found — defaulting to ['crack'].")

        generator = SyntheticDefectGenerator(seed=42)
        images_dir = (
            self.raw_data_dir / "images"
            if (self.raw_data_dir / "images").exists()
            else self.raw_data_dir
        )
        aug_out = self.output_dir / "augmented"

        for cls in classes:
            cls_out = str(aug_out / cls)
            count = max(100 - len(annotated), 10)
            if cls == "crack":
                generator.add_cracks(str(images_dir), cls_out, count)
            elif cls == "rust":
                generator.add_rust(str(images_dir), cls_out, count)
            elif cls in ("scratch", "scratches"):
                generator.add_scratches(str(images_dir), cls_out, count)
            else:
                # Unknown class — generate cracks as a proxy synthetic defect
                generator.add_cracks(str(images_dir), cls_out, count)
            console.print(f"  Generated {count} synthetic '{cls}' samples -> {cls_out}")

    def _step_generate_splits(self) -> None:
        """Create 70/20/10 train/val/test splits using file copies."""
        annotated = _find_annotated_images(self.raw_data_dir)

        if not annotated:
            console.print("  No annotated images found — skipping split generation.")
            return

        total = len(annotated)
        n_train = int(total * 0.70)
        n_val = int(total * 0.20)
        # Remaining images go to test (handles rounding)
        n_test = total - n_train - n_val

        splits = {
            "train": annotated[:n_train],
            "val": annotated[n_train : n_train + n_val],
            "test": annotated[n_train + n_val :],
        }

        labels_dir = (
            self.raw_data_dir / "labels"
            if (self.raw_data_dir / "labels").exists()
            else self.raw_data_dir
        )

        for split_name, img_paths in splits.items():
            split_img_dir = self.output_dir / split_name / "images"
            split_lbl_dir = self.output_dir / split_name / "labels"
            split_img_dir.mkdir(parents=True, exist_ok=True)
            split_lbl_dir.mkdir(parents=True, exist_ok=True)

            for img_path in img_paths:
                shutil.copy2(str(img_path), str(split_img_dir / img_path.name))
                lbl_path = (labels_dir / img_path.stem).with_suffix(".txt")
                if lbl_path.exists():
                    shutil.copy2(str(lbl_path), str(split_lbl_dir / lbl_path.name))

        console.print(
            f"  Split complete: train={n_train}, val={n_val}, test={n_test}"
        )

    def _step_export_report(self) -> None:
        """Generate HTML dataset report via DatasetVisualizer and DatasetHealthScore."""
        from opendefectkit.analyze import DatasetHealthScore, DatasetVisualizer

        html_path = str(self.output_dir / "dataset_report.html")

        visualizer = DatasetVisualizer(str(self.raw_data_dir))
        visualizer.generate_html_report(html_path)

        health = DatasetHealthScore(str(self.raw_data_dir)).run()
        console.print(str(health))
        console.print(f"  HTML report saved: {html_path}")
