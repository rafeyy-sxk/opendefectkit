"""
Crack detection pipeline example.

Shows: format detection -> validation -> YOLO conversion -> augmentation -> health check.

Usage:
    python examples/crack_detection_pipeline.py --dataset path/to/dataset --output path/to/output
"""
import argparse
import sys

from opendefectkit.augment import SyntheticDefectGenerator
from opendefectkit.analyze import DatasetHealthScore
from opendefectkit.convert import detect_format, auto_detect_and_convert
from opendefectkit.validate import AnnotationValidator, AnnotationFixer


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end crack detection pipeline")
    parser.add_argument("--dataset", required=True, help="Path to raw dataset directory")
    parser.add_argument(
        "--output", default="output/crack_pipeline", help="Output directory (default: output/crack_pipeline)"
    )
    args = parser.parse_args()

    dataset = args.dataset
    output = args.output

    try:
        # 1. Detect format
        fmt = detect_format(dataset)
        print(f"[detect_format] Detected format: {fmt}")

        # 2. Validate annotations
        report = AnnotationValidator(dataset).check()
        print(f"[validate] Found {len(report.issues)} issue(s)")

        # 3. Fix annotations
        fixed_dir = output + "/fixed"
        AnnotationFixer(dataset).fix_all(fixed_dir)
        print(f"[fix] Fixed annotations written to: {fixed_dir}")

        # 4. Convert to YOLO
        yolo_dir = output + "/yolo"
        if fmt == "yolo":
            print("[convert] Source already in YOLO format — skipping conversion.")
            yolo_dir = dataset
        else:
            auto_detect_and_convert(dataset, "yolo", yolo_dir)
            print(f"[convert] Converted to YOLO: {yolo_dir}")

        # 5. Generate synthetic crack samples
        augmented_dir = output + "/augmented"
        images_dir = yolo_dir + "/images"
        SyntheticDefectGenerator(seed=42).add_cracks(
            clean_images_dir=images_dir,
            output_dir=augmented_dir,
            num_samples=100,
        )
        print("[augment] Generated 100 crack samples")

        # 6. Health score
        result = DatasetHealthScore(yolo_dir).run()
        print(f"[health] Score: {result.score}/100")

        print(f"\nPipeline complete. Ready to train YOLOv8 on: {augmented_dir}")

    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print("Hint: check that --dataset points to a directory with annotation files.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
