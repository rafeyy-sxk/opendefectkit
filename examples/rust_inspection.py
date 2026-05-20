"""
Rust inspection pipeline example.

Shows: taxonomy mapping -> synthetic rust generation -> dataset profiling -> severity scoring setup.

Usage:
    python examples/rust_inspection.py --images path/to/clean/images --output path/to/output
"""
import argparse
import sys

from opendefectkit.augment import SyntheticDefectGenerator
from opendefectkit.analyze import DatasetProfiler, DatasetVisualizer
from opendefectkit.taxonomy import DefectTaxonomy


def main() -> None:
    parser = argparse.ArgumentParser(description="Rust/corrosion inspection pipeline")
    parser.add_argument("--images", required=True, help="Directory of clean (defect-free) images")
    parser.add_argument(
        "--output", default="output/rust_inspection", help="Output directory (default: output/rust_inspection)"
    )
    args = parser.parse_args()

    images = args.images
    output = args.output

    try:
        tax = DefectTaxonomy()

        # 1. Map custom labels to taxonomy standard names
        mapping = tax.map_labels(
            ["rusty", "corrosion", "oxidation", "brown_spot"], method="fuzzy_match"
        )
        print(f"[taxonomy] Label mapping: {mapping}")

        # 2. Standardize a single label
        defect = tax.standardize("rust spot")
        print(f"[taxonomy] Standardized 'rust spot' -> {defect.name} (id={defect.id}, severity={defect.severity_class})")

        # 3. Generate synthetic rust samples
        synthetic_dir = output + "/synthetic"
        SyntheticDefectGenerator(seed=0).add_rust(
            clean_images_dir=images,
            output_dir=synthetic_dir,
            num_samples=200,
            coverage_range=(0.05, 0.3),
        )
        print("[augment] Generated 200 rust samples")

        # 4. Profile the synthetic dataset
        profile = DatasetProfiler(synthetic_dir).run()
        print(profile.summary())

        # 5. Generate HTML report
        report_path = output + "/report.html"
        DatasetVisualizer(synthetic_dir).generate_html_report(report_path)
        print(f"[report] Report saved: {report_path}")

        print(
            f"\nTrain: yolo train data={synthetic_dir} model=yolov8n.pt epochs=50"
        )

    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print("Hint: check that --images points to a directory containing .jpg or .png files.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
