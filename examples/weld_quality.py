"""
Weld quality inspection example.

Shows: COCO dataset conversion -> weld defect taxonomy -> annotation validation -> model benchmarking setup.

Usage:
    python examples/weld_quality.py --coco-json path/to/annotations.json --images path/to/images --output path/to/output
"""
import argparse
import sys

from opendefectkit.convert import detect_format, coco_to_yolo, convert_with_label_map
from opendefectkit.taxonomy import DefectTaxonomy
from opendefectkit.validate import AnnotationValidator


def main() -> None:
    parser = argparse.ArgumentParser(description="Weld quality inspection pipeline")
    parser.add_argument("--coco-json", required=True, dest="coco_json", help="COCO annotations.json file")
    parser.add_argument("--images", required=True, help="Directory containing the source images")
    parser.add_argument(
        "--output", default="output/weld_qc", help="Output directory (default: output/weld_qc)"
    )
    args = parser.parse_args()

    coco_json = args.coco_json
    images = args.images
    output = args.output

    try:
        # 1. Detect format
        fmt = detect_format(coco_json)
        print(f"[detect_format] Detected: {fmt}")

        # 2. Convert COCO -> YOLO
        labels_dir = output + "/labels"
        coco_to_yolo(coco_json, labels_dir, images)
        print(f"[convert] Converted to YOLO: {labels_dir}")

        # 3. List weld defect types from taxonomy
        tax = DefectTaxonomy()
        weld_defects = tax.list_defects("Weld Defects")
        print(f"[taxonomy] Weld defect types: {[d.name for d in weld_defects]}")

        # 4. Normalize labels with alias mapping
        normalized_dir = output + "/normalized"
        convert_with_label_map(
            input_path=labels_dir,
            output_dir=normalized_dir,
            label_map={
                "porosity": ["pore", "gas_pocket"],
                "undercut": ["undercutting"],
            },
        )
        print(f"[label_map] Labels normalized: {normalized_dir}")

        # 5. Validate normalized annotations
        report = AnnotationValidator(normalized_dir).check()
        counts = report.count_by_type()
        if counts:
            print(f"[validate] Issues found: {counts}")
        else:
            print("[validate] No annotation issues found.")

        print(
            "\nNext steps:"
            "\n  1. Train:     yolo train data={output}/normalized model=yolov8s.pt epochs=100"
            "\n  2. Benchmark: from opendefectkit.benchmark import DefectBenchmark"
            "\n                DefectBenchmark('runs/train/best.pt').run('{output}/normalized')"
            "\n  3. Export:    from opendefectkit.deploy import ONNXExporter"
            "\n                ONNXExporter('runs/train/best.pt').export('{output}/model.onnx')"
        )

    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print("Hint: check that --coco-json and --images paths exist.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
