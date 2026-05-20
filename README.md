# OpenDefectKit

> Industrial Computer Vision Infrastructure for Python

[![PyPI](https://img.shields.io/pypi/v/opendefectkit)](https://pypi.org/project/opendefectkit/)
[![Python](https://img.shields.io/pypi/pyversions/opendefectkit)](https://pypi.org/project/opendefectkit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()

**OpenDefectKit** is the missing infrastructure layer for industrial defect detection AI.

Every factory AI engineer rebuilds the same pipelines — annotation converters, augmentation scripts, dataset validators, edge exporters. OpenDefectKit ships all of it as a single pip install.

## Install

```bash
pip install opendefectkit                    # core
pip install opendefectkit[benchmark]         # + model evaluation (requires PyTorch)
pip install opendefectkit[deploy]            # + ONNX edge export
pip install opendefectkit[all]               # everything
```

## Quick Start

```python
from opendefectkit.pipeline import DefectPipeline

pipeline = DefectPipeline(
    raw_data_dir="data/raw_annotations",
    output_dir="data/processed",
)
pipeline.run(steps=[
    "detect_format",
    "validate_annotations",
    "fix_annotations",
    "convert_to_yolo",
    "profile_dataset",
    "export_report",
])
# HTML report written to data/processed/dataset_report.html
```

## Modules

| Module | What it does | Key classes |
|---|---|---|
| `opendefectkit.convert` | Convert annotations between COCO, YOLO, VOC, LabelMe, CSV | `detect_format`, `auto_detect_and_convert`, `convert_with_label_map`, `coco_to_yolo` |
| `opendefectkit.validate` | Find and fix corrupt, OOB, duplicate, and missing annotations | `AnnotationValidator`, `AnnotationFixer` |
| `opendefectkit.augment` | Synthetic defect generation and industrial augmentation | `SyntheticDefectGenerator`, `IndustrialAugPipeline` |
| `opendefectkit.analyze` | Dataset profiling, health scoring, and HTML reports | `DatasetProfiler`, `DatasetHealthScore`, `DatasetVisualizer`, `SeverityScorer` |
| `opendefectkit.taxonomy` | 35-type defect registry with label normalization | `DefectTaxonomy`, `DefectType` |
| `opendefectkit.benchmark` | Model evaluation, multi-model comparison, industry reports | `DefectBenchmark`, `ModelComparison`, `IndustryReportGenerator` |
| `opendefectkit.deploy` | ONNX export and edge device optimization | `ONNXExporter`, `EdgeOptimizer`, `DEVICE_PROFILES` |
| `opendefectkit.pipeline` | One-command orchestrator for the full workflow | `DefectPipeline` |

## CLI

```bash
# Detect annotation format
opendefectkit convert --input data/annotations/ --target yolo --output data/yolo/

# Profile a dataset
opendefectkit analyze --dataset data/yolo/ --output report.html

# Validate annotations
opendefectkit validate --dataset data/yolo/

# Generate synthetic defects
opendefectkit augment --images data/clean/ --defect crack --samples 200 --output data/augmented/

# Benchmark a model
opendefectkit benchmark --model runs/train/best.pt --dataset data/test/ --output benchmark.json
```

## Defect Taxonomy

35 defect types across 6 categories: Surface Cracks, Corrosion, Surface Defects, Weld Defects, Coating Defects, Dimensional.

```python
from opendefectkit.taxonomy import DefectTaxonomy

tax = DefectTaxonomy()

# Normalize raw labels from any source dataset
mapping = tax.map_labels(["rusty", "fracture", "pore", "undercutting"], method="fuzzy_match")
# {"rusty": "surface_rust", "fracture": "structural_crack", "pore": "porosity", "undercutting": "undercut"}

defect = tax.standardize("rust spot")
print(defect.id, defect.severity_class)  # CO-001  medium

weld_defects = tax.list_defects("Weld Defects")
# [DefectType(id='WD-001', name='porosity', ...), ...]
```

## Synthetic Augmentation

```python
from opendefectkit.augment import SyntheticDefectGenerator, IndustrialAugPipeline

# Generate crack images with paired YOLO labels
gen = SyntheticDefectGenerator(seed=42)
gen.add_cracks(
    clean_images_dir="data/clean/",
    output_dir="data/synthetic/cracks/",
    num_samples=500,
    crack_types=["hairline", "structural", "fatigue"],
    severity_range=(0.1, 0.9),
)

# Apply industrial augmentation transforms
pipeline = IndustrialAugPipeline(
    transforms=["random_lighting", "motion_blur", "industrial_noise", "jpeg_compression"]
)
pipeline.run(
    input_dir="data/synthetic/cracks/",
    output_dir="data/augmented/",
    multiplier=5,  # 5 augmented versions per image
)
```

Available transforms: `random_lighting`, `motion_blur`, `jpeg_compression`, `perspective_warp`, `industrial_noise`, `surface_reflection`.

## Dataset Health Score

```python
from opendefectkit.analyze import DatasetHealthScore

result = DatasetHealthScore("data/yolo/").run()
print(result)
# Dataset Health Score: 75/100
#
#   [PASS] (20pt) Min samples per class >= 100
#          Min samples: 312 check
#   [FAIL] (15pt) Validation split exists
#          No val/ or valid/ subdirectory
#   [PASS] (10pt) Duplicate images < 1%
#          Duplicates: 2/500 (0.4%)
#   [WARN] (20pt) Class imbalance ratio < 10:1
#          Imbalance ratio: 8.3:1 (need < 10:1)
#   ...
# Recommendation: To improve dataset quality: create a val/ subdirectory.
```

## Edge Deployment

```python
from opendefectkit.deploy import ONNXExporter, EdgeOptimizer, DEVICE_PROFILES

# Export YOLOv8 model to ONNX
exporter = ONNXExporter("runs/train/best.pt")
exporter.export(
    output_path="deploy/model.onnx",
    input_size=(640, 640),
    optimize_for="edge",
    quantize=True,
)

# Get device-specific recommendations
optimizer = EdgeOptimizer("deploy/model.onnx")
result = optimizer.profile_device("jetson_nano").optimize(target_latency_ms=33.0)
print(result.recommendations)
# ["Recommended input size: (416, 416)", "Target FPS: 30.0"]

optimizer.generate_deployment_package("deploy/package/")
# Creates deploy/package/deployment_package.zip with model + inference script + requirements.txt
```

Available device profiles: `jetson_nano`, `jetson_orin`, `raspberry_pi_4`, `intel_nuc`, `generic_x86`.

## Who Uses This

Factory AI engineers who are tired of rebuilding:
- Annotation format converters for every new dataset
- Synthetic defect generators when real defect data is scarce
- Dataset validators that find the corrupt annotations before training
- Edge exporters that work with Jetson Nano and Raspberry Pi

## Contributing

Fork the repo. To add defect types, extend the `DEFECT_REGISTRY` list in `opendefectkit/taxonomy/defects.py` following the existing `DefectType` dataclass pattern. To add format support, implement `read_<format>` in `opendefectkit/convert/` and wire it into `_converters.py`. All contributions require tests in `tests/`. PR welcome.

## License

MIT
