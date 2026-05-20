"""ONNX export utilities for YOLOv8 models."""
import shutil
import warnings
from pathlib import Path
from typing import Tuple


class ONNXExporter:
    """Export YOLOv8 models to ONNX format with optional optimization."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    def export(
        self,
        output_path: str,
        input_size: Tuple[int, int] = (640, 640),
        optimize_for: str = "edge",
        quantize: bool = False,
        validate_export: bool = True,
    ) -> None:
        """Export a YOLOv8 model to ONNX format with optional optimization."""
        valid_targets = ("edge", "mobile", "server")
        if optimize_for not in valid_targets:
            raise ValueError(
                f"optimize_for must be one of {valid_targets}, got '{optimize_for}'"
            )

        try:
            from ultralytics import YOLO  # noqa: F401
        except ImportError:
            raise ImportError(
                "ONNXExporter requires ultralytics. "
                "Install with: pip install opendefectkit[benchmark]"
            )

        model = YOLO(self.model_path)

        # ultralytics saves the .onnx next to the source model
        model.export(format="onnx", imgsz=input_size, simplify=True)

        # Determine where ultralytics placed the exported file
        source_onnx = Path(self.model_path).with_suffix(".onnx")
        dest = Path(output_path)

        if source_onnx.resolve() != dest.resolve():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source_onnx), str(dest))

        # Annotate metadata for optimize_for target
        try:
            import onnx

            onnx_model = onnx.load(str(dest))
            meta = onnx_model.metadata_props.add()
            meta.key = "optimize_for"
            meta.value = optimize_for

            # Optimization-level notes (ultralytics handles graph opts internally)
            if optimize_for == "edge":
                note = onnx_model.metadata_props.add()
                note.key = "optimization_level"
                note.value = "maximum"
            elif optimize_for == "mobile":
                note = onnx_model.metadata_props.add()
                note.key = "optimization_level"
                note.value = "basic"

            onnx.save(onnx_model, str(dest))
        except ImportError:
            warnings.warn(
                "onnx not installed — skipping metadata annotation. "
                "Install with: pip install opendefectkit[deploy]",
                stacklevel=2,
            )

        if quantize:
            try:
                from onnxruntime.quantization import QuantType, quantize_dynamic

                quantize_dynamic(str(dest), str(dest), weight_type=QuantType.QInt8)
            except ImportError:
                try:
                    from rich.console import Console

                    Console().print(
                        "[yellow]Warning:[/yellow] onnxruntime not installed — "
                        "skipping INT8 quantization. "
                        "Install with: pip install opendefectkit[deploy]"
                    )
                except ImportError:
                    warnings.warn(
                        "onnxruntime not installed — skipping INT8 quantization.",
                        stacklevel=2,
                    )

        if validate_export:
            try:
                import onnx

                onnx_model = onnx.load(str(dest))
                onnx.checker.check_model(onnx_model)
            except ImportError:
                try:
                    from rich.console import Console

                    Console().print(
                        "[yellow]Warning:[/yellow] onnx not installed — "
                        "skipping export validation. "
                        "Install with: pip install opendefectkit[deploy]"
                    )
                except ImportError:
                    warnings.warn(
                        "onnx not installed — skipping export validation.",
                        stacklevel=2,
                    )
