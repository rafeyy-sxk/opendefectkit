"""Edge optimizer and device profiles for industrial deployment."""
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class DeviceProfile:
    """Hardware profile describing deployment constraints for a target device."""

    name: str
    target_fps: float
    max_model_size_mb: float
    recommended_input_size: Tuple[int, int]
    supports_gpu: bool
    notes: str


DEVICE_PROFILES: Dict[str, DeviceProfile] = {
    "jetson_nano": DeviceProfile(
        name="NVIDIA Jetson Nano",
        target_fps=30.0,
        max_model_size_mb=50.0,
        recommended_input_size=(416, 416),
        supports_gpu=True,
        notes="Use TensorRT export for best performance",
    ),
    "jetson_orin": DeviceProfile(
        name="NVIDIA Jetson Orin",
        target_fps=60.0,
        max_model_size_mb=200.0,
        recommended_input_size=(640, 640),
        supports_gpu=True,
        notes="Supports INT8 TensorRT; full YOLOv8l runs in real-time",
    ),
    "raspberry_pi_4": DeviceProfile(
        name="Raspberry Pi 4",
        target_fps=5.0,
        max_model_size_mb=20.0,
        recommended_input_size=(320, 320),
        supports_gpu=False,
        notes="Use INT8 quantized ONNX; YOLOv8n only",
    ),
    "intel_nuc": DeviceProfile(
        name="Intel NUC",
        target_fps=20.0,
        max_model_size_mb=100.0,
        recommended_input_size=(640, 640),
        supports_gpu=False,
        notes="OpenVINO export recommended for Intel hardware acceleration",
    ),
    "generic_x86": DeviceProfile(
        name="Generic x86 Industrial PC",
        target_fps=10.0,
        max_model_size_mb=150.0,
        recommended_input_size=(640, 640),
        supports_gpu=False,
        notes="Standard ONNX runtime; consider OpenVINO for Intel CPUs",
    ),
}

_INFERENCE_SCRIPT_TEMPLATE = '''\
"""Auto-generated inference script for OpenDefectKit deployment."""
import numpy as np
import onnxruntime as ort
import cv2
from pathlib import Path

MODEL_PATH = "{model_filename}"
INPUT_SIZE = {input_size}


def run_inference(image_path: str):
    session = ort.InferenceSession(MODEL_PATH)
    img = cv2.imread(image_path)
    img = cv2.resize(img, INPUT_SIZE)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[np.newaxis]
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {{input_name: img}})
    return outputs


if __name__ == "__main__":
    import sys
    result = run_inference(sys.argv[1])
    print(f"Inference complete. Output shape: {{result[0].shape}}")
'''

_REQUIREMENTS = "onnxruntime\nnumpy\nopencv-python\nPillow\n"


@dataclass
class OptimizationResult:
    """Result of an ONNX model optimization analysis."""

    original_size_mb: float
    optimized_size_mb: float
    estimated_speedup: float
    recommendations: List[str]


class EdgeOptimizer:
    """Analyze and package ONNX models for edge deployment."""

    def __init__(self, onnx_path: str) -> None:
        self.onnx_path = Path(onnx_path)
        self._profile: Optional[DeviceProfile] = None

    def profile_device(self, device: str) -> "EdgeOptimizer":
        """Set the target device profile. Returns self for chaining."""
        if device not in DEVICE_PROFILES:
            raise ValueError(
                f"Unknown device '{device}'. Available: {list(DEVICE_PROFILES)}"
            )
        self._profile = DEVICE_PROFILES[device]
        return self

    def optimize(
        self,
        target_latency_ms: float = 50.0,  # noqa: ARG002
        accuracy_drop_allowed: float = 0.02,  # noqa: ARG002
    ) -> OptimizationResult:
        """Analyze ONNX model and return optimization recommendations."""
        # Get file size; handle missing file gracefully
        if self.onnx_path.exists():
            original_size_mb = self.onnx_path.stat().st_size / (1024 * 1024)
        else:
            original_size_mb = 0.0

        recommendations: List[str] = []
        quantize_recommended = False

        if self._profile is None:
            recommendations.append(
                f"Model size: {original_size_mb:.2f} MB. "
                "Set a device profile with profile_device() for tailored recommendations."
            )
        else:
            profile = self._profile

            if original_size_mb > profile.max_model_size_mb:
                recommendations.append(
                    f"Model too large for {profile.name}. Consider quantization."
                )

            if not profile.supports_gpu:
                recommendations.append("Use INT8 quantization for CPU deployment")
                quantize_recommended = True

            recommendations.append(
                f"Recommended input size: {profile.recommended_input_size}"
            )
            recommendations.append(f"Target FPS: {profile.target_fps}")

        estimated_speedup = 1.5 if quantize_recommended else 1.0
        optimized_size_mb = original_size_mb * 0.7 if quantize_recommended else original_size_mb

        return OptimizationResult(
            original_size_mb=original_size_mb,
            optimized_size_mb=optimized_size_mb,
            estimated_speedup=estimated_speedup,
            recommendations=recommendations,
        )

    def generate_deployment_package(
        self,
        output_dir: str,
        include_inference_script: bool = True,
        include_requirements: bool = True,
    ) -> None:
        """Create a deployment package zip with model + scripts + requirements."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        files_in_package: List[Path] = []

        # Copy model file if it exists
        if self.onnx_path.exists():
            dest_model = out / self.onnx_path.name
            import shutil

            shutil.copy2(str(self.onnx_path), str(dest_model))
            files_in_package.append(dest_model)

        model_filename = self.onnx_path.name
        input_size = (
            self._profile.recommended_input_size
            if self._profile is not None
            else (640, 640)
        )

        if include_inference_script:
            script_content = _INFERENCE_SCRIPT_TEMPLATE.format(
                model_filename=model_filename,
                input_size=input_size,
            )
            script_path = out / "run_inference.py"
            script_path.write_text(script_content, encoding="utf-8")
            files_in_package.append(script_path)

        if include_requirements:
            req_path = out / "requirements.txt"
            req_path.write_text(_REQUIREMENTS, encoding="utf-8")
            files_in_package.append(req_path)

        zip_path = out / "deployment_package.zip"
        with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in files_in_package:
                zf.write(str(file_path), arcname=file_path.name)
