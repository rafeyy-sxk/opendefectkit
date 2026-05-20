"""Synthetic defect generation and industrial augmentation pipeline."""
from __future__ import annotations

import shutil
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from rich.progress import track

from .noise import industrial_noise, jpeg_compression, motion_blur, random_lighting
from .surface import perspective_warp, surface_reflection


# ---------------------------------------------------------------------------
# Internal drawing helpers
# ---------------------------------------------------------------------------

def _draw_crack(
    img: np.ndarray,
    severity: float,
    crack_type: str,
    rng: np.random.RandomState,
) -> Tuple[int, int, int, int]:
    """Draw a synthetic crack on *img* in-place; return (x_min, y_min, x_max, y_max)."""
    h, w = img.shape[:2]
    num_ctrl = rng.randint(3, 7)
    start_x = int(rng.uniform(0.1, 0.9) * w)
    start_y = int(rng.uniform(0.1, 0.9) * h)

    color = (int(rng.uniform(10, 30)),) * 3  # very dark BGR

    all_pts: List[np.ndarray] = []

    def _make_path(sx: int, sy: int, n: int) -> np.ndarray:
        pts = [(sx, sy)]
        for _ in range(n):
            dx = int(rng.uniform(-60, 60) * severity + rng.uniform(-20, 20))
            dy = int(rng.uniform(-60, 60) * severity + rng.uniform(-20, 20))
            nx_ = int(np.clip(pts[-1][0] + dx, 0, w - 1))
            ny_ = int(np.clip(pts[-1][1] + dy, 0, h - 1))
            pts.append((nx_, ny_))
        return np.array(pts, dtype=np.int32).reshape(-1, 1, 2)

    if crack_type == "hairline":
        thickness = 1
        path = _make_path(start_x, start_y, num_ctrl)
        cv2.polylines(img, [path], False, color, thickness)
        all_pts.append(path)

    elif crack_type == "structural":
        thickness = int(np.clip(severity * 5, 2, 5))
        num_branches = rng.randint(2, 4)
        for _ in range(num_branches):
            path = _make_path(start_x, start_y, num_ctrl)
            cv2.polylines(img, [path], False, color, thickness)
            all_pts.append(path)

    else:  # "fatigue" or default
        thickness = max(1, int(severity * 4))
        path = _make_path(start_x, start_y, num_ctrl)
        # approximate a gentle curve by drawing smaller segments
        pts_flat = path.reshape(-1, 2)
        smooth_pts = []
        for i in range(len(pts_flat) - 1):
            mid = ((pts_flat[i] + pts_flat[i + 1]) / 2).astype(int)
            smooth_pts.append(pts_flat[i])
            smooth_pts.append(mid)
        smooth_pts.append(pts_flat[-1])
        curved = np.array(smooth_pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [curved], False, color, thickness)
        all_pts.append(curved)

    # Compute bounding box over all drawn points
    all_coords = np.concatenate([p.reshape(-1, 2) for p in all_pts], axis=0)
    x_min = int(all_coords[:, 0].min())
    y_min = int(all_coords[:, 1].min())
    x_max = int(all_coords[:, 0].max())
    y_max = int(all_coords[:, 1].max())
    return x_min, y_min, x_max, y_max


def _draw_rust(
    img: np.ndarray,
    coverage: float,
    rng: np.random.RandomState,
) -> Tuple[int, int, int, int]:
    """Overlay rust coloring on *img* in-place; return bounding box."""
    h, w = img.shape[:2]
    raw = rng.normal(0, 1, (h, w)).astype(np.float32)
    threshold = 1.0 - coverage
    mask = (raw > threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.dilate(mask, kernel, iterations=2)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    rust_hue = float(rng.uniform(10, 25))
    hsv[:, :, 0] = np.where(mask, rust_hue, hsv[:, :, 0])
    hsv[:, :, 1] = np.where(mask, np.clip(hsv[:, :, 1] + 80, 0, 255), hsv[:, :, 1])
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    rust_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    img[:] = rust_bgr

    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, 0, w - 1, h - 1
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _draw_scratch(
    img: np.ndarray,
    depth_type: str,
    rng: np.random.RandomState,
) -> Tuple[int, int, int, int]:
    """Draw a scratch on *img* in-place; return bounding box."""
    h, w = img.shape[:2]
    angle = float(rng.uniform(0, 360))
    rad = np.deg2rad(angle)
    cx = int(rng.uniform(0.1, 0.9) * w)
    cy = int(rng.uniform(0.1, 0.9) * h)

    if depth_type == "micro":
        length = int(rng.uniform(10, 30))
        thickness = 1
        darkness = 40
    elif depth_type == "deep":
        length = int(rng.uniform(100, 300))
        thickness = rng.randint(2, 4)
        darkness = 120
    else:  # "surface"
        length = int(rng.uniform(50, 150))
        thickness = rng.randint(1, 3)
        darkness = 70

    dx = int(np.cos(rad) * length / 2)
    dy = int(np.sin(rad) * length / 2)
    x1 = int(np.clip(cx - dx, 0, w - 1))
    y1 = int(np.clip(cy - dy, 0, h - 1))
    x2 = int(np.clip(cx + dx, 0, w - 1))
    y2 = int(np.clip(cy + dy, 0, h - 1))

    # darken along the scratch
    scratch_color_val = int(np.clip(img[cy, cx, 0] - darkness, 0, 255))
    scratch_color = (scratch_color_val,) * 3
    cv2.line(img, (x1, y1), (x2, y2), scratch_color, thickness)

    if depth_type == "deep":
        # white highlight on edge
        highlight = (min(255, scratch_color_val + 100),) * 3
        cv2.line(img, (x1 + 1, y1 + 1), (x2 + 1, y2 + 1), highlight, 1)

    # slight Gaussian blur around scratch region
    pad = max(thickness + 2, 3)
    rx1, ry1 = max(0, min(x1, x2) - pad), max(0, min(y1, y2) - pad)
    rx2, ry2 = min(w, max(x1, x2) + pad), min(h, max(y1, y2) + pad)
    region = img[ry1:ry2, rx1:rx2]
    if region.size > 0:
        img[ry1:ry2, rx1:rx2] = cv2.GaussianBlur(region, (3, 3), 0)

    x_min = min(x1, x2)
    y_min = min(y1, y2)
    x_max = max(x1, x2)
    y_max = max(y1, y2)
    return x_min, y_min, x_max, y_max


# ---------------------------------------------------------------------------
# Helper: write YOLO label
# ---------------------------------------------------------------------------

def _write_yolo_label(
    label_path: Path,
    bbox: Tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    class_id: int = 0,
) -> None:
    x_min, y_min, x_max, y_max = bbox
    # clamp
    x_min = max(0, min(x_min, img_w - 1))
    y_min = max(0, min(y_min, img_h - 1))
    x_max = max(0, min(x_max, img_w - 1))
    y_max = max(0, min(y_max, img_h - 1))
    # ensure non-zero size
    if x_max <= x_min:
        x_max = min(x_min + 1, img_w - 1)
    if y_max <= y_min:
        y_max = min(y_min + 1, img_h - 1)
    cx = ((x_min + x_max) / 2) / img_w
    cy = ((y_min + y_max) / 2) / img_h
    bw = (x_max - x_min) / img_w
    bh = (y_max - y_min) / img_h
    label_path.write_text(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


# ---------------------------------------------------------------------------
# SyntheticDefectGenerator
# ---------------------------------------------------------------------------

class SyntheticDefectGenerator:
    """Generate synthetic defect images with paired YOLO annotations."""

    def __init__(self, seed: int = 42) -> None:
        """Initialise with a fixed random seed for reproducibility."""
        self._rng = np.random.RandomState(seed)

    def _setup_output_dirs(self, output_dir: str) -> Tuple[Path, Path]:
        out = Path(output_dir)
        images_out = out / "images"
        labels_out = out / "labels"
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)
        return images_out, labels_out

    def _collect_sources(self, clean_images_dir: str) -> List[Path]:
        src = Path(clean_images_dir)
        sources = list(src.glob("*.jpg")) + list(src.glob("*.jpeg")) + list(src.glob("*.png"))
        if not sources:
            raise FileNotFoundError(f"No images found in {clean_images_dir}")
        return sources

    def _sample_sources(self, sources: List[Path], num_samples: int) -> List[Path]:
        indices = self._rng.randint(0, len(sources), num_samples)
        return [sources[i] for i in indices]

    def add_cracks(
        self,
        clean_images_dir: str,
        output_dir: str,
        num_samples: int,
        crack_types: Optional[List[str]] = None,
        severity_range: Tuple[float, float] = (0.1, 0.9),
    ) -> None:
        """Generate images with synthetic cracks and YOLO labels."""
        if crack_types is None:
            crack_types = ["hairline", "structural", "fatigue"]
        images_out, labels_out = self._setup_output_dirs(output_dir)
        sources = self._collect_sources(clean_images_dir)
        sampled = self._sample_sources(sources, num_samples)

        (Path(output_dir) / "classes.txt").write_text("crack\n")

        for i, src_path in enumerate(sampled):
            img = cv2.imread(str(src_path))
            if img is None:
                continue
            severity = float(self._rng.uniform(*severity_range))
            ctype = crack_types[int(self._rng.randint(0, len(crack_types)))]
            h, w = img.shape[:2]
            bbox = _draw_crack(img, severity, ctype, self._rng)
            stem = f"{src_path.stem}_crack_{i}"
            cv2.imwrite(str(images_out / f"{stem}.jpg"), img)
            _write_yolo_label(labels_out / f"{stem}.txt", bbox, w, h)

    def add_rust(
        self,
        clean_images_dir: str,
        output_dir: str,
        num_samples: int,
        coverage_range: Tuple[float, float] = (0.05, 0.4),
    ) -> None:
        """Generate images with synthetic rust patches and YOLO labels."""
        images_out, labels_out = self._setup_output_dirs(output_dir)
        sources = self._collect_sources(clean_images_dir)
        sampled = self._sample_sources(sources, num_samples)

        (Path(output_dir) / "classes.txt").write_text("rust\n")

        for i, src_path in enumerate(sampled):
            img = cv2.imread(str(src_path))
            if img is None:
                continue
            coverage = float(self._rng.uniform(*coverage_range))
            h, w = img.shape[:2]
            bbox = _draw_rust(img, coverage, self._rng)
            stem = f"{src_path.stem}_rust_{i}"
            cv2.imwrite(str(images_out / f"{stem}.jpg"), img)
            _write_yolo_label(labels_out / f"{stem}.txt", bbox, w, h)

    def add_scratches(
        self,
        clean_images_dir: str,
        output_dir: str,
        num_samples: int,
        depth_types: Optional[List[str]] = None,
    ) -> None:
        """Generate images with synthetic scratches and YOLO labels."""
        if depth_types is None:
            depth_types = ["surface", "deep", "micro"]
        images_out, labels_out = self._setup_output_dirs(output_dir)
        sources = self._collect_sources(clean_images_dir)
        sampled = self._sample_sources(sources, num_samples)

        (Path(output_dir) / "classes.txt").write_text("scratch\n")

        for i, src_path in enumerate(sampled):
            img = cv2.imread(str(src_path))
            if img is None:
                continue
            dtype = depth_types[int(self._rng.randint(0, len(depth_types)))]
            h, w = img.shape[:2]
            bbox = _draw_scratch(img, dtype, self._rng)
            stem = f"{src_path.stem}_scratch_{i}"
            cv2.imwrite(str(images_out / f"{stem}.jpg"), img)
            _write_yolo_label(labels_out / f"{stem}.txt", bbox, w, h)


# ---------------------------------------------------------------------------
# IndustrialAugPipeline
# ---------------------------------------------------------------------------

_TRANSFORM_FN = {
    "random_lighting": random_lighting,
    "motion_blur": motion_blur,
    "jpeg_compression": jpeg_compression,
    "perspective_warp": perspective_warp,
    "industrial_noise": industrial_noise,
    "surface_reflection": surface_reflection,
}


class IndustrialAugPipeline:
    """Apply a chain of industrial augmentation transforms to a directory of images."""

    AVAILABLE_TRANSFORMS: List[str] = list(_TRANSFORM_FN.keys())

    def __init__(self, transforms: List[str]) -> None:
        """Validate and store the list of transform names."""
        invalid = [t for t in transforms if t not in _TRANSFORM_FN]
        if invalid:
            raise ValueError(
                f"Unknown transform(s): {invalid}. "
                f"Available: {self.AVAILABLE_TRANSFORMS}"
            )
        self._transforms = transforms
        self._spatial = {"perspective_warp"}

    def run(
        self,
        input_dir: str,
        output_dir: str,
        multiplier: int = 5,
    ) -> None:
        """Augment every image in input_dir, writing multiplier versions each."""
        in_path = Path(input_dir)
        out_path = Path(output_dir)
        images_out = out_path / "images"
        labels_out = out_path / "labels"
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        sources = (
            list(in_path.glob("*.jpg"))
            + list(in_path.glob("*.jpeg"))
            + list(in_path.glob("*.png"))
        )

        # Check whether any spatial transforms are used
        has_spatial = any(t in self._spatial for t in self._transforms)
        if has_spatial:
            warnings.warn(
                "Pipeline contains spatial transforms (perspective_warp). "
                "Labels are copied unchanged — bounding box accuracy may be reduced.",
                UserWarning,
                stacklevel=2,
            )

        # Possible label locations
        in_labels = in_path / "labels"

        items = [(src, v) for src in sources for v in range(multiplier)]
        for src, v in track(items, description="Augmenting"):
            img = cv2.imread(str(src))
            if img is None:
                continue
            for t in self._transforms:
                img = _TRANSFORM_FN[t](img)
            stem = f"{src.stem}_aug_{v}"
            cv2.imwrite(str(images_out / f"{stem}.jpg"), img)

            # Copy label if it exists
            label_src = in_labels / f"{src.stem}.txt"
            if label_src.exists():
                shutil.copy(str(label_src), str(labels_out / f"{stem}.txt"))
