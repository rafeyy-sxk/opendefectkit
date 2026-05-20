"""Surface-specific augmentation transforms."""
from __future__ import annotations

import random

import cv2
import numpy as np


def perspective_warp(img: np.ndarray) -> np.ndarray:
    """Apply a small random perspective warp (max 5% shift per corner)."""
    h, w = img.shape[:2]
    max_shift = 0.05

    def _jitter() -> Tuple[float, float]:
        return (
            random.uniform(-max_shift * w, max_shift * w),
            random.uniform(-max_shift * h, max_shift * h),
        )

    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    jx = [_jitter() for _ in range(4)]
    dst = np.array(
        [
            [0 + jx[0][0], 0 + jx[0][1]],
            [w + jx[1][0], 0 + jx[1][1]],
            [w + jx[2][0], h + jx[2][1]],
            [0 + jx[3][0], h + jx[3][1]],
        ],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h))


def surface_reflection(img: np.ndarray) -> np.ndarray:
    """Blend a bright elliptical highlight to simulate metallic reflection."""
    h, w = img.shape[:2]
    cx = random.randint(0, w)
    cy = random.randint(0, h)
    rx = random.randint(20, 100)
    ry = random.randint(20, 100)
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
    alpha = random.uniform(0.2, 0.6)
    highlight = np.full_like(img, 255, dtype=np.float32)
    mask3 = np.stack([mask, mask, mask], axis=-1)
    out = img.astype(np.float32) * (1 - mask3 * alpha) + highlight * mask3 * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


# Type alias needed in function signature above
from typing import Tuple  # noqa: E402
