"""Industrial noise augmentation transforms."""
from __future__ import annotations

import random

import cv2
import numpy as np


def random_lighting(img: np.ndarray) -> np.ndarray:
    """Apply random brightness and gamma correction."""
    factor = random.uniform(0.5, 1.5)
    out = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    gamma = random.uniform(0.5, 2.0)
    lut = np.array(
        [np.clip(((i / 255.0) ** (1.0 / gamma)) * 255, 0, 255) for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(out, lut)


def motion_blur(img: np.ndarray) -> np.ndarray:
    """Apply motion blur with a random horizontal or diagonal kernel."""
    size = random.choice(range(5, 16, 2))
    diagonal = random.random() < 0.5
    kernel = np.zeros((size, size), dtype=np.float32)
    if diagonal:
        np.fill_diagonal(kernel, 1.0 / size)
    else:
        kernel[size // 2, :] = 1.0 / size
    return cv2.filter2D(img, -1, kernel)


def jpeg_compression(img: np.ndarray) -> np.ndarray:
    """Simulate JPEG compression artefacts."""
    quality = random.randint(20, 60)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buf = cv2.imencode(".jpg", img, encode_param)
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return decoded if decoded is not None else img


def industrial_noise(img: np.ndarray) -> np.ndarray:
    """Add Gaussian noise and salt-and-pepper noise."""
    out = img.astype(np.float32)
    gauss = np.random.normal(0, 15, img.shape).astype(np.float32)
    out = np.clip(out + gauss, 0, 255).astype(np.uint8)
    num_pixels = int(out.size * 0.005 / 3)
    h, w = out.shape[:2]
    ys = np.random.randint(0, h, num_pixels)
    xs = np.random.randint(0, w, num_pixels)
    out[ys, xs] = 0
    ys = np.random.randint(0, h, num_pixels)
    xs = np.random.randint(0, w, num_pixels)
    out[ys, xs] = 255
    return out
