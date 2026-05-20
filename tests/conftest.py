"""Shared test fixtures — generates synthetic images and annotation files in-memory."""
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_image(tmp_dir):
    """640x480 blank grey image saved to disk."""
    img = np.full((480, 640, 3), 180, dtype=np.uint8)
    path = tmp_dir / "test_image.jpg"
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture
def sample_images_dir(tmp_dir):
    """Directory with 5 synthetic 640x480 images."""
    images_dir = tmp_dir / "images"
    images_dir.mkdir()
    for i in range(5):
        img = np.full((480, 640, 3), 150 + i * 10, dtype=np.uint8)
        cv2.imwrite(str(images_dir / f"img_{i:03d}.jpg"), img)
    return images_dir


@pytest.fixture
def coco_dataset(tmp_dir):
    """Minimal COCO JSON dataset with 3 images and annotations."""
    images_dir = tmp_dir / "images"
    images_dir.mkdir()
    ann_dir = tmp_dir / "annotations"
    ann_dir.mkdir()

    coco = {
        "images": [
            {"id": 1, "file_name": "img_001.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "img_002.jpg", "width": 640, "height": 480},
            {"id": 3, "file_name": "img_003.jpg", "width": 640, "height": 480},
        ],
        "categories": [
            {"id": 0, "name": "crack"},
            {"id": 1, "name": "rust"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 0, "bbox": [100, 100, 200, 150], "area": 30000},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [50, 50, 100, 80], "area": 8000},
            {"id": 3, "image_id": 3, "category_id": 0, "bbox": [300, 200, 150, 100], "area": 15000},
        ],
    }
    for img_info in coco["images"]:
        img = np.full((img_info["height"], img_info["width"], 3), 160, dtype=np.uint8)
        cv2.imwrite(str(images_dir / img_info["file_name"]), img)

    ann_path = ann_dir / "annotations.json"
    ann_path.write_text(json.dumps(coco))
    return {"images_dir": images_dir, "ann_path": ann_path, "coco": coco}


@pytest.fixture
def yolo_dataset(tmp_dir):
    """Minimal YOLO dataset with labels directory."""
    images_dir = tmp_dir / "images"
    labels_dir = tmp_dir / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    classes = ["crack", "rust"]
    classes_path = tmp_dir / "classes.txt"
    classes_path.write_text("\n".join(classes))

    for i in range(3):
        img = np.full((480, 640, 3), 160, dtype=np.uint8)
        cv2.imwrite(str(images_dir / f"img_{i:03d}.jpg"), img)
        label_content = "0 0.5 0.5 0.3 0.25\n1 0.2 0.3 0.15 0.1\n"
        (labels_dir / f"img_{i:03d}.txt").write_text(label_content)

    return {"images_dir": images_dir, "labels_dir": labels_dir, "classes_path": classes_path}
