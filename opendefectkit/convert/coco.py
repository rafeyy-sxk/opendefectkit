"""COCO JSON format read/write utilities."""
import json
from pathlib import Path
from typing import Dict, List, Tuple

from rich.console import Console

from .base import Annotation, BoundingBox

console = Console()


def read_coco(ann_path: str) -> Tuple[List[Annotation], List[str]]:
    """Read a COCO JSON file and return (annotations, class_names)."""
    path = Path(ann_path)
    data = json.loads(path.read_text())

    id_to_image: Dict[int, dict] = {img["id"]: img for img in data["images"]}
    id_to_cat: Dict[int, str] = {cat["id"]: cat["name"] for cat in data["categories"]}
    class_names: List[str] = [cat["name"] for cat in data["categories"]]

    ann_by_image: Dict[int, List[dict]] = {img["id"]: [] for img in data["images"]}
    for ann in data.get("annotations", []):
        image_id = ann["image_id"]
        if image_id in ann_by_image:
            ann_by_image[image_id].append(ann)

    annotations: List[Annotation] = []
    for img_id, img_info in id_to_image.items():
        boxes: List[BoundingBox] = []
        for ann in ann_by_image.get(img_id, []):
            x, y, w, h = ann["bbox"]
            label = id_to_cat.get(ann["category_id"], str(ann["category_id"]))
            boxes.append(BoundingBox(
                x_min=float(x),
                y_min=float(y),
                x_max=float(x + w),
                y_max=float(y + h),
                label=label,
            ))
        annotations.append(Annotation(
            image_path=img_info["file_name"],
            image_width=img_info["width"],
            image_height=img_info["height"],
            boxes=boxes,
        ))

    return annotations, class_names


def write_coco(annotations: List[Annotation], class_names: List[str], output_path: str) -> None:
    """Write annotations to a COCO JSON file."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    name_to_id: Dict[str, int] = {name: idx for idx, name in enumerate(class_names)}

    images = []
    ann_list = []
    ann_id = 1

    for img_id, ann in enumerate(annotations, start=1):
        images.append({
            "id": img_id,
            "file_name": ann.image_path,
            "width": ann.image_width,
            "height": ann.image_height,
        })
        for box in ann.boxes:
            cat_id = name_to_id.get(box.label)
            if cat_id is None:
                name_to_id[box.label] = len(name_to_id)
                class_names.append(box.label)
                cat_id = name_to_id[box.label]
            w = box.x_max - box.x_min
            h = box.y_max - box.y_min
            ann_list.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [box.x_min, box.y_min, w, h],
                "area": w * h,
                "iscrowd": 0,
            })
            ann_id += 1

    categories = [{"id": idx, "name": name} for name, idx in sorted(name_to_id.items(), key=lambda x: x[1])]

    coco_data = {
        "images": images,
        "annotations": ann_list,
        "categories": categories,
    }
    out.write_text(json.dumps(coco_data, indent=2))
