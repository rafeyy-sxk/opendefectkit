"""YOLO format read/write utilities."""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rich.console import Console

from .base import Annotation, BoundingBox

console = Console()

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def _find_classes_file(directory: Path) -> Optional[Path]:
    """Find classes.txt or obj.names in directory."""
    for name in ("classes.txt", "obj.names"):
        p = directory / name
        if p.exists():
            return p
    return None


def read_yolo(
    labels_dir: str,
    classes_source: str,
    image_dir: Optional[str] = None,
    image_width: int = 640,
    image_height: int = 640,
) -> Tuple[List[Annotation], List[str]]:
    """Read YOLO label files and return (annotations, class_names).

    image_dir is used to determine actual image dimensions when provided.
    """
    labels_path = Path(labels_dir)
    classes_path = Path(classes_source)
    class_names: List[str] = [ln.strip() for ln in classes_path.read_text().splitlines() if ln.strip()]

    img_path_map: Dict[str, Path] = {}
    if image_dir:
        img_dir = Path(image_dir)
        for f in img_dir.iterdir():
            if f.suffix.lower() in _IMAGE_EXTENSIONS:
                img_path_map[f.stem] = f

    annotations: List[Annotation] = []
    for label_file in sorted(labels_path.glob("*.txt")):
        stem = label_file.stem
        w, h = image_width, image_height

        img_file: Optional[Path] = img_path_map.get(stem)
        if img_file:
            try:
                import cv2
                img = cv2.imread(str(img_file))
                if img is not None:
                    h, w = img.shape[:2]
            except ImportError:
                pass

        img_name = stem + (img_file.suffix if img_file else ".jpg")
        boxes: List[BoundingBox] = []
        text = label_file.read_text().strip()
        if text:
            for line in text.splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    console.print(f"[yellow]Warning:[/yellow] skipping malformed line in {label_file}: {line!r}")
                    continue
                class_id = int(parts[0])
                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x_min = (cx - bw / 2) * w
                y_min = (cy - bh / 2) * h
                x_max = (cx + bw / 2) * w
                y_max = (cy + bh / 2) * h
                label = class_names[class_id] if class_id < len(class_names) else str(class_id)
                boxes.append(BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, label=label))

        annotations.append(Annotation(
            image_path=img_name,
            image_width=w,
            image_height=h,
            boxes=boxes,
        ))

    return annotations, class_names


def write_yolo(annotations: List[Annotation], class_names: List[str], output_dir: str) -> None:
    """Write YOLO .txt label files and classes.txt to output_dir."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    name_to_id: Dict[str, int] = {name: idx for idx, name in enumerate(class_names)}

    for ann in annotations:
        stem = Path(ann.image_path).stem
        lines: List[str] = []
        for box in ann.boxes:
            if box.label not in name_to_id:
                name_to_id[box.label] = len(name_to_id)
                class_names.append(box.label)
            class_id = name_to_id[box.label]
            cx = (box.x_min + box.x_max) / 2 / ann.image_width
            cy = (box.y_min + box.y_max) / 2 / ann.image_height
            bw = (box.x_max - box.x_min) / ann.image_width
            bh = (box.y_max - box.y_min) / ann.image_height
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (out / f"{stem}.txt").write_text("\n".join(lines))

    classes_txt = out / "classes.txt"
    classes_txt.write_text("\n".join(class_names))
