"""LabelMe JSON format read/write utilities."""
import json
from pathlib import Path
from typing import List, Tuple

from rich.console import Console

from .base import Annotation, BoundingBox

console = Console()


def read_labelme(input_dir: str) -> Tuple[List[Annotation], List[str]]:
    """Read LabelMe JSON files from a directory and return (annotations, class_names)."""
    dir_path = Path(input_dir)
    class_set: List[str] = []
    annotations: List[Annotation] = []

    for json_file in sorted(dir_path.glob("*.json")):
        try:
            data = json.loads(json_file.read_text())
        except json.JSONDecodeError as exc:
            console.print(f"[yellow]Warning:[/yellow] skipping malformed JSON {json_file}: {exc}")
            continue

        if "shapes" not in data:
            continue

        image_path = data.get("imagePath", json_file.stem + ".jpg")
        image_width = data.get("imageWidth", 640)
        image_height = data.get("imageHeight", 480)

        boxes: List[BoundingBox] = []
        for shape in data["shapes"]:
            shape_type = shape.get("shape_type", "")
            label = shape.get("label", "unknown")
            points = shape.get("points", [])

            if shape_type != "rectangle":
                console.print(
                    f"[yellow]Warning:[/yellow] {json_file.name}: shape_type='{shape_type}' for label='{label}' is not "
                    f"rectangle — skipping."
                )
                continue

            if len(points) < 2:
                console.print(f"[yellow]Warning:[/yellow] {json_file.name}: rectangle has fewer than 2 points — skipping.")
                continue

            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            if label not in class_set:
                class_set.append(label)
            boxes.append(BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, label=label))

        annotations.append(Annotation(
            image_path=image_path,
            image_width=image_width,
            image_height=image_height,
            boxes=boxes,
        ))

    return annotations, class_set
