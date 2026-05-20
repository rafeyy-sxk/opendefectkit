"""Pascal VOC XML format read/write utilities."""
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple

from rich.console import Console

from .base import Annotation, BoundingBox

console = Console()


def read_voc(input_dir: str) -> Tuple[List[Annotation], List[str]]:
    """Read Pascal VOC XML files from a directory and return (annotations, class_names)."""
    dir_path = Path(input_dir)
    class_set: List[str] = []
    annotations: List[Annotation] = []

    for xml_file in sorted(dir_path.glob("*.xml")):
        try:
            tree = ET.parse(str(xml_file))
        except ET.ParseError as exc:
            console.print(f"[yellow]Warning:[/yellow] skipping malformed XML {xml_file}: {exc}")
            continue

        root = tree.getroot()
        if root.tag != "annotation":
            console.print(f"[yellow]Warning:[/yellow] {xml_file} root is '{root.tag}', expected 'annotation'. Skipping.")
            continue

        filename_el = root.find("filename")
        filename = (filename_el.text or (xml_file.stem + ".jpg")) if filename_el is not None else xml_file.stem + ".jpg"

        size_el = root.find("size")
        if size_el is not None:
            width = int(size_el.findtext("width", "640"))
            height = int(size_el.findtext("height", "480"))
        else:
            width, height = 640, 480

        boxes: List[BoundingBox] = []
        for obj in root.findall("object"):
            name_el = obj.find("name")
            bndbox = obj.find("bndbox")
            if name_el is None or bndbox is None:
                continue
            label = name_el.text or "unknown"
            if label not in class_set:
                class_set.append(label)
            x_min = float(bndbox.findtext("xmin", "0"))
            y_min = float(bndbox.findtext("ymin", "0"))
            x_max = float(bndbox.findtext("xmax", "0"))
            y_max = float(bndbox.findtext("ymax", "0"))
            boxes.append(BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, label=label))

        annotations.append(Annotation(
            image_path=filename,
            image_width=width,
            image_height=height,
            boxes=boxes,
        ))

    return annotations, class_set


def write_voc(annotations: List[Annotation], output_dir: str, folder: str = "images") -> None:
    """Write Pascal VOC XML files (one per image) to output_dir."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for ann in annotations:
        filename = Path(ann.image_path).name
        stem = Path(ann.image_path).stem

        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = folder
        ET.SubElement(root, "filename").text = filename

        size_el = ET.SubElement(root, "size")
        ET.SubElement(size_el, "width").text = str(ann.image_width)
        ET.SubElement(size_el, "height").text = str(ann.image_height)
        ET.SubElement(size_el, "depth").text = "3"

        for box in ann.boxes:
            obj_el = ET.SubElement(root, "object")
            ET.SubElement(obj_el, "name").text = box.label
            ET.SubElement(obj_el, "pose").text = "Unspecified"
            ET.SubElement(obj_el, "truncated").text = "0"
            ET.SubElement(obj_el, "difficult").text = "0"
            bndbox = ET.SubElement(obj_el, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(int(box.x_min))
            ET.SubElement(bndbox, "ymin").text = str(int(box.y_min))
            ET.SubElement(bndbox, "xmax").text = str(int(box.x_max))
            ET.SubElement(bndbox, "ymax").text = str(int(box.y_max))

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(str(out / f"{stem}.xml"), encoding="unicode", xml_declaration=False)
