"""High-level conversion functions for the convert module."""
import json
from pathlib import Path
from typing import Dict, List

from rich.console import Console

from .coco import read_coco, write_coco
from .yolo import read_yolo, write_yolo, _find_classes_file
from .voc import read_voc, write_voc
from .labelme import read_labelme

console = Console()


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def detect_format(path: str) -> str:
    """Detect annotation format from a file or directory path.

    Returns one of: 'coco_json', 'yolo', 'voc_xml', 'labelme_json', 'csv'.
    Raises ValueError if format cannot be determined.
    """
    p = Path(path)

    if p.is_file():
        return _detect_file_format(p)

    if p.is_dir():
        return _detect_dir_format(p)

    raise ValueError(f"Path does not exist: {path}")


def _detect_file_format(p: Path) -> str:
    """Detect format from a single file."""
    suffix = p.suffix.lower()

    if suffix == ".json":
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            raise ValueError(f"Cannot parse JSON file: {p}")
        if _is_coco_json(data):
            return "coco_json"
        if "shapes" in data:
            return "labelme_json"
        raise ValueError(f"Cannot determine JSON annotation format for: {p}")

    if suffix == ".xml":
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(str(p))
            if tree.getroot().tag == "annotation":
                return "voc_xml"
        except Exception:
            pass
        raise ValueError(f"Cannot determine XML annotation format for: {p}")

    if suffix == ".txt":
        return _detect_txt_format(p)

    if suffix == ".csv":
        return "csv"

    raise ValueError(f"Cannot determine annotation format for file: {p}")


def _is_coco_json(data: dict) -> bool:
    return all(k in data for k in ("images", "annotations", "categories"))


def _detect_txt_format(p: Path) -> str:
    """Heuristic: 5-column space-separated → YOLO."""
    text = p.read_text().strip()
    if not text:
        return "yolo"
    first_line = text.splitlines()[0].strip().split()
    if len(first_line) == 5:
        try:
            int(first_line[0])
            [float(v) for v in first_line[1:]]
            return "yolo"
        except ValueError:
            pass
    raise ValueError(f"Cannot determine format for .txt file: {p}")


def _detect_dir_format(p: Path) -> str:
    """Detect format by inspecting directory structure and file extensions."""
    files = list(p.iterdir())
    names = {f.name for f in files}
    extensions = {f.suffix.lower() for f in files if f.is_file()}

    # Check for COCO: annotations.json or any .json with COCO keys
    if "annotations.json" in names:
        return "coco_json"

    json_files = [f for f in files if f.suffix.lower() == ".json"]
    for jf in json_files:
        try:
            data = json.loads(jf.read_text())
            if _is_coco_json(data):
                return "coco_json"
        except (json.JSONDecodeError, OSError):
            pass

    # Check for YOLO: labels/ subdir or *.txt files with 5-column format
    if (p / "labels").is_dir():
        return "yolo"
    if _find_classes_file(p) is not None:
        return "yolo"
    txt_files = [f for f in files if f.suffix.lower() == ".txt" and f.name not in ("classes.txt", "obj.names")]
    if txt_files:
        for tf in txt_files[:3]:
            try:
                fmt = _detect_txt_format(tf)
                if fmt == "yolo":
                    return "yolo"
            except ValueError:
                pass

    # Check for VOC XML
    xml_files = [f for f in files if f.suffix.lower() == ".xml"]
    if xml_files:
        for xf in xml_files[:3]:
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(str(xf))
                if tree.getroot().tag == "annotation":
                    return "voc_xml"
            except Exception:
                pass

    # Check for LabelMe JSON
    for jf in json_files:
        try:
            data = json.loads(jf.read_text())
            if "shapes" in data:
                return "labelme_json"
        except (json.JSONDecodeError, OSError):
            pass

    # CSV fallback
    if ".csv" in extensions:
        return "csv"

    raise ValueError(f"Cannot determine annotation format for directory: {p}")


# ---------------------------------------------------------------------------
# COCO ↔ YOLO
# ---------------------------------------------------------------------------

def coco_to_yolo(input_path: str, output_dir: str, image_dir: str) -> None:
    """Convert a COCO JSON annotation file to YOLO .txt label files."""
    annotations, class_names = read_coco(input_path)
    img_dir = Path(image_dir)

    # Warn for images missing from disk
    for ann in annotations:
        img_file = img_dir / ann.image_path
        if not img_file.exists():
            console.print(f"[yellow]Warning:[/yellow] image not found on disk: {img_file}")

    write_yolo(annotations, class_names, output_dir)


def yolo_to_coco(input_dir: str, output_path: str, image_dir: str) -> None:
    """Convert a YOLO labels directory to a single COCO JSON file."""
    in_dir = Path(input_dir)

    # Determine labels directory
    labels_dir = in_dir / "labels" if (in_dir / "labels").is_dir() else in_dir

    # Find classes file
    classes_file = _find_classes_file(in_dir)
    if classes_file is None:
        raise FileNotFoundError(f"No classes.txt or obj.names found in {in_dir}")

    annotations, class_names = read_yolo(
        str(labels_dir),
        str(classes_file),
        image_dir=image_dir,
    )
    write_coco(annotations, class_names, output_path)


# ---------------------------------------------------------------------------
# YOLO ↔ VOC
# ---------------------------------------------------------------------------

def yolo_to_voc(input_dir: str, output_dir: str, image_dir: str) -> None:
    """Convert a YOLO labels directory to Pascal VOC XML files."""
    in_dir = Path(input_dir)
    labels_dir = in_dir / "labels" if (in_dir / "labels").is_dir() else in_dir

    classes_file = _find_classes_file(in_dir)
    if classes_file is None:
        raise FileNotFoundError(f"No classes.txt or obj.names found in {in_dir}")

    annotations, class_names = read_yolo(
        str(labels_dir),
        str(classes_file),
        image_dir=image_dir,
    )
    write_voc(annotations, output_dir)


def voc_to_yolo(input_dir: str, output_dir: str) -> None:
    """Convert Pascal VOC XML files to YOLO .txt label files."""
    annotations, class_names = read_voc(input_dir)
    write_yolo(annotations, class_names, output_dir)


# ---------------------------------------------------------------------------
# LabelMe → YOLO
# ---------------------------------------------------------------------------

def labelme_to_yolo(input_dir: str, output_dir: str) -> None:
    """Convert LabelMe JSON files to YOLO .txt label files."""
    annotations, class_names = read_labelme(input_dir)
    write_yolo(annotations, class_names, output_dir)


# ---------------------------------------------------------------------------
# Auto-detect and convert
# ---------------------------------------------------------------------------

def auto_detect_and_convert(input_path: str, target_format: str, output_dir: str) -> None:
    """Auto-detect annotation format and convert to the target format."""
    source_format = detect_format(input_path)

    if source_format == target_format:
        raise ValueError(f"Source and target formats are the same: {source_format}")

    _dispatch_convert(input_path, source_format, target_format, output_dir)


def _dispatch_convert(
    input_path: str,
    source_format: str,
    target_format: str,
    output_dir: str,
) -> None:
    """Dispatch to the appropriate converter based on source and target formats."""
    key = (source_format, target_format)

    if key == ("coco_json", "yolo"):
        # Need a dummy image_dir — use parent of input
        image_dir = str(Path(input_path).parent.parent / "images")
        coco_to_yolo(input_path, output_dir, image_dir)

    elif key == ("yolo", "coco_json"):
        image_dir = str(Path(input_path) / "images")
        out_file = str(Path(output_dir) / "annotations.json")
        yolo_to_coco(input_path, out_file, image_dir)

    elif key == ("yolo", "voc_xml"):
        image_dir = str(Path(input_path) / "images")
        yolo_to_voc(input_path, output_dir, image_dir)

    elif key == ("voc_xml", "yolo"):
        voc_to_yolo(input_path, output_dir)

    elif key == ("labelme_json", "yolo"):
        labelme_to_yolo(input_path, output_dir)

    else:
        raise ValueError(
            f"Conversion from '{source_format}' to '{target_format}' is not supported directly. "
            f"Supported paths: coco_json↔yolo, yolo↔voc_xml, labelme_json→yolo."
        )


# ---------------------------------------------------------------------------
# Label remapping
# ---------------------------------------------------------------------------

def convert_with_label_map(
    input_path: str,
    output_dir: str,
    label_map: Dict[str, List[str]],
    target_format: str = "yolo",
) -> None:
    """Detect format, remap labels using label_map aliases, then write target format.

    label_map maps canonical_name → list of aliases.
    Labels not in any alias list are kept as-is with a warning.
    """
    source_format = detect_format(input_path)

    # Build alias → canonical lookup
    alias_to_canonical: Dict[str, str] = {}
    for canonical, aliases in label_map.items():
        alias_to_canonical[canonical] = canonical  # canonical maps to itself
        for alias in aliases:
            alias_to_canonical[alias] = canonical

    # Read annotations depending on source format
    annotations, class_names = _read_annotations(input_path, source_format)

    # Remap labels
    remapped_class_set: List[str] = []
    for ann in annotations:
        for box in ann.boxes:
            if box.label in alias_to_canonical:
                box.label = alias_to_canonical[box.label]
            else:
                console.print(
                    f"[yellow]Warning:[/yellow] label '{box.label}' not in label_map — keeping as-is."
                )
            if box.label not in remapped_class_set:
                remapped_class_set.append(box.label)

    # Write to target format
    if target_format == "yolo":
        write_yolo(annotations, remapped_class_set, output_dir)
    elif target_format == "coco_json":
        out_file = str(Path(output_dir) / "annotations.json")
        write_coco(annotations, remapped_class_set, out_file)
    elif target_format == "voc_xml":
        write_voc(annotations, output_dir)
    else:
        raise ValueError(f"Unsupported target_format: {target_format}")


def _read_annotations(input_path: str, source_format: str):
    """Read annotations from a path given the source format."""
    if source_format == "coco_json":
        return read_coco(input_path)
    if source_format == "yolo":
        in_dir = Path(input_path)
        labels_dir = in_dir / "labels" if (in_dir / "labels").is_dir() else in_dir
        classes_file = _find_classes_file(in_dir)
        if classes_file is None:
            raise FileNotFoundError(f"No classes.txt or obj.names found in {in_dir}")
        return read_yolo(str(labels_dir), str(classes_file))
    if source_format == "voc_xml":
        return read_voc(input_path)
    if source_format == "labelme_json":
        return read_labelme(input_path)
    raise ValueError(f"Cannot read annotations for format: {source_format}")
