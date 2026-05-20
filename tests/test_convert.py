"""Tests for the opendefectkit.convert module."""
import json
import xml.etree.ElementTree as ET

import pytest

from opendefectkit.convert import (
    auto_detect_and_convert,
    coco_to_yolo,
    convert_with_label_map,
    detect_format,
    voc_to_yolo,
    yolo_to_coco,
    yolo_to_voc,
)


# ---------------------------------------------------------------------------
# detect_format tests
# ---------------------------------------------------------------------------


def test_detect_format_coco(coco_dataset, tmp_dir):
    """detect_format on a COCO JSON file returns 'coco_json'."""
    fmt = detect_format(str(coco_dataset["ann_path"]))
    assert fmt == "coco_json"


def test_detect_format_coco_directory(coco_dataset):
    """detect_format on the annotations directory containing a COCO JSON returns 'coco_json'."""
    ann_dir = coco_dataset["ann_path"].parent
    fmt = detect_format(str(ann_dir))
    assert fmt == "coco_json"


def test_detect_format_yolo(yolo_dataset):
    """detect_format on a YOLO dataset directory (with classes.txt) returns 'yolo'."""
    # The fixture root has classes.txt alongside images/ and labels/
    dataset_root = yolo_dataset["classes_path"].parent
    fmt = detect_format(str(dataset_root))
    assert fmt == "yolo"


def test_detect_format_voc(tmp_dir):
    """detect_format on a directory with Pascal VOC XML files returns 'voc_xml'."""
    # Create a minimal VOC XML
    root_el = ET.Element("annotation")
    ET.SubElement(root_el, "filename").text = "img.jpg"
    size_el = ET.SubElement(root_el, "size")
    ET.SubElement(size_el, "width").text = "640"
    ET.SubElement(size_el, "height").text = "480"
    tree = ET.ElementTree(root_el)
    tree.write(str(tmp_dir / "img.xml"), encoding="unicode")

    fmt = detect_format(str(tmp_dir))
    assert fmt == "voc_xml"


def test_detect_format_labelme(tmp_dir):
    """detect_format on a directory with LabelMe JSON files returns 'labelme_json'."""
    lm_data = {
        "imagePath": "img.jpg",
        "imageWidth": 640,
        "imageHeight": 480,
        "shapes": [],
    }
    (tmp_dir / "img.json").write_text(json.dumps(lm_data))
    fmt = detect_format(str(tmp_dir))
    assert fmt == "labelme_json"


def test_detect_format_unknown_raises(tmp_dir):
    """detect_format raises ValueError when format cannot be determined."""
    with pytest.raises(ValueError):
        detect_format(str(tmp_dir))


# ---------------------------------------------------------------------------
# coco_to_yolo tests
# ---------------------------------------------------------------------------


def test_coco_to_yolo_creates_labels(coco_dataset, tmp_dir):
    """coco_to_yolo creates a .txt file for each image and classes.txt."""
    out_dir = tmp_dir / "yolo_out"
    coco_to_yolo(
        str(coco_dataset["ann_path"]),
        str(out_dir),
        str(coco_dataset["images_dir"]),
    )
    assert (out_dir / "classes.txt").exists()
    txt_files = list(out_dir.glob("*.txt"))
    # One .txt per image + classes.txt
    image_count = len(coco_dataset["coco"]["images"])
    label_files = [f for f in txt_files if f.name != "classes.txt"]
    assert len(label_files) == image_count


def test_coco_to_yolo_bbox_normalization(coco_dataset, tmp_dir):
    """coco_to_yolo produces normalized bbox values in [0, 1]."""
    out_dir = tmp_dir / "yolo_norm"
    coco_to_yolo(
        str(coco_dataset["ann_path"]),
        str(out_dir),
        str(coco_dataset["images_dir"]),
    )
    for txt_file in out_dir.glob("*.txt"):
        if txt_file.name == "classes.txt":
            continue
        for line in txt_file.read_text().strip().splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            assert len(parts) == 5
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            assert 0.0 <= cx <= 1.0, f"cx={cx} out of range in {txt_file.name}"
            assert 0.0 <= cy <= 1.0, f"cy={cy} out of range in {txt_file.name}"
            assert 0.0 <= bw <= 1.0, f"bw={bw} out of range in {txt_file.name}"
            assert 0.0 <= bh <= 1.0, f"bh={bh} out of range in {txt_file.name}"


def test_coco_to_yolo_empty_image(tmp_dir):
    """Image with no annotations results in an empty .txt label file."""
    # COCO dataset where image 2 has no annotations
    coco = {
        "images": [
            {"id": 1, "file_name": "img_001.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "img_002.jpg", "width": 640, "height": 480},
        ],
        "categories": [{"id": 0, "name": "crack"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 0, "bbox": [10, 10, 50, 50], "area": 2500},
        ],
    }
    ann_path = tmp_dir / "ann.json"
    ann_path.write_text(json.dumps(coco))

    import numpy as np
    import cv2
    images_dir = tmp_dir / "images"
    images_dir.mkdir()
    for img_info in coco["images"]:
        img = np.full((480, 640, 3), 160, dtype=np.uint8)
        cv2.imwrite(str(images_dir / img_info["file_name"]), img)

    out_dir = tmp_dir / "yolo_empty"
    coco_to_yolo(str(ann_path), str(out_dir), str(images_dir))

    label_file = out_dir / "img_002.txt"
    assert label_file.exists()
    assert label_file.read_text().strip() == ""


# ---------------------------------------------------------------------------
# yolo_to_coco tests
# ---------------------------------------------------------------------------


def test_yolo_to_coco_roundtrip(yolo_dataset, tmp_dir):
    """yolo_to_coco produces a JSON with images, annotations, and categories keys."""
    out_file = tmp_dir / "coco_out.json"
    dataset_root = yolo_dataset["classes_path"].parent
    yolo_to_coco(
        str(dataset_root),
        str(out_file),
        str(yolo_dataset["images_dir"]),
    )
    assert out_file.exists()
    data = json.loads(out_file.read_text())
    assert "images" in data
    assert "annotations" in data
    assert "categories" in data
    assert len(data["images"]) == 3
    # Each image has 2 labels → 6 annotations total
    assert len(data["annotations"]) == 6
    category_names = {cat["name"] for cat in data["categories"]}
    assert "crack" in category_names
    assert "rust" in category_names


# ---------------------------------------------------------------------------
# yolo_to_voc tests
# ---------------------------------------------------------------------------


def test_yolo_to_voc_creates_xml(yolo_dataset, tmp_dir):
    """yolo_to_voc creates one XML file per image with <annotation> root."""
    out_dir = tmp_dir / "voc_out"
    dataset_root = yolo_dataset["classes_path"].parent
    yolo_to_voc(
        str(dataset_root),
        str(out_dir),
        str(yolo_dataset["images_dir"]),
    )
    xml_files = list(out_dir.glob("*.xml"))
    assert len(xml_files) == 3  # one per image

    for xml_file in xml_files:
        tree = ET.parse(str(xml_file))
        root = tree.getroot()
        assert root.tag == "annotation"
        assert root.find("size") is not None
        objects = root.findall("object")
        assert len(objects) == 2  # each image has 2 labels


# ---------------------------------------------------------------------------
# auto_detect_and_convert tests
# ---------------------------------------------------------------------------


def test_auto_detect_and_convert_coco_to_yolo(coco_dataset, tmp_dir):
    """auto_detect_and_convert from coco fixture produces YOLO output."""
    out_dir = tmp_dir / "auto_yolo"
    # Pass annotation file path; function will derive image_dir from parent
    auto_detect_and_convert(
        str(coco_dataset["ann_path"]),
        "yolo",
        str(out_dir),
    )
    assert (out_dir / "classes.txt").exists()
    label_files = [f for f in out_dir.glob("*.txt") if f.name != "classes.txt"]
    assert len(label_files) == 3


def test_auto_detect_same_format_raises(coco_dataset):
    """auto_detect_and_convert raises ValueError when source == target."""
    with pytest.raises(ValueError, match="same"):
        auto_detect_and_convert(str(coco_dataset["ann_path"]), "coco_json", "/tmp/out")


# ---------------------------------------------------------------------------
# convert_with_label_map tests
# ---------------------------------------------------------------------------


def test_convert_with_label_map(coco_dataset, tmp_dir):
    """convert_with_label_map remaps label aliases to canonical names in output."""
    out_dir = tmp_dir / "remapped"
    label_map = {
        "crack": ["fracture", "split"],
        "rust": ["corrosion", "oxidation"],
    }
    convert_with_label_map(
        str(coco_dataset["ann_path"]),
        str(out_dir),
        label_map=label_map,
        target_format="yolo",
    )
    classes_txt = out_dir / "classes.txt"
    assert classes_txt.exists()
    classes = classes_txt.read_text().strip().splitlines()
    # Output classes should only be canonical names
    assert "crack" in classes or "rust" in classes
    for name in classes:
        assert name in label_map, f"Unexpected class '{name}' in output classes.txt"


def test_convert_with_label_map_unknown_label_kept(tmp_dir):
    """Labels not in any alias list are kept as-is (with a warning)."""
    coco = {
        "images": [{"id": 1, "file_name": "img.jpg", "width": 640, "height": 480}],
        "categories": [{"id": 0, "name": "unknown_defect"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 0, "bbox": [10, 10, 50, 50], "area": 2500}
        ],
    }
    ann_path = tmp_dir / "ann.json"
    ann_path.write_text(json.dumps(coco))

    out_dir = tmp_dir / "remapped_unknown"
    convert_with_label_map(
        str(ann_path),
        str(out_dir),
        label_map={"crack": ["fracture"]},
        target_format="yolo",
    )
    classes_txt = out_dir / "classes.txt"
    assert classes_txt.exists()
    classes = classes_txt.read_text().strip().splitlines()
    assert "unknown_defect" in classes


# ---------------------------------------------------------------------------
# voc_to_yolo tests
# ---------------------------------------------------------------------------


def test_voc_to_yolo_roundtrip(tmp_dir):
    """voc_to_yolo reads VOC XML and produces YOLO .txt + classes.txt."""
    voc_dir = tmp_dir / "voc_in"
    voc_dir.mkdir()

    for i, label in enumerate(["crack", "rust"]):
        root_el = ET.Element("annotation")
        ET.SubElement(root_el, "filename").text = f"img_{i}.jpg"
        size_el = ET.SubElement(root_el, "size")
        ET.SubElement(size_el, "width").text = "640"
        ET.SubElement(size_el, "height").text = "480"
        obj_el = ET.SubElement(root_el, "object")
        ET.SubElement(obj_el, "name").text = label
        bndbox = ET.SubElement(obj_el, "bndbox")
        ET.SubElement(bndbox, "xmin").text = "100"
        ET.SubElement(bndbox, "ymin").text = "100"
        ET.SubElement(bndbox, "xmax").text = "200"
        ET.SubElement(bndbox, "ymax").text = "200"
        tree = ET.ElementTree(root_el)
        tree.write(str(voc_dir / f"img_{i}.xml"), encoding="unicode")

    out_dir = tmp_dir / "yolo_from_voc"
    voc_to_yolo(str(voc_dir), str(out_dir))

    assert (out_dir / "classes.txt").exists()
    label_files = [f for f in out_dir.glob("*.txt") if f.name != "classes.txt"]
    assert len(label_files) == 2
